import io
from sklearn.metrics import f1_score
import numpy as np
from typing import Optional, Literal, Callable, Iterable, Tuple
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import Tensor, LongTensor
import torchhd
from torchhd.embeddings import Random, Level, Projection, Sinusoid, Density
import torchhd.functional as functional
from torchhd.models import Centroid
from pynvml import *


class FLowHD(nn.Module):
    """ 
    FlowHD 
    Implement modified OnlineHD using FeatAppend encoding method

    `n_features`      : num features on each chanel
    `n_channel`       : num channels used
    `n_dimensions`    : Hv dimension
    `n_classes`       : num classes
    `n_levels`        : num Hv for value
    `min_level`       : Low limit value
    `max_level`       : Max limit value
    'type'            : None is FlowLD ,plus is FlowLD+, fast is FastFlowLD+
    """

    model: Centroid
    DataLoader = Iterable[Tuple[Tensor, LongTensor]]
    
    def __init__(
        self,
        n_features: int,
        n_dimensions: int,
        n_classes: int,
        type: str = None,
        device: torch.device = None,
        dtype: torch.dtype = None
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_dimensions = n_dimensions
        self.n_classes = n_classes
        self.types = type
        self.dtype = dtype
        self.device = device
        self.encode = Sinusoid(n_features, n_dimensions, device=device, dtype=dtype)
        self.model = Centroid(
            in_features = n_dimensions,
            out_features=n_classes,
            device=device,
            dtype=dtype
        )


    def encoder(
            self,
            samples: Tensor,
        )-> Tensor:
        DIM = 2
        samples = self.encode(samples)
        n = samples.size(DIM)
        enum = enumerate(torch.unbind(samples, DIM))
        permuted = [torch.roll(hv, shifts=int(n - i - 1), dims=-1) for i, hv in enum]
        permuted = torchhd.multiset(torch.stack(permuted, DIM))
        return torchhd.soft_quantize(permuted).to(self.device)
            
    def forward(self,samples):
        encoded = self.encoder(samples)
        return torchhd.cosine_similarity(encoded,self.model.weight)
    
    def fit(self, x,y):
        samples = x.to(self.device)
        labels = y.to(self.device)

        encoded = self.encoder(samples)
        logit = self.model(encoded)
        sc,pred = torch.max(logit, 1)
        is_wrong = labels != pred
        wrong_not_overfit = sc[is_wrong] < 0.9
        weak_conf = sc[labels == pred] <= 0

        lc = logit[torch.arange(len(labels)), labels]  
        a = sc - lc  

        if is_wrong.sum().item() == 0:
            return

         # Syarat 2
        if weak_conf.sum().item()!=0 and self.types != None:
            self.model.weight.index_add_(0, labels[labels == pred][weak_conf], encoded[labels == pred][weak_conf], alpha=0.1)
   
        input = encoded[is_wrong]
        input_t = input[wrong_not_overfit]
        labels = labels[is_wrong]
        labels_w = labels[wrong_not_overfit]
        pred = pred[is_wrong]

        self.model.weight.index_add_(0, labels_w, input_t, alpha=0.1)
        self.model.weight.index_add_(0, pred, input, alpha=-0.1)
        return self
    
    def reset(self):
        self.model = Centroid(
            in_features = self.n_dimensions,
            out_features=self.n_classes,
            device=self.device,
            dtype=self.dtype
        )
    
    def add_plus(self, input: Tensor, target: Tensor) -> None:
        logit = self.model(input)
        sc,pred = torch.max(logit, 1)
        is_wrong = target != pred
        wrong_not_overfit = sc[is_wrong] < 0.9
        weak_conf = sc[target == pred] <= 0

        lc = logit[torch.arange(len(target)), target]  
        a = sc - lc  

        if is_wrong.sum().item() == 0:
            return

         # Syarat 2
        if weak_conf.sum().item()!=0 and self.types != None:
            self.model.weight.index_add_(0, target[target == pred][weak_conf], input[target == pred][weak_conf], alpha=-1 * sc[target == pred][weak_conf])
   
        input = input[is_wrong]
        input_t = input[wrong_not_overfit]
        target = target[is_wrong]
        target_w = target[wrong_not_overfit]
        pred = pred[is_wrong]

        self.model.weight.index_add_(0, target_w, input_t, alpha=a)
        self.model.weight.index_add_(0, pred, input, alpha=-a)

    
    def LOOCV(
            self,
            data_loader: DataLoader,
            num_epochs:int = 20,
            batch_size:int = 16,
            wait:int = 5,
            )->None:
        a_list,p_list,l_list,joules = [],[],[],[]
        f1 = float('-inf')
        s = len(data_loader.dataset)

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        timings=np.zeros((s,1))


        #GPU WARMUP  
        dummy = torch.randn(tuple(data_loader.dataset.tensors[0][:64,:,:].shape)).to(self.device)
        warmup = nn.Linear(dummy.size(-1), 256).to(self.device)
        for _ in range(100):
             warmup(dummy)

        pbar = tqdm(enumerate(data_loader),total=len(data_loader.dataset), desc="LOOCV Progress")
        for i, (d, t) in pbar:
            t_ind = list(range(s))
            del t_ind[i] 

            i_model = self.init_self()
            train_dataset = torch.utils.data.Subset(data_loader.dataset, t_ind)
            val_data,val_target=d.to(self.device),t.to(self.device)
            count = 0
            for epoch in range(num_epochs):
                for st in range(0, len(train_dataset), batch_size):
                    batch_indices = range(st, min(st + batch_size, len(train_dataset)))
                    train_data,train_target = torch.stack([train_dataset[idx][0] for idx in batch_indices]).to(self.device),torch.stack([train_dataset[idx][1] for idx in batch_indices]).to(self.device)
    
                    encoded = i_model.encoder(train_data)
                    i_model.add_plus(encoded,train_target,0.1)

                del encoded,train_data,train_target,batch_indices
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

                #Validation
                with torch.no_grad():
                    self.nvml_handle = nvmlInit()
                    handle =  nvmlDeviceGetHandleByIndex(0)
                    starter.record()
                    encoded = i_model.encoder(val_data)
                    outputs = i_model.model(encoded)
                    ender.record()
                    torch.cuda.synchronize()
                    curr_time = starter.elapsed_time(ender)
                    timings[i] = curr_time
                    power_usage = (nvmlDeviceGetPowerUsage(handle) / 1000)
                    nvmlShutdown()
                total_energy_consumed = power_usage
                pred = torch.max(outputs, 1)[1]
                accuracy = (pred == val_target).sum().item() 
                
                if len(a_list)!= 0:
                        pbar.set_description(f"LOOCV Epoch {epoch}/{num_epochs}| Early {count}/{wait} ")
                        pbar.set_postfix({"Accuracy": sum(a_list) / len(a_list),"F1": f1,"AVG Comp Cost (J)":sum(joules) / len(joules),"AVG Latency (ms)":np.sum(timings)/i,"STD Latency (ms)":np.std(timings)})
                else:
                    pbar.set_description(f"LOOCV | Epoch {epoch}| Early {count}/{wait} ")

                if accuracy == 1:
                    joules.append(total_energy_consumed) 

                    a_list.append(accuracy)
                    p_list.append(pred.cpu().numpy())
                    l_list.append(val_target.cpu().numpy())
                    f1 = f1_score(l_list, p_list, average='macro')
                    break
                else:
                    if count >= wait:
                       
                        joules.append(total_energy_consumed) 

                        a_list.append(0)
                        p_list.append(pred.cpu().numpy())
                        l_list.append(val_target.cpu().numpy())
                        f1 = f1_score(l_list, p_list, average='macro')
                        break
                    else:
                        count+=1
                        
    def get_size(self)->float:
        model = self.cpu()
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        size_bytes = buffer.tell()
        size_mb = size_bytes / (1024 * 1024)
        return size_mb