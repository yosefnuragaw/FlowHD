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
CFC_COMPONENT = 4
class FLowLD(nn.Module):
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
    `n_gram`          : Use n-gram statistic on sequence
    """

    model: Centroid
    DataLoader = Iterable[Tuple[Tensor, LongTensor]]
    
    def __init__(
        self,
        n_features: int,
        n_channel:int,
        n_dimensions: int,
        n_classes: int,
        n_levels: int = 100,
        min_level: int = -1,
        max_level: int = 1,
        turbo:bool = False,
        device: torch.device = None,
        dtype: torch.dtype = None
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_channel = n_channel
        self.n_dimensions = n_dimensions
        self.n_classes = n_classes
        self.n_levels = n_levels
        self.min_level = min_level
        self.max_level = max_level
        self.turbo = turbo
        self.dtype = dtype
        self.device = device

        #IM initialization
        self.component = torchhd.embeddings.Random(
            num_embeddings=CFC_COMPONENT,
            embedding_dim= n_dimensions,
            device=device
            )
        
        self.feat = torchhd.embeddings.Random(
            num_embeddings=n_features,
            embedding_dim= n_dimensions,
            device=device
            )
        
        
        self.value = torchhd.embeddings.Level(
            num_embeddings=n_levels,
            embedding_dim=n_dimensions,
            device=device,
            low=min_level,
            high=max_level
            )


        #AM intialization
        self.model = Centroid(
            in_features = n_dimensions*CFC_COMPONENT,
            out_features=n_classes,
            device=device,
            dtype=dtype
        )


    def encoder(
            self,
            samples: Tensor,
            timespans = None
        )-> Tuple[Tensor, Tensor]:

        batch,seq,feat = samples.shape[0],samples.shape[1],samples.shape[2]

        if self.turbo:
            samples = torchhd.hash_table(
            self.value(samples.view(batch,seq,self.n_channel,self.n_features)),
            self.feat.weight
            )
        else:
            logits = []
            for x in range(batch):
                logits.append(torchhd.multiset(self.value(samples[x].view(seq,self.n_channel,self.n_features)).mul_(self.feat.weight)))
            samples = torch.stack(logits)
            
        
        samples = samples.unsqueeze(3).repeat_interleave(4, dim=3)
        samples.mul_(self.component.weight)
        samples = torchhd.multiset(
            samples.view(batch,seq,4,self.n_channel,-1)
            )
        
        t_interp = torch.sigmoid(
            samples[:, :, 2, :].add(samples[:, :, 3, :])
            )
        h_states =samples[:, :, 0, :].mul(1-t_interp).add_(t_interp.mul_(samples[:, :, 1, :]))
        h_states = h_states.roll(shifts=1, dims=0)
        h_states[0] = torch.zeros_like(h_states[0])


  
        samples.add_(
            h_states.unsqueeze(2).repeat_interleave(4, dim=2)
            )
        return torchhd.bundle_sequence(samples.view(batch,seq,-1)).sign()
 
            
    
    def fit(
            self,
            data_loader: DataLoader,
            num_epochs:int = 1,
            val_data_loader: DataLoader = None
        )->list:
        
        best_val= float('-inf')
        voltages = []
        for epoch in range(num_epochs):
            pbar = tqdm(enumerate(data_loader), total=len(data_loader))
            c,t = 0,0
            for i, data in pbar:
                samples,labels = data
                samples = samples.to(self.device)
                labels = labels.to(self.device)

                encoded = self.encoder(samples)
                outputs = self.model(encoded)         
                c += (torch.max(outputs, 1)[1] == labels).sum().item() 
                t += labels.size(0)

                self.model.add(encoded, labels)
                if val_data_loader != None:
                    pbar.set_description(f"Epoch {epoch+1}/{num_epochs},Accuracy: {100*c/t:5f}, Best test{(best_val):2f}")
                else:
                    pbar.set_description(f"Epoch {epoch+1}/{num_epochs},Accuracy: {100*c/t:5f}")

            if val_data_loader != None:
                val_acc,voltage = self.test(val_data_loader)
                voltages.append(voltage)
                if val_acc > best_val:
                    best_val = val_acc
        return voltages
    
    def test(self, data_loader: DataLoader) -> Tuple:
        c,l = 0,0
        v=[]
        t = len(data_loader)
        pbar = tqdm(enumerate(data_loader), total=t)
        for i, data in pbar:
            samples,labels = data
            samples,labels = samples.to(self.device),labels.to(self.device)
            self.meter.start(tag='_test_step')
            encoded = self.encoder(samples)
            outputs = self.model(encoded)
            self.meter.stop()
            trace = self.meter.get_trace()
            for sample in trace:
                for tag in sample.energy:
                    v.append(sample.energy[tag]) 

            loss = self.criterion(outputs, labels)
            l += loss.item()
            c += (torch.max(outputs, 1)[1] == labels).sum().item() 
        
        pbar.set_description(f"Testing Loss: {(l/(i+1)):5f} Testing Accuracy: {sum(v)/len(v):5f}")
        print(f"Energy Usage on {tag} : {sum(v)/len(v)}")
        return 100*(c/t),sum(v)/len(v)

    
    def init_self(self):
        return FLowLD(
                n_features=self.n_features,
                n_channel=self.n_channel,
                n_dimensions=self.n_dimensions,
                n_classes=self.n_classes,
                n_levels=self.n_levels,
                device=self.device,
                min_level=self.min_level,
                max_level=self.max_level,
            )
    
    def add_online(self, input: Tensor, target: Tensor, lr: float = 1.0) -> None:
        r"""Only updates the prototype vectors on wrongly predicted inputs.

        Implements the iterative training method as described in `OnlineHD: Robust, Efficient, and Single-Pass Online Learning Using Hyperdimensional System <https://ieeexplore.ieee.org/abstract/document/9474107>`_.

        Adds the input to the mispredicted class prototype scaled by :math:`\epsilon - 1`
        and adds the input to the target prototype scaled by :math:`1 - \delta`,
        where :math:`\epsilon` is the cosine similarity of the input with the mispredicted class prototype
        and :math:`\delta` is the cosine similarity of the input with the target class prototype.
        """
        # Adapted from: https://gitlab.com/biaslab/onlinehd/-/blob/master/onlinehd/onlinehd.py
        # credit : TorchHD
        
        logit = self.model(input)
        sc,pred = torch.max(logit, 1)
        is_wrong = target != pred
        wrong_not_overfit = sc[is_wrong] < 0.9
        weak_conf = sc[target == pred] <= 0


        # cancel update if all predictions were correct
        if is_wrong.sum().item() == 0:
            return

         # Add if similarity negative
        if weak_conf.sum().item()!=0:
            self.model.weight.index_add_(0, target[target == pred][weak_conf], input[target == pred][weak_conf], alpha=lr)
   
        input = input[is_wrong]
        input_t = input[wrong_not_overfit]
        target = target[is_wrong]
        target_w = target[wrong_not_overfit]
        pred = pred[is_wrong]

        self.model.weight.index_add_(0, target_w, input_t, alpha=lr)
        self.model.weight.index_add_(0, pred, input, alpha=-lr)

        return logit
    
    def LOOCV(
            self,
            data_loader: DataLoader,
            num_epochs:int = 20,
            batch_size:int = 16,
            wait:int = 5,
            )->None:
        # baseline = get_baseline(0)
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
                for i in range(0, len(train_dataset), batch_size):
                    batch_indices = range(i, min(i + batch_size, len(train_dataset)))
                    train_data,train_target = torch.stack([train_dataset[idx][0] for idx in batch_indices]).to(self.device),torch.stack([train_dataset[idx][1] for idx in batch_indices]).to(self.device)
    
                    encoded = i_model.encoder(train_data)
                    i_model.add_online(encoded,train_target,0.1)
                    

                #Validation
                
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
                        pbar.set_postfix({"Accuracy": sum(a_list) / len(a_list),"F1": f1,"AVG Comp Cost (J)":sum(joules) / len(joules),"AVG Latency (ms)":np.sum(timings) / s*1000})
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
    

                
           

