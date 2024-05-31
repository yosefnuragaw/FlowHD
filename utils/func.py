import torch
import io 
from sklearn.metrics import f1_score
from typing import Optional, Literal, Callable, Iterable, Tuple
import torch
import torch.nn as nn
from pynvml import *
from tqdm import tqdm
from torch import Tensor, LongTensor

import numpy as np
import torch.optim as optim

DataLoader = Iterable[Tuple[Tensor, LongTensor]]


def remove_inf(tensor, replace_value=1e-6):
    # Replace -inf and inf with replace_value
    tensor[tensor == float('-inf')] = replace_value
    tensor[tensor == float('inf')] = replace_value
    return tensor

def prep_x(x_i):
    MIN = 78 #By smallers observed sequence (CHANGE ACCORDING DATA)
    INDEX = 'FIRST'

    arr = "empty"
    for x in x_i:
        if INDEX == "FIRST":
            INDEX = "SECOND"
            arr = np.transpose(np.vstack(x))[:MIN]
            print(arr.shape)
        elif INDEX =="SECOND":
            INDEX = "LAST"
            arr = np.stack((arr,np.transpose(np.vstack(x))[:MIN]))
        else:
            temp = np.expand_dims(np.transpose(np.vstack(x))[:MIN], axis=0)
            arr = np.concatenate((arr, temp), axis=0)
    return arr

def get_size(model)->float:
        model = model.cpu()
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        size_bytes = buffer.tell()
        size_mb = size_bytes / (1024 * 1024)
        return size_mb


def LOOCV(
        model,
        data_loader: DataLoader,
        num_epochs:int = 20,
        batch_size:int = 16,
        wait:int = 5,
        device:str='cpu'
        )->None:
    # baseline = get_baseline(0)
    a_list,p_list,l_list,joules = [],[],[],[]
    f1 = float('-inf')
    s = len(data_loader.dataset)

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings=np.zeros((s,1))


    #GPU WARMUP  
    dummy = torch.randn(tuple(data_loader.dataset.tensors[0][:64,:,:].shape)).to(device)
    warmup = nn.Linear(dummy.size(-1), 256).to(device)
    for _ in range(100):
            warmup(dummy)
 

    pbar = tqdm(enumerate(data_loader),total=len(data_loader.dataset), desc="LOOCV Progress")
    for i, (d, t) in pbar:
        t_ind = list(range(s))
        del t_ind[i] 

        i_model = model
        i_model.to(device)
        train_dataset = torch.utils.data.Subset(data_loader.dataset, t_ind)
        val_data,val_target=d.to(device),t.to(device)
        count = 0
        for epoch in range(num_epochs):
            for i in range(0, len(train_dataset), batch_size):
                batch_indices = range(i, min(i + batch_size, len(train_dataset)))
                train_data,train_target = torch.stack([train_dataset[idx][0] for idx in batch_indices]).to(device),torch.stack([train_dataset[idx][1] for idx in batch_indices]).to(device)

                i_model.fit(train_data,train_target)
                # outputs = i_model.model(encoded)
       
                

            #Validation
            
            nvml_handle = nvmlInit()
            handle =  nvmlDeviceGetHandleByIndex(0)
            starter.record()
            encoded = i_model.encoder(val_data)
            outputs = i_model(encoded)
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