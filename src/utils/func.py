import torch
import io 
from sklearn.metrics import f1_score
from typing import  Iterable, Tuple
import torch
import torch.nn as nn
from pynvml import *
from tqdm import tqdm
from torch import Tensor, LongTensor
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader as TensorDataLoader
import numpy as np

DataLoader = Iterable[Tuple[Tensor, LongTensor]]


def remove_inf(tensor, replace_value=1e-6):
    # Replace -inf and inf with replace_value
    tensor[tensor == float('-inf')] = replace_value
    tensor[tensor == float('inf')] = replace_value
    return tensor

def prep_x(x_i):
    MIN = 78 #By smallest observed sequence (CHANGE ACCORDING DATA)
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

def KFoldCV_HD(
        model,
        data_loader: DataLoader,
        num_folds: int = 5,
        num_epochs: int = 20,
        batch_size: int = 16,
        wait: int = 5,
        device: str = 'cpu'
) -> None:
    joules = []
    f1 = float('-inf')
    dataset = data_loader.dataset
    targets = np.array([dataset[i][1].item() for i in range(len(dataset))])
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
   

    # GPU WARMUP  
    dummy = torch.randn(tuple(dataset.tensors[0][:64, :, :].shape)).to(device)
    warmup = nn.Linear(dummy.size(-1), 256).to(device)
    for _ in range(100):
        warmup(dummy)
    epoch_list = [0 for x in range(num_folds)]
    
    pbar = tqdm(total=num_folds, desc="K-Fold Progress")
    for fold, (train_idx, val_idx) in enumerate(skf.split(dataset, targets)):
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        
        val_data = torch.stack([dataset[i][0] for i in val_idx]).to(device)
        val_target = torch.stack([dataset[i][1] for i in val_idx]).to(device)

        model.reset()

        count = 0
        for epoch in range(num_epochs):
            for index in range(0, len(train_subset), batch_size):
                batch_indices = range(index, min(index + batch_size, len(train_subset)))
                train_data = torch.stack([train_subset[idx][0] for idx in batch_indices]).to(device)
                train_target = torch.stack([train_subset[idx][1] for idx in batch_indices]).to(device)
                

                model.fit(train_data, train_target)
                
            # Validation
            nvml_handle = nvmlInit()
            handle = nvmlDeviceGetHandleByIndex(0)
            starter.record()
            encoded = model.encoder(val_data)
            outputs = model.model(encoded)
            ender.record()
            torch.cuda.synchronize()
            power_usage = (nvmlDeviceGetPowerUsage(handle) / 1000)
            nvmlShutdown()
            
            total_energy_consumed = power_usage
            pred = torch.max(outputs, 1)[1]
            f1 = f1_score(val_target.cpu().numpy(), pred.cpu().numpy(), average='micro')*100
            if f1 > epoch_list[fold]:
                epoch_list[fold] = f1
                joules.append(total_energy_consumed)
                count=0
            else:
                count+=1
            if count == wait:
                break
            pbar.set_description(f"Fold {fold+1}/{num_folds}| Epoch: {epoch}/{num_epochs} |Completed | F1: {sum(epoch_list) / (fold+1)}")
        pbar.update(1)

def prep_dataset(dir):
    cv_dataloader = torch.load(dir)
    c_x = []
    c_y = []
    for x,y in cv_dataloader:
        c_x.append(remove_inf(x))
        c_y.append(y)


    filtered_dataset = TensorDataset(torch.cat(c_x), torch.cat(c_y))
    return TensorDataLoader(filtered_dataset, batch_size=1)