
import torch
import io 
from sklearn.metrics import f1_score
from typing import Optional, Literal, Callable, Iterable, Tuple
import torch
import torch.nn as nn
from pynvml import *
from tqdm import tqdm
from torch import Tensor, LongTensor
from ncps.wirings import AutoNCP
from ncps.torch import CfC
import numpy as np
import torch.optim as optim

DataLoader = Iterable[Tuple[Tensor, LongTensor]]



class CFC(nn.Module):
    def __init__(
            self,
            input_size :int,
            n_classes :int,
            units :int,
            proj_size :int,
            return_sequence:bool = False,
            mixed_memory: bool = False,
            device :torch.device = 'cpu',
            ):
        
        super(CFC, self).__init__()
        self.input_size = input_size
        self.n_classes = n_classes
        self.units = units
        self.proj_size = proj_size
        self.return_sequence = return_sequence
        self.mixed_memory = mixed_memory
        self.device = device


        self.cfc = CfC(
            input_size,
            units=units,
            proj_size=proj_size,
            return_sequences = return_sequence,
            mixed_memory= mixed_memory
            )
        self.W_out = nn.Linear(proj_size, n_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = torch.nn.Softmax(dim=1)
        self.optimizer = torch.optim.AdamW(self.parameters(),weight_decay=0.04)

        
    def forward(
            self,
            input:Tensor
            ) -> Tensor:
        
        l,hc = self.cfc(input)
        return self.W_out(l),hc
         
    
    def train_model(
            self,
            train_loader:DataLoader,
            test_loader:DataLoader = None,
            num_epochs:int=10,
            test_mode:bool = False,
            show_bar:bool = True,
            )->None:
        
        b_loss = float("inf")
        j_list = []
        for epoch in range(num_epochs):
            c,t = 0,0
            r_loss = 0.0
            if show_bar:
                pbar = tqdm(enumerate(train_loader), total=len(train_loader))
            else:
                pbar = enumerate(train_loader)

            for i, data in pbar:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                self.optimizer.zero_grad()
                outputs,__ = self(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                c += (torch.max(outputs, 1)[1] == labels).sum().item()
                r_loss += loss.item()
                t += labels.size(0)
                if show_bar:
                    pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {r_loss / (i+1):.5f}, Accuracy: {100*c/t:5f}, Best test{(b_loss):2f}")

            if test_mode:
                t_loss,j = self.test_model(test_loader)
                j_list.append(j)

            if loss.item() < b_loss:
                b_loss = loss.item()
                torch.save(self.state_dict(), 'best_cfc_config.pth')

    def test_model(
            self,
            test_loader:DataLoader,
            )->Tuple:
        j = 0
        with torch.no_grad():
            self.meter.start(tag='_test_step')
            accuracy = self._test_step(test_loader)
            self.meter.stop()
        
        trace = self.meter.get_trace()
        print(f'Accuracy of the network on the test images: {(100*accuracy):2f}')
        

        for sample in trace:
            for tag in sample.energy:
                print(f"Energy Usage on {tag} : {(sample.energy[tag] / 1000) / sample.duration} Mw")
        return accuracy,(sample.energy[tag] / 1000) / sample.duration

    def _test_step(
            self,
            test_loader:DataLoader,
        )->float:
        correct = 0
        total = 0
        for data in test_loader:
            images, labels = data
            images, labels = images.to(self.device), labels.to(self.device)
            outputs,__ = self(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return self.criterion(outputs, labels)

    def init_self(self):
        return CFC(
            input_size = self.input_size,
            n_classes =self.n_classes,
            units =self.units,
            proj_size =self.proj_size,
            return_sequence = self.return_sequence,
            mixed_memory = self.mixed_memory,
            device =self.device,
        )


    def LOOCV(
            self,
            data_loader: DataLoader,
            num_epochs:int = 20,
            batch_size:int = 16,
            wait:int = 5,
            )->None:
        
        a_list,p_list,l_list,joules,times = [],[],[],[],[]
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
            i_model.to(device=self.device)
            optimizer = optim.AdamW(i_model.parameters(),weight_decay=0.04,lr=0.01)
            train_dataset = torch.utils.data.Subset(data_loader.dataset, t_ind)
            val_data,val_target=d.to(self.device),t.to(self.device)
            count = 0
            for epoch in range(num_epochs):
                for i in range(0, len(train_dataset), batch_size):
                    batch_indices = range(i, min(i + batch_size, len(train_dataset)))
                    train_data,train_target = torch.stack([train_dataset[idx][0] for idx in batch_indices]).to(self.device),torch.stack([train_dataset[idx][1] for idx in batch_indices]).to(self.device)
    
                    train_data = train_data.to(self.device)
                    train_target = train_target.to(self.device)
              
                    optimizer.zero_grad()
                    outputs,__ = i_model(train_data)
                    
                    loss = i_model.criterion(outputs, train_target)
                    loss.backward()
                    optimizer.step()
                    pbar.set_description(f"LOOCV Epoch {epoch}/{num_epochs}| Early {count}/{wait} Loss| {loss.item()}")

                #Validation
                with torch.no_grad():
                    nvml_handle = nvmlInit()
                    handle =  nvmlDeviceGetHandleByIndex(0)
                    starter.record()
                    outputs,__ = i_model(val_data)
                    ender.record()
                    torch.cuda.synchronize()
                    curr_time = starter.elapsed_time(ender)
                    timings[i] = curr_time
                    power_usage = (nvmlDeviceGetPowerUsage(handle) / 1000)
                    
                    nvmlShutdown()
                    pred = torch.max(self.softmax(outputs), 1)[1]
                    accuracy = (pred == val_target).sum().item() 
                
                    if len(a_list)!= 0: 

                        pbar.set_postfix({"Accuracy": sum(a_list) / len(a_list),"F1": f1,"AVG Comp Cost (J)":sum(joules) / len(joules),"AVG Latency (ms)":np.sum(timings) / s*1000})
                    else:
                        pbar.set_description(f"LOOCV | Epoch {epoch}| Early {count}/{wait} ")

                    if accuracy == 1:
                 
                        joules.append(power_usage) 

                        a_list.append(accuracy)
                        p_list.append(pred.cpu().numpy())
                        l_list.append(val_target.cpu().numpy())
                        f1 = f1_score(l_list, p_list, average='macro')
                        break
                    else:
                        if count >= wait:
                        
                            joules.append(power_usage) 
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
