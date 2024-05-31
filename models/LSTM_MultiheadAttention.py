
import torch
import io 
from sklearn.metrics import f1_score
from typing import Optional, Literal, Callable, Iterable, Tuple
import torch
import torch.nn as nn
from pynvml import *
from tqdm import tqdm
from torch import Tensor, LongTensor
import math
import numpy as np
import torch.optim as optim

DataLoader = Iterable[Tuple[Tensor, LongTensor]]

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.fc_q = nn.Linear(hidden_size, hidden_size)
        self.fc_k = nn.Linear(hidden_size, hidden_size)
        self.fc_v = nn.Linear(hidden_size, hidden_size)
        self.fc_o = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        Q = self.fc_q(x)
        K = self.fc_k(x)
        V = self.fc_v(x)

        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attention = torch.softmax(Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim), dim=-1)
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e9)
        x = self.dropout(torch.matmul(attention, V))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hidden_size)
        x = self.fc_o(x)
        return x
    
    def get_size(self)->float:
        model = self.cpu()
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        size_bytes = buffer.tell()
        size_mb = size_bytes / (1024 * 1024)
        return size_mb
    
class LSTMClassifier(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_class: int,
            num_layers: int = 1,
            attention:bool = False,
            num_attention:int = 16,
            device: str = 'cpu'
            ):
        super(LSTMClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_class = num_class
        self.num_layers = num_layers
        self.attention = attention
        self.num_attention = num_attention
        
        self.lstm = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first=True
            )
        self.fc = nn.Linear(
            in_features=hidden_size,
            out_features=num_class)
        
        if attention:
            self.multihead =MultiHeadAttention(
            hidden_size, 
            num_heads=num_attention)

        self.softmax = torch.nn.Softmax(dim=1)
        self.criterion = nn.CrossEntropyLoss()
        self.device = device

       

    def forward(
            self,
            x: Tensor
            ) -> Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        
        out, __ = self.lstm(x, (h0, c0)) 
        if self.attention:
            for x in range(4):
                out = self.multihead(out)
        out = self.fc(out[:, -1, :])

        
        return out
    
    def train_model(
            self,
            train_loader: DataLoader,
            test_loader: DataLoader,
            num_epochs: int = 10,
            optimizer=None
            ) -> None:
        best_test_accur = -1
        voltage_list = []
        for epoch in range(num_epochs):
            total = 0
            running_loss = 0.0
            correct = 0
            pbar = tqdm(enumerate(train_loader), total=len(train_loader))

            for i, data in pbar:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()

                outputs,__ = self(inputs)
                loss = self.L2(outputs, labels,0.017)
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(self.softmax(outputs), 1)
                correct += (predicted == labels).sum().item()
                running_loss += loss.item()
                total += labels.size(0)
                pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / (i+1):.5f}, Accuracy: {100*correct/total:5f}, Best test: {(100*best_test_accur):2f}")
            test_accur, voltage = self.test_model(test_loader)
            voltage_list.append(voltage)
            if test_accur >= best_test_accur:
                best_test_accur = test_accur
                torch.save(self.state_dict(), 'best_lstm_config.pth')

        print('Finished Training')
        return voltage_list

    def test_model(
            self,
            test_loader: DataLoader,
    ) -> Tuple:
        with torch.no_grad():
            self.meter.start(tag='_test_step')
            accuracy = self._test_step(test_loader)
            self.meter.stop()

        trace = self.meter.get_trace()
        print(f'Accuracy of the network on the test images: {(100*accuracy):2f}')

        voltage = 0
        for sample in trace:
            for tag in sample.energy:
                voltage = (sample.energy[tag] / 1000) / sample.duration
                print(f"Energy Usage on {tag} : {(sample.energy[tag] / 1000) / sample.duration} Mw")
        return accuracy, voltage

    def _test_step(
            self,
            test_loader: DataLoader,
    ) -> int:
        correct = 0
        total = 0
        for data in test_loader:
            images, labels = data
            images, labels = images.to(self.device), labels.to(self.device)
            outputs,__ = self(images)
            _, predicted = torch.max(self.softmax(outputs), 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return correct / total
    
    def init_self(self):
        return LSTMClassifier(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_class=self.num_class,
            num_layers=self.num_layers,
            attention=self.attention,
            num_attention=self.num_attention,
            device=self.device)
        

   

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
            optimizer = optim.AdamW(i_model.parameters(),lr=0.05,weight_decay=0.001)
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
                    outputs = i_model(train_data)
                    
                    loss = i_model.criterion(outputs, train_target)
                    loss.backward()
                    optimizer.step()

                #Validation
                with torch.no_grad():
                    nvml_handle = nvmlInit()
                    handle =  nvmlDeviceGetHandleByIndex(0)
                    starter.record()
                    outputs = i_model(val_data)
                    pred = torch.max(self.softmax(outputs), 1)[1]
                    ender.record()
                    torch.cuda.synchronize()
                    curr_time = starter.elapsed_time(ender)
                    timings[i] = curr_time
                    power_usage = (nvmlDeviceGetPowerUsage(handle) / 1000)
                    
                    nvmlShutdown()
                   
                    accuracy = (pred == val_target).sum().item() 
                
                    if len(a_list)!= 0: 
                        pbar.set_description(f"LOOCV Epoch {epoch}/{num_epochs}| Early {count}/{wait} ")
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