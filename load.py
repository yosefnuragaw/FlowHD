from src.utils.Dataset import KaraOneDataset
from src.utils.func import prep_x
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch

dataset_dir = "KaraOne"
participants = ["MM05", "MM08", "MM09", "MM10", "MM11", "MM12", "MM15", "MM16", "MM18", "MM19", "MM20", "MM21", "P02"]

loaders = [KaraOneDataset(dataset_dir, pts=(pt,), start_idx=0, end_idx=200, n_mel_channels=32, eeg_types=["imagined"], raw_only=False) for pt in participants]
data = [loader.get_data() for loader in loaders]
x_data, y_data = zip(*data)

X = torch.tensor(
    np.concatenate([prep_x(x) for x in x_data], axis=0),
    dtype=torch.float32
    )
Y = torch.tensor(
    np.hstack(y_data), 
    dtype=torch.long
    )

cv_dataloader = DataLoader(TensorDataset(X, Y), shuffle=True)
torch.save(cv_dataloader, 'FFA_CV.pth')
