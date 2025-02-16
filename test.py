import os
import random
import warnings
import torch
import numpy as np
from src.utils.func import prep_dataset, get_size, KFoldCV_HD
import torchprofile
from src.models.FlowHD import FLowHD

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# Set seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)  # For multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load dataset
cvc_dataloader = prep_dataset('FFA_CV.pth')

# Initialize model
model = FLowHD(n_features=170, n_dimensions=256, n_classes=11, device=device)
print(f"AdaptHD size {get_size(model)}")
model.to(device)

# Measure MACs
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
starter.record()
macs = torchprofile.profile_macs(model, cvc_dataloader.dataset[0][0].unsqueeze(0).to(device))
ender.record()
torch.cuda.synchronize()
print(f"Elapsed time: {starter.elapsed_time(ender)} ms")

gmacs = macs / 1e9
print(f"MACs: {macs}")
print(f"GMACs: {gmacs}")

# Calculate FLOPs
flops = macs * 2
gflops = flops / 1e9
print(f"FLOPs: {flops}")
print(f"GFLOPs: {gflops}")

# Run K-Fold Cross-Validation
warnings.filterwarnings("ignore")
KFoldCV_HD(model, cvc_dataloader, k_folds=10, epochs=20, batch_size=16, patience=5, device=device)
torch.cuda.empty_cache()
warnings.filterwarnings("always")
