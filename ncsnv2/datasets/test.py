import os
import torch
from torch.utils.data import DataLoader

# import your dataset class
from mri import MRIDataset, MRI_Dataset   # adjust path if needed
import torch.nn.functional as F

# ✅ path to your MRI .h5 folder
data_root = "/home/yubai402b/MZJ/csgm-mri-langevin-main/datasets/"

# ✅ dataset (no torchvision transforms)
dataset = MRI_Dataset(dir=data_root)

# ✅ dataloader
loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

# ✅ iterate through batches
print(f"Total samples: {len(dataset)}")
for i, batch in enumerate(loader):
    # resize batch to (N, C, 384, 384)
    print("Batch", i, "shape:", batch.shape)
    if i == 2:  # just show first 3 batches
        break
