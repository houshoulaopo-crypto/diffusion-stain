import h5py
import matplotlib.pyplot as plt
import torch

# Path to one of your MRI .h5 files
path = "/home/yubai402b/MZJ/csgm-mri-langevin-main/datasets/brain_T2/file_brain_AXT2_200_2000019.h5"
with h5py.File(path, "r") as f:
    rss = f["reconstruction_rss"][()]   # shape (16, 384, 384)

print("RSS shape:", rss.shape)

# Pick a slice
slice_idx = 5
img = rss[slice_idx]

# Convert to torch and normalize to [0,1] for display
img_tensor = torch.from_numpy(img)
img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())

# Plot
plt.imshow(img_tensor.numpy(), cmap="gray")
plt.title(f"Reconstruction RSS slice {slice_idx}")
plt.axis("off")
plt.show()
