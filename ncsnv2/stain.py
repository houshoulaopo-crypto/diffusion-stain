import os

import torch

from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.utils as vutils

@torch.no_grad()
def ncsn_img2img(model, x0, sigmas, start_level, n_steps_each=1, step_size=0.00002):
    """
    model: trained NCSN model (score network)
    x0: input image tensor (B, C, H, W), normalized [-1,1]
    sigmas: list/array of noise levels (descending)
    start_level: index in sigmas to begin sampling
    n_steps_each: Langevin steps per sigma
    step_size: base step size for Langevin updates
    """
    # 1. Add noise to input image at start_level
    sigma = sigmas[start_level]
    noise = torch.randn_like(x0) * sigma
    x = x0 + noise

    # 2. Reverse process: go down the sigma schedule
    for c, sigma in enumerate(sigmas[:start_level][::-1]):
        for step in range(n_steps_each):
            grad = model(x, torch.tensor([c], device=x.device))  # score function
            step_size_scaled = step_size * (sigma / sigmas[start_level])**2
            x = x + step_size_scaled * grad
            x = x + torch.sqrt(2 * step_size_scaled) * torch.randn_like(x)
    return x


def Init_sample(path, image_size=256, device="cuda"):
    """
    Load an image as initial sample for diffusion (NCSN).

    Args:
        path (str): path to image file
        image_size (int): target resize
        device (str): "cuda" or "cpu"

    Returns:
        torch.Tensor: shape (1, C, H, W), normalized to [-1,1]
    """
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),  # scales to [0,1]
        T.Normalize(mean=[0.5], std=[0.5])  # -> [-1,1], extend for RGB
    ])

    img = Image.open(path).convert("RGB")  # ensure 3 channels
    tensor = transform(img).unsqueeze(0).to(device)  # add batch dim

    return tensor

def save_init_sample(init_samples, name):
    save_dir = "/home/yubai402b/MZJ/Diffusion_stain/ncsnv2/exp/image_samples/images"
    os.makedirs(save_dir, exist_ok=True)

    # Take the first few samples for visualization
    to_save = init_samples[:16]  # save 16 images (batch-first)

    # Denormalize if your data_transform maps images to [-1,1]
    to_save = (to_save.clamp(-1, 1) + 1) / 2.0  # back to [0,1]

    # Save as grid
    grid_path = os.path.join(save_dir, name + ".png")
    vutils.save_image(to_save, grid_path, nrow=4)
    print(f"✅ Saved init grid at {grid_path}")

    # Optionally: save each image separately
    # for i, img in enumerate(to_save):
    #     img_path = os.path.join(save_dir, f"init_{i}.png")
    #     vutils.save_image(img, img_path)



def save_tensor_images(tensor, out_dir='/home/yubai402b/MZJ/Diffusion_stain/ncsnv2/exp/image_samples/images', prefix="sample", nrow=4, denorm=True):
    os.makedirs(out_dir, exist_ok=True)
    if denorm:
        # assumes input in [-1,1], clamp to [0,1]
        tensor = (tensor.clamp(-1, 1) + 1) / 2.0
    grid_path = os.path.join(out_dir, f"{prefix}.png")
    vutils.save_image(tensor, grid_path, nrow=nrow)
    print(f"✅ Saved {grid_path}")
