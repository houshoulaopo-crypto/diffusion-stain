import os
import torch
import numpy as np
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm

# ---- Metrics ----
from pytorch_fid import fid_score
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric


def load_images_from_dir(directory, image_size=256):
    """Load images and preprocess for evaluation"""
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    imgs = []
    for fname in os.listdir(directory):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(os.path.join(directory, fname)).convert("RGB")
            imgs.append(tfm(img))
    return torch.stack(imgs)  # shape: (N,C,H,W)


def compute_psnr_ssim(real_dir, gen_dir, image_size=256):
    real_imgs = load_images_from_dir(real_dir, image_size)
    gen_imgs = load_images_from_dir(gen_dir, image_size)

    assert len(real_imgs) == len(gen_imgs), "Real and generated sets must have same length"

    psnr_list, ssim_list = [], []
    for r, g in zip(real_imgs, gen_imgs):
        r_np = r.permute(1, 2, 0).numpy()
        g_np = g.permute(1, 2, 0).numpy()
        psnr_list.append(psnr_metric(r_np, g_np, data_range=1.0))
        ssim_list.append(ssim_metric(r_np, g_np, channel_axis=2, data_range=1.0))

    return np.mean(psnr_list), np.mean(ssim_list)


def compute_fid(real_dir, gen_dir, device="cuda", batch_size=50):
    fid = fid_score.calculate_fid_given_paths(
        [real_dir, gen_dir],
        batch_size=batch_size,
        device=device,
        dims=2048  # InceptionV3 feature dim
    )
    return fid


if __name__ == "__main__":
    # Example usage
    real_dir = "/home/yubai402b/MZJ/Diffusion_stain/ncsnv2/patches/SLE_0033_HE_patch_2549.png"
    gen_dir = "/home/yubai402b/MZJ/Diffusion_stain/ncsnv2/exp/logs/stain_sigma_1/samples/image_grid_80000.png"

    print("📊 Evaluating metrics...")

    fid = compute_fid(real_dir, gen_dir)
    psnr, ssim = compute_psnr_ssim(real_dir, gen_dir)

    print(f"FID  = {fid:.4f}")
    print(f"PSNR = {psnr:.4f} dB")
    print(f"SSIM = {ssim:.4f}")
