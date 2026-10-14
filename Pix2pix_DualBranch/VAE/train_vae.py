import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from types import SimpleNamespace  # to create opt-like object

# Your dataset class (save this as data/aligned_dataset.py or paste here)
from data.aligned_dataset import AlignedDataset
from VAE.vae import VAE





# ────────────────────────────────────────────────
# Losses
# ────────────────────────────────────────────────
def vae_loss(recon_x, x, mean, logvar, kl_weight=1.0):
    recon_loss = F.l1_loss(recon_x, x)  # or MSE / BCE depending on data range

    # Stable KL
    kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    kl_loss = torch.clamp(kl_loss, min=-1e6, max=1e6)  # extra safety

    return recon_loss + kl_weight * kl_loss


# ────────────────────────────────────────────────
# Training function
# ────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create opt-like object for AlignedDataset
    opt = SimpleNamespace(
        dataroot=args.dataroot,
        phase='train',
        max_dataset_size=args.max_dataset_size,
        direction=args.direction,
        input_nc=args.input_nc,
        output_nc=args.output_nc,
        serial_batches=args.serial_batches,
        num_threads=args.num_threads,
        preprocess=args.preprocess,
        load_size=args.load_size,
        crop_size=args.crop_size,
        no_flip=args.no_flip,
    )

    # Training dataset (only mIHC = B)
    dataset_train = AlignedDataset(opt)
    print(f'The number of training images (mIHC) = {len(dataset_train)}')

    # Validation dataset
    opt.phase = 'val'
    dataset_validation = AlignedDataset(opt)
    print(f'The number of validation images (mIHC) = {len(dataset_validation)}')

    train_loader = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=not args.serial_batches,
        num_workers=args.num_threads,
        pin_memory=True,
    )

    val_loader = DataLoader(
        dataset_validation,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = VAE(in_channels=3, latent_channels=8).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    os.makedirs(args.checkpoints_dir, exist_ok=True)

    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = train_recon = train_kl = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in progress:
            # Extract mIHC images only (real_B)
            x = batch['A'].to(device)  # shape: (B, 3, H, W)
            B = batch['B'].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                recon, mean, logvar = model(x, noise=True)
                loss = vae_loss(recon, B, mean, logvar, args.kl_weight)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # important when using AMP!
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # try 0.5–5.0
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch['A'].to(device)
                B = batch['B'].to(device)
                with torch.cuda.amp.autocast():
                    recon, mean, logvar = model(x, noise=False)
                    loss = vae_loss(recon, B, mean, logvar, args.kl_weight)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'val_loss': avg_val_loss,
        }
        torch.save(checkpoint, os.path.join(args.checkpoints_dir, f"vae_epoch_{epoch}.pth"))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, os.path.join(args.checkpoints_dir, "vae_best.pth"))
            print("→ Saved best model")

        # if epoch % 5 == 0:
        #     with torch.no_grad():
        #         # Take first image in batch
        #         recon_img = recon[0].cpu()  # (C,H,W)
        #         orig_img = x[0].cpu()
        #
        #         # Assume model was trained with data in [-1,1] → map to [0,1]
        #         # If your data was normalized to [0,1] originally → remove the *0.5 + 0.5
        #         recon_vis = (recon - recon.min()) / (recon.max() - recon.min() + 1e-8)
        #         orig_vis = orig_img * 0.5 + 0.5
        #
        #         # Safety clamps — prevents clipping warnings & out-of-range crashes
        #         recon_vis = torch.clamp(recon_vis, 0.0, 1.0)
        #         orig_vis = torch.clamp(orig_vis, 0.0, 1.0)
        #
        #         # Convert to uint8 → most stable for matplotlib/PIL backend
        #         recon_vis = (recon_vis * 255).byte()  # → 0–255 uint8
        #         orig_vis = (orig_vis * 255).byte()
        #
        #         # To numpy + correct channel order (H,W,C)
        #         recon_np = recon_vis.permute(1, 2, 0).numpy()
        #         orig_np = orig_vis.permute(1, 2, 0).numpy()
        #
        #         # Create figure
        #         fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        #
        #         axs[0].imshow(orig_np)
        #         axs[0].set_title("Original mIHC")
        #         axs[0].axis('off')
        #
        #         axs[1].imshow(recon_np)
        #         axs[1].set_title(f"Reconstructed (epoch {epoch})")
        #         axs[1].axis('off')
        #
        #         plt.tight_layout()
        #         save_path = os.path.join(args.checkpoints_dir, f"recon_epoch_{epoch}.png")
        #         plt.savefig(save_path, bbox_inches='tight', dpi=150)
        #         plt.close(fig)  # important: prevents memory leak
        #
        #         print(f"Saved reconstruction: {save_path}")
    print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VAE on HEMIT mIHC images (using pix2pix-style dataset loader)")

    # Required
    parser.add_argument("--dataroot", type=str, required=True,
                        help="Path to HEMIT root (with trainA/trainB symlinks or folders)")

    # VAE-specific
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--kl_weight", type=float, default=1e-3,
                        help="KL annealing weight - start small (e.g. 1e-5) if recon is bad")

    # Checkpoint & logging
    parser.add_argument("--checkpoints_dir", type=str, default="./vae_checkpoints",
                        help="Where to save models and reconstructions")

    # Required by AlignedDataset (from pix2pix args)
    parser.add_argument("--phase", type=str, default="train")
    parser.add_argument("--direction", type=str, default="AtoB")
    parser.add_argument("--input_nc", type=int, default=3)
    parser.add_argument("--output_nc", type=int, default=3)
    parser.add_argument("--serial_batches", action="store_true")
    parser.add_argument("--num_threads", type=int, default=4)
    parser.add_argument("--max_dataset_size", type=int, default=float("inf"))
    parser.add_argument("--load_size", type=int, default=1024)
    parser.add_argument("--crop_size", type=int, default=1024)
    parser.add_argument("--preprocess", type=str, default="none")
    parser.add_argument("--no_flip", action="store_true")

    # Optional extras from pix2pix (ignored by VAE but needed for dataset)
    parser.add_argument("--name", type=str, default="vae_hemit")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--gpu_ids", type=str, default="0")

    args = parser.parse_args()
    train(args)