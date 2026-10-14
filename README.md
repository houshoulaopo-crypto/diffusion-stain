# Diffusion stain

Backup of the virtual-staining experiments in `Diffusion_stain`.

## What is here

- `My_latent_diffusion/` — latent diffusion implementation (encoder, decoder, DDPM, CLIP conditioning, sampling pipeline) and a small transformer module.
- `Pix2pix_DualBranch/` — working copy of [BianChang/Pix2pix_DualBranch](https://github.com/BianChang/Pix2pix_DualBranch) (MIT) plus a VAE trained on HEMIT multiplex-IHC images.
  - `VAE/` — VAE model, training, and test scripts.
  - `VAE/results/` — 945 reconstructed test patches from the HEMIT `testA` set.
  - `vae_checkpoints/recon_epoch_*.png` — training reconstruction previews.
  - `checkpoints/hemit_SwinTResnet_New_2/` — pix2pix training log and options (no weights).
- `scripts/cut_svs.py` — OpenSlide helper that cuts non-empty 512×512 patches from an SVS slide.

`test_vae.py` and `cut_svs.py` still contain absolute paths from the original machine.

## Left out on purpose

These were in the local project and are not in this repo:

- Raw slides and datasets (`data/`, the HEMIT image folders, about 50 GB). `data/SLE_svs` is clinical whole-slide data and was not uploaded.
- Third-party checkouts: Stable Diffusion, latent diffusion, NCSNv2, stain-transformation, and super-resolved virtual staining.
- Paper PDFs.
- The Chinese translation JSON under `My_latent_diffusion/transformer/` (about 1.2 GB).
- VAE weights `vae_best.pth` and `vae_epoch_100.pth` (about 958 MB each). GitHub rejects files over 100 MB.

The original project folder was kept locally. This repo is only the backup.
