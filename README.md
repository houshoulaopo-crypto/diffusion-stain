# Diffusion stain

NCSNv2 stain translation is the main experiment. The model is a score-based network (NCSNv2) trained on histology patches, then used to sample PAS→H&E and Masson→H&E translations.

## Results

### PAS → H&E

Real PAS patches, then the sample at the last noise level:

![PAS real patches](ncsnv2/exp/image_samples/PAS2HE_1/real_sample.png)

![PAS to H&E sample](ncsnv2/exp/image_samples/PAS2HE_1/level_2310.png)

### Masson → H&E

![Masson real patches](ncsnv2/exp/image_samples/Massion2HE_2/real_sample.png)

![Masson to H&E sample](ncsnv2/exp/image_samples/Massion2HE_2/level_2310.png)

### 128px sampling

![128 real patches](ncsnv2/exp/image_samples/sigma_0.5_begin_0.1_128_3/real_sample.png)

![128 sample](ncsnv2/exp/image_samples/sigma_0.5_begin_0.1_128_3/level_2310.png)

Every run’s intermediate noise levels and grids are in [`ncsnv2/exp/image_samples/`](ncsnv2/exp/image_samples). Training grids are in [`ncsnv2/exp/logs/`](ncsnv2/exp/logs).

## Code

`ncsnv2/` is a working copy of [yang-song/ncsnv2](https://github.com/ermongroup/ncsnv2) plus the stain experiments:

- `configs/stain.yml`, `configs/stain_sample.yml` — stain training and sampling
- `datasets/stain.py` — stain dataset
- `stain.py`, `cut_svs.py`, `demo.py`, `cmd.txt` — sampling and slide-cutting helpers
- `Real-ESRGAN-master/` — super-resolution helper (weights not included)

Sampling command from `cmd.txt`:

```bash
python main.py --doc stain_sigma_1 --config stain_sample.yml --sample
```

Paths inside the configs still point at the original machine.

## Also in this repo

An earlier HEMIT VAE experiment: code in `Pix2pix_DualBranch/VAE/` and a preview in [`preview/test_reconstructions.png`](preview/test_reconstructions.png). Latent-diffusion sources are in `My_latent_diffusion/`.
