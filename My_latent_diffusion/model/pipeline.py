import torch
import torch.nn.functional as F
import numpy as np
from sympy.plotting.textplot import rescale
from tensorflow import timestamp
from tqdm import tqdm
from utils import *
from ddpm import DDPM

WIDTH = 512
HEIGHT = 512
LATENTS_HEIGHT = HEIGHT / 8
LATENTS_WIDTH = WIDTH / 8


def generate(prompt: str,
             uncond_prompt: str,
             input_image=None,
             strength=0.8,
             do_cfg=True,  # classify free guidance
             cfg_scale=7.5,
             sample_name='ddpm',
             n_inference_steps=50,
             model={},
             seed=None,
             device='gpu',
             idle_device=None,
             tokenizer=None, ):
    with torch.no_grad():

        if not (0 < strength < 1):
            raise ValueError('strength must be between 0 and 1')

        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        generator = torch.Generator(device=device)
        if seed:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = model["clip"]
        clip.to(device)

        if do_cfg:
            # convert the prompt into tokes using the tokenizer
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
            # batch, seq_le
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # batch, seq_len -> batch, seq_len, dim
            cond_context = clip(cond_tokens)

            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding="max_length", max_length=77).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            uncond_context = clip(uncond_tokens)

            # 2, seq_len, din = 2,77,768
            context = torch.cat([cond_context, uncond_context])
        else:
            tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            context = clip(tokens)

        to_idle(clip)

        if sample_name == 'ddpm':
            sampler = DDPM(generator=generator, device=device)
            sampler.set_inference_step(n_inference_steps)
        else:
            raise ValueError('sample_name must be either "ddpm" or "ddpm."')
        latent_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        if input_image:
            encoder = model["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize(WIDTH, HEIGHT)
            input_image_tensor = np.array(input_image_tensor)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)

            input_image_tensor = rescale(input_image_tensor, WIDTH, HEIGHT, (0, 255), (-1, 1))
            # H, W ,C -> Batch, H ,W, C
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # Batch, H, W, C -> batch, C, H, W
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            encoder_noise = torch.randn(latent_shape, generator=generator, device=device)

            #           run the image through to VAE
            latents = encoder(input_image_tensor, encoder_noise)

            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)
        else:
            # start with random noise N(0,1)
            latents = torch.randn(latent_shape, device=device, generator=generator)
        diffusion = model["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            #                 (1,320)
            time_embedding = get_time_embedding(timestep).to(device)
            # batch, 4, Latent_H, Latent_W
            model_input = latents

            if do_cfg:
                # batch, 4 , L_h, L_w -> 2 *batch, 4 , L_h, L_w
                model_input = model_input.repeat(2,1,1,1)

            model_output = diffusion(model_input, context,time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)

        decoder = model["decoder"]
        decoder.to(device)

        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, WIDTH, HEIGHT, (0, 255), (-1, 1))
        # batch, C, H, W => batch, H, W, C
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu",torch.uint8).numpy()
        return images

def get_time_embedding(timestep):
    freqs = torch.pow(10000, -torch.range(start=0, end=160,dtype=torch.float32) / 160)

    x = torch.tensor([timestep], dtype=torch.float32)[:,None] * freqs

    return torch.cat([torch.cos(x), torch.sin(x)], dim = -1)