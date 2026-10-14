from torch import nn

from VAE.decoder import VAE_Decoder
from VAE.encoder import VAE_Encoder


# ────────────────────────────────────────────────
# VAE Model wrapper
# ────────────────────────────────────────────────
class VAE(nn.Module):
    def __init__(self, in_channels=3, latent_channels=8):
        super().__init__()
        self.encoder = VAE_Encoder(in_channels=in_channels, out_channels=latent_channels)
        self.decoder = VAE_Decoder(in_channels=latent_channels // 2)

    def forward(self, x, noise=True):
        z, mean, logvar = self.encoder(x, noise=noise)
        recon = self.decoder(z, condition=None)
        return recon, mean, logvar