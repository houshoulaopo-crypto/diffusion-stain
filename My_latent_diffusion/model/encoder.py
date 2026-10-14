import torch
from torch.nn import functional as F
from torch import nn
from decoder import VAE_ResidualBlock, VAE_AttentionBlock

class VAE_Encoder(nn.Module):

    def __init__(self,):
        super().__init__(
            # feature
            nn.Conv2d(3, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            # residual
            VAE_ResidualBlock(128,128),
            # Down sample -> 1/2
            nn.Conv2d(128, 128, 3, stride=2, padding=0),

            VAE_ResidualBlock(128, 256),

            VAE_ResidualBlock(256, 256),

            # Down sample-> 1/4
            nn.Conv2d(256, 256, 3, stride=2, padding=0),

            VAE_ResidualBlock(256, 512),

            VAE_ResidualBlock(512, 512),

            # Down sample-> 1/8
            nn.Conv2d(512, 512, 3, stride=2, padding=0),

            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),

            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512, 512),

            nn.GroupNorm(32, 512),

            nn.SiLU(),

            nn.Conv2d(512, 8, 3, stride=1, padding=1),

            nn.Conv2d(8, 8, 1, stride=1, padding=0),

        )

    def forward(self, x, noise):
        # x : (batch_size, Channel, H, W)
        # noise : (batch_size, Channel, H/8, W/8)
        for module in self:
            if getattr(module, 'stride', None) == (2,2):
                x = F.pad(x,(0,1,0,1))
            x = module(x)

        # chunk is (8,H,W) -> 2* (4,H,W)
        mean, log_variance = x.chunk(2, dim=1)

        log_std = torch.clamp(log_variance,-30, 20)

        variance = log_std.exp()

        stdev = variance.sqrt()

        x = mean + (noise*stdev)

        x *= 0.18215

        return x


