from torch import nn
import torch.nn.functional as F
import torch
from attention import SelfAttention


class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        self.group_norm_1 = nn.GroupNorm(num_groups=32, num_channels=in_channel)

        self.conv_1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)

        self.group_norm_2 = nn.GroupNorm(num_groups=32, num_channels=out_channel)

        self.conv_2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)

        if in_channel == out_channel:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        residual = x

        x = self.group_norm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.group_norm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residual)

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels):
        self.group_norm_1 = nn.GroupNorm(num_groups=32, num_channels=channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x):
        residual = x

        x = self.group_norm_1(x)

        n,c,h,w = x.size()

        x =x.view(n, c, h*w)
        # B C HW -> B HW C
        x = x.transpose(-1, -2)

        x = self.attention(x)

        x = x.transpose(-1, -2)

        x = x.view(n, c, h, w)

        x += residual

        return x



class VAE_Decoder(nn.Module):
    def __init__(self):
        super(VAE_Decoder, self).__init__(
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1),

            nn.Conv2d(in_channels=4, out_channels=512, kernel_size=3, stride=1, padding=1),

            VAE_ResidualBlock(in_channel=512, out_channel=512),

            VAE_AttentionBlock(512),

            VAE_ResidualBlock(in_channel=512, out_channel=512),
            VAE_ResidualBlock(in_channel=512, out_channel=512),
            VAE_ResidualBlock(in_channel=512, out_channel=512),

            VAE_ResidualBlock(in_channel=512, out_channel=512),

            # upsample 1/8 -> 1/4
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            nn.Conv2d(512,512,kernel_size=3, stride=1, padding=1),

            VAE_ResidualBlock(in_channel=512, out_channel=256),
            VAE_ResidualBlock(in_channel=256, out_channel=256),
            VAE_ResidualBlock(in_channel=256, out_channel=256),

            # upsample 1/4 -> 1/2
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),

            VAE_ResidualBlock(in_channel=256, out_channel=128),
            VAE_ResidualBlock(in_channel=128, out_channel=128),
            VAE_ResidualBlock(in_channel=128, out_channel=128),

            nn.GroupNorm(num_groups=32, num_channels=128),

            nn.SiLU(),

            # return to image
            nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1),

        )

    def forward(self, x):
        # x: batch, 4, H/8, W/8
        x /= 0.18215
        for model in self.children():
            x = model(x)

        return x



