import torch
import torch.nn.functional as F
from mpmath import residual
from tensorflow.python.data.experimental.ops.testing import sleep
from torch import nn
from attention import SelfAttention, CrossAttenition


class TimeEmbedding(nn.Module):
    def __init__(self, n_embd):
        super(TimeEmbedding, self).__init__()
        self.layer_1 = nn.Linear(n_embd, n_embd * 4)
        self.layer_2 = nn.Linear(n_embd * 4, 4 * n_embd)

    def forward(self, x):
        x = self.layer_1(x)

        x = F.silu(x)

        x = self.layer_2(x)

        # (1,1280)
        return x


class Upsample(nn.Module):
    def __init__(self, channel):
        super(Upsample, self).__init__()
        self.conv = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # (batch, features, H, W) ->(batch, features, 2*H, 2*W)
        x = F.interpolate(x, scale_factor=2, mode='nearest', align_corners=False)
        return x


class SwitchSequential(nn.Sequential):

    def forward(self, x, context, time):
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x


class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, n_time=1280):
        super(UNET_ResidualBlock, self).__init__()
        self.groupnorm = nn.GroupNorm(num_groups=32, num_channels=in_channel)
        self.conv_feature = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)

        self.linear_time = nn.Linear(n_time, out_channel)
        self.groupnorm_merged = nn.GroupNorm(num_groups=32, num_channels=out_channel)
        self.conv_feature_merged = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)

        if in_channel != out_channel:
            self.residual_layer = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)
        else:
            self.residual_layer = nn.Identity()

    def forward(self, x, time):

        residual = x

        x = self.groupnorm(x)

        x = F.silu(x)

        x = self.conv_feature(x)

        time = F.silu(time)

        time = self.liner_time(time)

        merged = x + time.unsqueeze(-1).unsqueeze(-1)

        merged = self.groupnorm_merged(merged)

        merged = F.silu(merged)

        merged = self.conv_feature_merged(merged)

        return merged + self.residual_layer(residual)


class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_head: int, n_embd: int, d_context=768):
        super(UNET_AttentionBlock, self).__init__()
        channel = n_embd * n_head

        self.groupnorm = nn.GroupNorm(num_groups=32, num_channels=channel, eps=1e-6)
        self.conv_input = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channel)
        self.attention_1 = SelfAttention(N_heads=n_head, d_embed=channel, in_proj_bias=True, out_proj_bias=True)
        self.layernorm_2 = nn.LayerNorm(channel)
        self.attention_2 = CrossAttenition(N_heads=n_head, d_embed=channel, in_proj_bias=True, out_proj_bias=True)
        self.layernorm_3 = nn.LayerNorm(channel)
        self.liner_geglu_1 = nn.Linear(channel*4, channel*2)
        self.liner_geglu_2 = nn.Linear(channel*4, channel)

        self.conv_output = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x, context):
#             x: (batch, Feature, Height, Weight)
#           context: (batch, Seq_Len, Dim)
        residualLong = x

        x = self.groupnorm(x)

        x = self.conv_input(x)

        n, c, h, w = x.shape

        x = x.view(n, c, h*w)

        # batch , h*w , feature
        x = x.transpose(-1, -2)
#      Normalization + Self Attention with skip connection

        residual_short = x

        x = self.layernorm_1(x)
        self.attention_1(x)
        x += residual_short

        # Normalization + Cross Attention with skip connection
        residual_short = x

        x = self.layernorm_2(x)
        # cross attention
        self.attention_2(x)
        x += residual_short

        # Normalization + FF whti GEGLU and skip connection
        x = self.layernorm_3(x)

        x,gate = self.liner_geglu_1(x)

        x = x * F.gelu(gate)

        x = self.liner_geglu_2(x)

        x += residual_short
        # batch, H*W, features -> batch, features, H*W
        x = x.transpose(-1, -2)

        x = x.view(n, c, h, w)

class UNET(nn.Module):
    def __init__(self):
        super(UNET, self).__init__()

        self.encoder = nn.Sequential(
            # batch, 4, H/8, W/8
            SwitchSequential(nn.Conv2d(in_channels=4, out_channels=320, kernel_size=3, stride=1, padding=1)),

            SwitchSequential(UNET_ResidualBlock(320,320), UNET_AttentionBlock(8,40)),

            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            # batch, 320, H/16, W/16

            SwitchSequential(nn.Conv2d(in_channels=320, out_channels=320, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(nn.Conv2d(in_channels=640, out_channels=640, kernel_size=3, stride=2, padding=1)),
            # batch, 640, H/32, W/32

            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(nn.Conv2d(in_channels=1280, out_channels=1280, kernel_size=3, stride=2, padding=1)),
            # batch, 1280, H/64, W/64

            SwitchSequential(UNET_ResidualBlock(1280, 1280)),

            SwitchSequential(UNET_ResidualBlock(1280, 1280)),


        )

        self.bottleneck = nn.Sequential(
            UNET_ResidualBlock(1280, 1280),

            UNET_AttentionBlock(8, 160),

            UNET_ResidualBlock(1280,1280),

        )


        self.decoder = nn.Sequential(
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280),Upsample(1280)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280),UNET_AttentionBlock(8,160)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8,160),Upsample(1280)),

            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), Upsample(640)),

            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),

            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),

            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
        )
class UNET_output_layer(nn.Module):
    def __init__(self, in_channels, output_channels):
        super(UNET_output_layer, self).__init__()
        self.groupnorm = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv = nn.Conv2d(in_channels, output_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.groupnorm(x)

        x = F.silu(x)

        x = self.conv(x)

        return x


class Diffusion(nn.Module):

    def __init__(self, ):
        super(Diffusion, self).__init__()
        self.time_embedding = None
        self.unet = UNET()
        self.final = UNET_output_layer(320, 4)

    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        # latent : (batch, 4, Height /8 ,weight / 8)
        # context : (batch, seq_len, dim)
        # time : (1,320)

        # (1,320) ->(1,1280)
        time = self.time_embedding(time)

        # (batch, 4, hewght/8, Weight/8) -> (batch, 320, hewght/8, Weight/8)
        output = self.unet(latent, context, time)

        # (batch, 320, hewght / 8, Weight / 8) -> (batch, 4, hewght/8, Weight/8)
        output = self.final(output)

        return output
