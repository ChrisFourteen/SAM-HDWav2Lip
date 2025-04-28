import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None
        self.activation = activation
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = F.leaky_relu(out, 0.2)
        else:
            out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)
        return out

class ConvLayer(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size, downsample=False):
        layers = []
        if downsample:
            layers.append(nn.AvgPool2d(2))
        layers.append(
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=kernel_size // 2,
            )
        )
        super().__init__(*layers)

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, downsample=True):
        super().__init__()
        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=downsample)
        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=downsample
        )

    def forward(self, input):
        out = self.conv1(input)
        out = F.leaky_relu(out, 0.2)
        out = self.conv2(out)
        out = F.leaky_relu(out, 0.2)
        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)
        return out

class StyleGAN2Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        
        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))
        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            convs.append(ResBlock(in_channel, out_channel))
            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.final_conv = ConvLayer(in_channel, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)
        out = self.final_conv(out)
        out = out.view(out.shape[0], -1)
        out = self.final_linear(out)
        return out

class ImageQualityDiscriminator(StyleGAN2Discriminator):
    """
    Image Quality Assessment Discriminator based on StyleGAN2 architecture
    Returns a quality score between 0 (low quality) and 1 (high quality)
    """
    def __init__(self, size=256, channel_multiplier=2):
        super().__init__(size, channel_multiplier)
        
    def forward(self, input):
        # Get the base discriminator output
        out = super().forward(input)
        # Normalize to [0,1] range using sigmoid for quality score
        quality_score = torch.sigmoid(out)
        return quality_score
