"""Original U-Net implementation using explicit skip connections."""

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    """Two spatial convolutions with group normalization for small batches."""

    def __init__(self, incoming: int, outgoing: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
            nn.GroupNorm(8, outgoing),
            nn.ReLU(),
            nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
            nn.GroupNorm(8, outgoing),
            nn.ReLU(),
        )

    def forward(self, image):
        return self.layers(image)


class UNet(nn.Module):
    """Map [batch, VV/VH, height, width] to one water logit per input pixel."""

    def __init__(self, base_channels=32):
        super().__init__()
        if base_channels < 8 or base_channels % 8:
            raise ValueError("base_channels must be a positive multiple of eight.")
        widths = [base_channels * multiplier for multiplier in (1, 2, 4, 8)]
        self.encoders = nn.ModuleList(
            [DoubleConv(2 if index == 0 else widths[index - 1], width) for index, width in enumerate(widths)]
        )
        self.bottleneck = DoubleConv(widths[-1], widths[-1] * 2)
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        incoming = widths[-1] * 2
        for width in reversed(widths):
            self.upsamples.append(nn.ConvTranspose2d(incoming, width, 2, stride=2))
            self.decoders.append(DoubleConv(width * 2, width))
            incoming = width
        self.output = nn.Conv2d(base_channels, 1, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        skips = []
        for encoder in self.encoders:
            image = encoder(image)
            skips.append(image)
            image = F.max_pool2d(image, 2)
        image = self.bottleneck(image)
        for upsample, decoder, skip in zip(self.upsamples, self.decoders, reversed(skips), strict=True):
            image = upsample(image)
            if image.shape[-2:] != skip.shape[-2:]:
                image = F.interpolate(image, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            image = decoder(torch.cat([skip, image], dim=1))
        return self.output(image)
