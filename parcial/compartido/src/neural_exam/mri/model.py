"""Original 2.5D CNN and ImageNet backbone with a custom study-level head."""

from pathlib import Path

import torch
from torch import nn
from torchvision.models import resnet50


class StudyClassifier(nn.Module):
    """Map [batch, stacks, 3 adjacent slices, height, width] to twelve logits."""

    def __init__(self, architecture: str = "scratch", pretrained: bool = True, weights_path: Path | None = None):
        super().__init__()
        self.architecture = architecture
        if architecture == "scratch":
            blocks = []
            previous = 3
            for channels in (32, 64, 128, 256):
                blocks.append(
                    nn.Sequential(
                        nn.Conv2d(previous, channels, 3, padding=1, bias=False),
                        nn.GroupNorm(8, channels),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                    )
                )
                previous = channels
            self.encoder = nn.Sequential(*blocks)
            features = 256
        elif architecture == "resnet50":
            backbone = resnet50(weights=None)
            if pretrained:
                if weights_path is None or not weights_path.is_file():
                    raise FileNotFoundError("Download ImageNet weights explicitly with download --dataset weights.")
                backbone.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
            self.encoder = nn.Sequential(*list(backbone.children())[:-2])
            features = 2048
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(features, 12))

    @property
    def cam_layer(self):
        return self.encoder[-1]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        batch, stacks, channels, height, width = image.shape
        features = self.encoder(image.reshape(batch * stacks, channels, height, width))
        logits = self.head(self.pool(features).flatten(1))
        return logits.reshape(batch, stacks, 12).mean(dim=1)
