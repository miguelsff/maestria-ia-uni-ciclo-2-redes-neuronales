"""Construct pretrained classifiers with a 40-logit output."""
from pathlib import Path

import torch
from torch import nn
from torchvision.models import resnet18
from ultralytics import YOLO
from ultralytics.nn.tasks import ClassificationModel


class LogitsModel(nn.Module):
    """Expose logits in both training and evaluation for either family."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.model(inputs)
        return output[1] if isinstance(output, tuple) else output


def build_model(family: str, weights: Path | None, regime: str, architecture: dict | None = None) -> LogitsModel:
    """Load local pretrained weights and replace only the final linear layer."""
    if family == "resnet18":
        base = resnet18(weights=None)
        if weights is not None:
            base.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        base.fc = nn.Linear(base.fc.in_features, 40)
        head = base.fc
    elif family == "yolo26m":
        if weights is not None:
            base = YOLO(str(weights)).model.float()
        elif architecture is not None:
            base = ClassificationModel(cfg=architecture, nc=40, verbose=False)
        else:
            raise ValueError("YOLO architecture is required without initial weights")
        classifier = base.model[-1]
        classifier.linear = nn.Linear(classifier.linear.in_features, 40)
        head = classifier.linear
    else:
        raise ValueError(f"Unknown family: {family}")
    if regime not in {"head", "full"}:
        raise ValueError(f"Unknown regime: {regime}")
    base.requires_grad_(True)
    if regime == "head":
        base.requires_grad_(False)
        head.requires_grad_(True)
    return LogitsModel(base)


def set_training_mode(model: nn.Module, regime: str) -> None:
    """Freeze batch normalization statistics when training the final layer."""
    model.train()
    if regime == "head":
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
