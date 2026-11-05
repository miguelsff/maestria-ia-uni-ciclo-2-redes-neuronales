"""Datasets and shared image preprocessing for PeruFoodNet."""
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def image_transform(family: str, training: bool = False) -> transforms.Compose:
    """Apply identical geometry and family-specific normalization."""
    steps = [transforms.Resize((224, 224))]
    if training:
        steps.append(transforms.RandomHorizontalFlip())
    steps.append(transforms.ToTensor())
    if family == "resnet18":
        steps.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    return transforms.Compose(steps)


class FoodDataset(Dataset):
    """Load manifest rows; optionally cache resized RGB images in RAM."""
    def __init__(self, root: Path, rows: pd.DataFrame, family: str, training: bool, cache: bool = False) -> None:
        self.root, self.rows = root, rows.reset_index(drop=True)
        self.transform = image_transform(family, training)
        self.images = None
        if cache:
            self.images = []
            for path in self.rows.relative_path:
                with Image.open(root / path) as image:
                    self.images.append(image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows.iloc[index]
        if self.images is not None:
            image = self.images[index]
        else:
            with Image.open(self.root / row.relative_path) as source:
                image = source.convert("RGB")
        return self.transform(image), int(row.class_index)
