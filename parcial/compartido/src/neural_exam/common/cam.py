"""Grad-CAM computed from differentiable scores before postprocessing."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class GradCAM:
    """Capture feature gradients while avoiding full backward-hook/in-place conflicts."""

    def __init__(self, layer: nn.Module):
        self.features = None
        self.gradients = None
        self.handle = layer.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        self.features = output
        self.gradients = None
        output.register_hook(self._save_gradients)

    def _save_gradients(self, gradient):
        self.gradients = gradient

    def maps(self, score: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """Backpropagate a scalar score and normalize each captured image map."""
        score.backward()
        if self.gradients is None:
            raise RuntimeError("The selected score does not depend on the captured layer.")
        weights = self.gradients.mean(dim=(-2, -1), keepdim=True)
        maps = (weights * self.features).sum(dim=1, keepdim=True).relu()
        maps = F.interpolate(maps, size=size, mode="bilinear", align_corners=False).squeeze(1)
        minimum = maps.flatten(1).amin(1)[:, None, None]
        maximum = maps.flatten(1).amax(1)[:, None, None]
        return ((maps - minimum) / (maximum - minimum).clamp_min(1e-8)).detach()

    def close(self):
        """Remove hooks after the explanation to release feature references."""
        self.handle.remove()
        self.features = self.gradients = None
