"""Masked confidence-weighted BCE and imbalance-aware sigmoid focal loss."""

import torch
from torch.nn import functional as F


def classification_loss(logits, targets, weights, kind="bce", gamma=2.0, alpha=None):
    """Normalize by supervised label mass; absent labels have exactly zero gradient."""
    valid = torch.isfinite(targets)
    targets = torch.nan_to_num(targets)
    mass = weights * valid
    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if kind == "focal":
        probability = logits.sigmoid()
        correct_probability = probability * targets + (1 - probability) * (1 - targets)
        balance = 0.25 if alpha is None else alpha.to(logits.device)
        alpha_factor = balance * targets + (1 - balance) * (1 - targets)
        losses = losses * alpha_factor * (1 - correct_probability).pow(gamma)
    elif kind != "bce":
        raise ValueError(f"Unknown loss: {kind}")
    return (losses * mass).sum() / mass.sum().clamp_min(1)
