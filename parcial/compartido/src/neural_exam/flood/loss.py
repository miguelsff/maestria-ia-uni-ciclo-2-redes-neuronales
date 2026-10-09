"""Segmentation losses ignore nodata and weight supervision at chip level."""

from torch.nn import functional as F


def segmentation_loss(logits, target, weight, kind="combined"):
    """Normalize each chip over valid pixels before mixing strong and weak labels."""
    logits, target, weight = logits.float(), target.float(), weight.float()
    valid = (target == 0) | (target == 1)
    truth = target.clamp(0, 1)
    axes = tuple(range(1, logits.ndim))
    pixels = valid.sum(dim=axes)
    bce = (F.binary_cross_entropy_with_logits(logits, truth, reduction="none") * valid).sum(dim=axes)
    bce = bce / pixels.clamp_min(1)
    probability = logits.sigmoid() * valid
    intersection = (probability * truth).sum(dim=axes)
    dice = 1 - (2 * intersection + 1e-6) / (probability.sum(dim=axes) + (truth * valid).sum(dim=axes) + 1e-6)
    if kind == "bce":
        losses = bce
    elif kind == "dice":
        losses = dice
    elif kind == "combined":
        losses = bce + dice
    else:
        raise ValueError(f"Unknown segmentation loss: {kind}")
    confidence = weight.flatten(1).mean(1) * (pixels > 0)
    # Preserve weak_weight even in a batch containing weak chips exclusively.
    return (losses * confidence).sum() / (pixels > 0).sum().clamp_min(1)
