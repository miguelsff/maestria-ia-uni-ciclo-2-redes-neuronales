"""Metrics with explicit missing-label and missing-class handling."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


def multilabel_metrics(targets: np.ndarray, probabilities: np.ndarray, labels: list[str], threshold: float) -> dict:
    """Compute macro AUC only when every required target has two classes."""
    targets, probabilities = np.asarray(targets), np.asarray(probabilities)
    if targets.shape != probabilities.shape or targets.ndim != 2 or targets.shape[1] != len(labels):
        raise ValueError("Expected matching [studies, labels] arrays.")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be finite and between zero and one.")
    results = []
    for index, name in enumerate(labels):
        valid = np.isfinite(targets[:, index])
        truth = targets[valid, index]
        scores = probabilities[valid, index]
        if not np.isin(truth, (0, 1)).all():
            raise ValueError("Observed labels must be binary.")
        precision, recall, f1, _ = (
            precision_recall_fscore_support(truth, scores >= threshold, average="binary", zero_division=0)
            if len(truth)
            else (None, None, None, None)
        )
        auc = float(roc_auc_score(truth, scores)) if len(np.unique(truth)) == 2 else None
        results.append(
            {
                "label": name,
                "support": len(truth),
                "positives": int(truth.sum()),
                "auc": auc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    defined = [row["auc"] for row in results if row["auc"] is not None]
    return {
        "threshold": threshold,
        "per_label": results,
        "auc_labels_defined": len(defined),
        "macro_auc": float(np.mean(defined)) if len(defined) == len(labels) else None,
        "partial_macro_auc": float(np.mean(defined)) if defined else None,
        "macro_f1": float(np.mean([row["f1"] for row in results if row["f1"] is not None]))
        if any(row["f1"] is not None for row in results)
        else None,
    }


def binary_counts(target: np.ndarray, probability: np.ndarray, threshold: float = 0.5) -> dict:
    """Count pixel confusion after masking invalid ground truth (-1)."""
    valid = np.isin(target, (0, 1))
    truth = target[valid].astype(bool)
    prediction = probability[valid] >= threshold
    return {
        "tp": int((truth & prediction).sum()),
        "fp": int((~truth & prediction).sum()),
        "fn": int((truth & ~prediction).sum()),
        "tn": int((~truth & ~prediction).sum()),
    }


def metrics_from_counts(counts: dict) -> dict:
    """Return undefined ratios as null rather than assigning perfect scores."""
    tp, fp, fn, tn = (counts[key] for key in ("tp", "fp", "fn", "tn"))

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None

    return {
        **counts,
        "valid_pixels": tp + fp + fn + tn,
        "iou": ratio(tp, tp + fp + fn),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "omission": ratio(fn, tp + fn),
        "commission": ratio(fp, tp + fp),
    }
