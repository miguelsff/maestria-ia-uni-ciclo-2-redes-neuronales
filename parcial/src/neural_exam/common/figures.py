"""Publication-oriented figures generated only from recorded experiments."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]


def save_figure(figure, path: Path) -> None:
    """Save a legible raster figure and release its memory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def history_plot(rows: list[dict], path: Path) -> None:
    """Plot training and validation losses on the same axes."""
    figure, axis = plt.subplots(figsize=(7, 4))
    for key, label, style, color in (
        ("train_loss", "Entrenamiento", "-", COLORS[0]),
        ("val_loss", "Validación", "--", COLORS[1]),
    ):
        axis.plot([row["epoch"] for row in rows], [row[key] for row in rows], style, label=label, color=color)
    axis.set(xlabel="Época", ylabel="Pérdida", title="Evolución del entrenamiento")
    axis.legend()
    save_figure(figure, path)


def bar_plot(names: list[str], values: list[float], path: Path, title: str, xlabel: str) -> None:
    """Produce sorted horizontal comparisons with a zero baseline."""
    order = np.argsort(values)
    figure, axis = plt.subplots(figsize=(8, max(3, len(names) * 0.3)))
    axis.barh([names[index] for index in order], [values[index] for index in order], color=COLORS[0])
    axis.set(xlim=(0, None), title=title, xlabel=xlabel)
    save_figure(figure, path)


def cam_plot(image: np.ndarray, heatmap: np.ndarray, path: Path, title: str) -> None:
    """Show the input and activation overlay side by side."""
    figure, axes = plt.subplots(1, 2, figsize=(8, 4))
    for axis in axes:
        axis.imshow(image, cmap="gray" if image.ndim == 2 else None)
        axis.axis("off")
    axes[0].set_title("Imagen")
    axes[1].imshow(heatmap, cmap="magma", alpha=0.5)
    axes[1].set_title(title)
    save_figure(figure, path)
