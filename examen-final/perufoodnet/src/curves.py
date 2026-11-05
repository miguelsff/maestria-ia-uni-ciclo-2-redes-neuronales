"""Render training histories and optionally update their MLflow figure artifact."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd

CURVES_REVISION = "panels-es-v3"


def render_curves(history: pd.DataFrame, output: Path) -> None:
    """Plot loss, accuracy and validation macro F1 in separate panels."""
    if history.empty:
        raise ValueError("The history contains no epochs")
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    panels = [
        ("Pérdida", [("train/loss", "Entrenamiento"), ("val/loss", "Validación")]),
        ("Exactitud", [("train/accuracy", "Entrenamiento"), ("val/accuracy", "Validación")]),
        ("F1 macro", [("val/f1_macro", "Validación")]),
    ]
    for axis, (title, series) in zip(axes, panels):
        for column, label in series:
            axis.plot(history["epoch"], history[column], marker="o", markersize=4, label=label)
        axis.set_title(title)
        axis.set_xlabel("Época")
        if len(history) == 1:
            axis.set_xticks([int(history["epoch"].iloc[0])])
        else:
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.set_ylabel(title)
        if title == "Pérdida":
            axis.margins(y=0.1)
        else:
            axis.set_ylim(0, 1)
        axis.legend()
        axis.grid(alpha=0.25)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--tracking-uri", default="http://localhost:4567/")
    parser.add_argument("--revision", default=CURVES_REVISION)
    args = parser.parse_args()
    render_curves(pd.read_csv(args.history), args.output)
    if args.run_id:
        import mlflow
        mlflow.set_tracking_uri(args.tracking_uri)
        client = mlflow.MlflowClient()
        client.log_artifact(args.run_id, str(args.output))
        client.set_tag(args.run_id, "curves_revision", args.revision)


if __name__ == "__main__":
    main()
