"""Evaluate classification using fixed 40-class macro F1."""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Evaluate loss, accuracy and macro F1 with all 40 labels."""
    model.eval()
    targets, predictions, loss_sum = [], [], 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        if logits.shape != (len(labels), 40):
            raise ValueError(f"Expected 40 logits, received {tuple(logits.shape)}")
        loss_sum += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
        targets.extend(labels.cpu().tolist())
        predictions.extend(logits.argmax(1).cpu().tolist())
    return {"loss": loss_sum / len(targets), "accuracy": accuracy_score(targets, predictions),
            "f1_macro": f1_score(targets, predictions, labels=np.arange(40), average="macro", zero_division=0)}


def evaluate_winner(winner_path: Path, dataset: Path, output: Path) -> None:
    """Evaluate the committed winner on the held-out test split when authorized."""
    import pandas as pd
    from sklearn.metrics import classification_report, confusion_matrix
    from datasets import FoodDataset
    from models import build_model

    project = Path(__file__).resolve().parents[1]
    relative = winner_path.resolve().relative_to(project)
    subprocess.run(["git", "ls-files", "--error-unmatch", str(relative)], cwd=project, check=True, capture_output=True)
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(relative)], cwd=project, check=True)
    selection = json.loads(winner_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(selection["checkpoint"], map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model = build_model(config["family"], None, config["regime"], checkpoint.get("architecture"))
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    rows = pd.read_csv(project / "data/manifest.csv")
    rows = rows[rows.split == "test"].reset_index(drop=True)
    loader = DataLoader(FoodDataset(dataset, rows, config["family"], False), batch_size=config["batch_size"])
    metrics = evaluate(model, loader, device)
    predictions, scores = [], []
    with torch.no_grad():
        for images, _ in loader:
            probability = model(images.to(device)).softmax(1)
            score, index = probability.max(1)
            predictions.extend(index.cpu().tolist())
            scores.extend(score.cpu().tolist())
    labels = selection["classes"]
    rows["predicción"] = predictions
    rows["clase_predicha"] = [labels[index] for index in predictions]
    rows["puntuación"] = scores
    output.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output / "predicciones.csv", index=False)
    report = classification_report(rows.class_index, predictions, labels=list(range(40)), target_names=labels,
                                   output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(output / "métricas_por_clase.csv")
    matrix = confusion_matrix(rows.class_index, predictions, labels=list(range(40)))
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(output / "matriz_confusión.csv")
    (output / "métricas.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--winner", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate_winner(args.winner, args.dataset, args.output)


if __name__ == "__main__":
    main()
