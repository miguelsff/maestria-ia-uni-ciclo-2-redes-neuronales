"""Evaluate classification using fixed 40-class macro F1."""
import argparse
import json
import shutil
import time
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



def validate_inputs(selection: dict, checkpoint: dict, labels: list[str], rows) -> None:
    """Check that the selected checkpoint and sequential test manifest agree."""
    config = checkpoint["config"]
    winner = selection["winner"]
    for key, expected in (("family", winner["modelo"]), ("regime", winner["régimen"]),
                          ("budget", winner["fotos_por_clase"]), ("lr", winner["tasa"]), ("seed", 42)):
        if config[key] != expected:
            raise ValueError(f"Checkpoint {key} does not match the selected configuration")
    if checkpoint["epoch"] != winner.get("best_epoch", checkpoint["epoch"]):
        raise ValueError("Checkpoint epoch does not match the selection")
    if len(labels) != 40 or len(set(labels)) != 40 or labels != selection["classes"]:
        raise ValueError("The 40 class labels do not match the selected order")
    if len(rows) != 600 or rows.relative_path.duplicated().any():
        raise ValueError("The test manifest must contain 600 unique paths")
    for index, label in enumerate(labels):
        group = rows[rows.class_index == index]
        if len(group) != 15 or set(group.image_number) != set(range(71, 86)) or set(group.class_name) != {label}:
            raise ValueError(f"Incorrect test rows for class {index}: {label}")


def model_latency_ms(model: nn.Module, device: torch.device) -> float:
    """Measure only model forward time for a synthetic batch of one image."""
    generator = torch.Generator().manual_seed(42)
    sample = torch.rand((1, 3, 224, 224), generator=generator).to(device)
    with torch.no_grad():
        for _ in range(30):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(200):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / 200


def _evaluate_winner(winner_path: Path, checkpoint_path: Path, manifest_path: Path, classes_path: Path,
                     dataset: Path, output: Path, tracking_uri: str | None, selection_commit: str) -> dict:
    """Evaluate the selected checkpoint on the sequential held-out photographs."""
    import pandas as pd
    from datasets import FoodDataset
    from evaluation_reports import choose_example, generate_test_reports
    from models import build_model

    start = time.perf_counter()
    if (output / "predicciones.csv").exists():
        raise FileExistsError("Evaluation output already exists; choose a new output directory")
    selection = json.loads(winner_path.read_text(encoding="utf-8-sig"))
    index = json.loads((winner_path.parent / "matrix_index.json").read_text(encoding="utf-8-sig"))
    selected_records = [record for record in index if record["run_id"] == selection["winner"]["run_id_seed42"]]
    if len(selected_records) != 1:
        raise ValueError("The selected run must appear exactly once in the training index")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    labels = json.loads(classes_path.read_text(encoding="utf-8-sig"))
    manifest = pd.read_csv(manifest_path)
    rows = manifest[manifest.split == "test"].reset_index(drop=True)
    validate_inputs(selection, checkpoint, labels, rows)
    if (checkpoint["epoch"] != selected_records[0]["best_epoch"]
            or checkpoint["config"] != selected_records[0]["config"]):
        raise ValueError("Checkpoint configuration or epoch differs from the selected run")
    config = checkpoint["config"]
    model = build_model(config["family"], None, config["regime"], checkpoint.get("architecture"))
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    loader = DataLoader(FoodDataset(dataset, rows, config["family"], False), batch_size=config["batch_size"])
    predictions, scores, image_losses, losses = [], [], [], 0.0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            if logits.shape != (len(targets), 40):
                raise ValueError(f"Expected 40 logits, received {tuple(logits.shape)}")
            losses += nn.functional.cross_entropy(logits, targets, reduction="sum").item()
            image_losses.extend(nn.functional.cross_entropy(logits, targets, reduction="none").cpu().tolist())
            score, indices = logits.softmax(1).max(1)
            predictions.extend(indices.cpu().tolist())
            scores.extend(score.cpu().tolist())
    if device.type == "cuda":
        torch.cuda.synchronize()
    total_seconds = time.perf_counter() - start
    target = rows.class_index.to_numpy()
    metrics = {"loss": losses / len(rows), "accuracy": accuracy_score(target, predictions),
               "f1_macro": f1_score(target, predictions, labels=np.arange(40), average="macro", zero_division=0),
               "images": len(rows), "uncertain_count": sum(score < .5 for score in scores),
               "evaluation_seconds": total_seconds, "latency_model_ms": model_latency_ms(model, device)}
    evidence = rows[["relative_path", "image_number"]].copy()
    evidence["true_class"] = rows.class_name
    evidence["true_index"] = rows.class_index
    evidence["predicted_index"] = predictions
    evidence["predicted_class"] = [labels[index] for index in predictions]
    evidence["score"] = scores
    evidence["cross_entropy"] = image_losses
    evidence["correct"] = evidence.true_index == evidence.predicted_index
    evidence["uncertain"] = evidence.score < .5
    output.mkdir(parents=True, exist_ok=True)
    evidence.to_csv(output / "predicciones.csv", index=False, encoding="utf-8")
    catalog = json.loads(classes_path.with_name("ingredients.json").read_text(encoding="utf-8-sig"))
    case = choose_example(evidence, catalog)
    example = output / "ejemplos/error.jpg"
    example.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(dataset / case["relative_path"], example)
    generate_test_reports(evidence, labels, catalog, example, output)
    description = {**metrics, "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
                   "evaluation_time_definition": "Checkpoint loading, data preparation and one complete inference pass; excludes plotting, logging and latency benchmark.",
                   "latency_definition": "Mean forward-only latency of 200 synthetic 224x224 batch-one inputs after 30 warmups on the stated device; excludes loading and preprocessing.",
                   "uncertainty_rule": "Maximum softmax below 0.50; all predictions are included in metrics.",
                   "split_policy": "sequential; correlated views may appear across splits",
                   "selection_commit": selection_commit,
                   "training_source_commit": selection["training_source_commit"],
                   "winner_run_id": selection["winner"]["run_id_seed42"],
                   "model_uri": selection["winner"]["model_uri"]}
    (output / "metricas.json").write_text(json.dumps(description, ensure_ascii=False, indent=2), encoding="utf-8")
    if tracking_uri:
        import mlflow
        run = mlflow.active_run()
        model_id = selection["winner"]["model_uri"].split("/", 1)[1]
        mlflow.set_tags({"phase": "test", "winner_run_id": selection["winner"]["run_id_seed42"],
                         "selection_commit": selection_commit,
                         "training_source_commit": selection["training_source_commit"], "model_id": model_id})
        mlflow.log_params({"checkpoint": str(checkpoint_path), "family": config["family"],
                           "regime": config["regime"], "budget": config["budget"], "lr": config["lr"],
                           "seed": config["seed"], "best_epoch": checkpoint["epoch"], "classes": len(labels),
                           "split_policy": "sequential", "threshold": 0.5, "device": description["device"]})
        test_data = mlflow.data.from_pandas(rows, source=str(manifest_path), targets="class_index", name="perufoodnet_test")
        mlflow.log_input(test_data, context="testing")
        mlflow.log_metrics({f"test/{key}": value for key, value in metrics.items()}, model_id=model_id, dataset=test_data)
        description["mlflow_run_id"] = run.info.run_id
        (output / "metricas.json").write_text(json.dumps(description, ensure_ascii=False, indent=2), encoding="utf-8")
        mlflow.log_artifacts(str(output))
        for name in ("comparacion_validacion.png", "curva_ganadora.png"):
            path = winner_path.parent / name
            if path.is_file():
                mlflow.log_artifact(str(path), "validacion")
    return description


def evaluate_winner(winner_path: Path, checkpoint_path: Path, manifest_path: Path, classes_path: Path,
                    dataset: Path, output: Path, tracking_uri: str | None = None,
                    implementation_version: str | None = None, selection_commit: str | None = None) -> dict:
    """Evaluate offline or under a tracked run that records failure causes."""
    selection = json.loads(winner_path.read_text(encoding="utf-8-sig"))
    recorded_commit = selection.get("selection_commit")
    if selection_commit and recorded_commit and selection_commit != recorded_commit:
        raise ValueError("The supplied selection commit conflicts with the selection metadata")
    provenance = selection_commit or recorded_commit or "unknown"
    if not tracking_uri:
        return _evaluate_winner(winner_path, checkpoint_path, manifest_path, classes_path, dataset, output, None, provenance)
    import mlflow
    mlflow.set_tracking_uri(tracking_uri)
    parent = mlflow.MlflowClient().get_run(selection["parent_run_id"])
    with mlflow.start_run(experiment_id=parent.info.experiment_id, parent_run_id=parent.info.run_id,
                          run_name=f"Evaluación final · prueba · {selection['winner']['modelo']}"):
        mlflow.set_tag("phase", "test")
        if implementation_version:
            mlflow.set_tag("evaluation_source_commit", implementation_version)
        try:
            return _evaluate_winner(winner_path, checkpoint_path, manifest_path, classes_path,
                                    dataset, output, tracking_uri, provenance)
        except Exception as exc:
            mlflow.set_tag("failure_reason", repr(exc)[:1000])
            raise


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--winner", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=project / "data/manifest.csv")
    parser.add_argument("--classes", type=Path, default=project / "data/classes.json")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tracking-uri", help="Optional MLflow tracking server")
    parser.add_argument("--implementation-version", help="Optional evaluation source identifier for tracking")
    parser.add_argument("--selection-commit", help="Optional commit that fixed this selection")
    args = parser.parse_args()
    result = evaluate_winner(args.winner, args.checkpoint, args.manifest, args.classes,
                             args.dataset, args.output, args.tracking_uri, args.implementation_version,
                             args.selection_commit)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
