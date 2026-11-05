"""Create reproducible validation tables and select a model without using test data."""
import argparse
import json
import itertools
import math
import statistics
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from mlflow.tracking import MlflowClient

from models import build_model

PROJECT = Path(__file__).resolve().parents[1]


def latency_ms(checkpoint: Path, device: torch.device) -> float:
    """Measure batch-one model latency with identical synthetic input on the same device."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = saved["config"]
    model = build_model(config["family"], None, config["regime"], saved.get("architecture"))
    model.load_state_dict(saved["state_dict"])
    model.to(device).eval()
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


def validation_tables(records: list[dict], destination: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build validation tables and comparison figure from the saved index."""
    rows = []
    for record in records:
        config = record["config"]
        rows.append({"modelo": config["family"], "régimen": config["regime"], "fotos_por_clase": config["budget"],
                     "tasa": config["lr"], "semilla": config["seed"], "f1_validación": record["best_f1"],
                     "mejor_época": record["best_epoch"], "épocas_completadas": record["epochs_completed"],
                     "motivo_parada": record["stopping_reason"], "segundos": record["seconds"],
                     "memoria_gpu_mib": record["peak_gpu_mib"], "run_id": record["run_id"],
                     "modelo_uri": record["model_uri"]})
    destination.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    names = {"head": "capa final", "full": "red completa"}
    frame.replace({"régimen": names}).to_csv(destination / "corridas.csv", index=False)
    initial = frame[frame["semilla"] == 42]
    initial.sort_values(["modelo", "régimen", "fotos_por_clase", "tasa"]).replace(
        {"régimen": names}).to_csv(destination / "comparacion_misma_tasa.csv", index=False)
    rates = (initial.sort_values(["f1_validación", "tasa"], ascending=[False, True])
             .drop_duplicates(["modelo", "régimen", "fotos_por_clase"]))
    chosen = frame.merge(rates[["modelo", "régimen", "fotos_por_clase", "tasa"]],
                         on=["modelo", "régimen", "fotos_por_clase", "tasa"])
    grouped = (chosen.groupby(["modelo", "régimen", "fotos_por_clase", "tasa"], as_index=False)
               .agg(media_f1=("f1_validación", "mean"), desviación_muestral=("f1_validación", "std"),
                    semillas=("semilla", "count")))
    if len(grouped) != 8 or not (grouped["semillas"] == 3).all():
        raise ValueError("Expected three seeds for each selected rate")
    grouped.replace({"régimen": names}).to_csv(destination / "resumen_validacion.csv", index=False)
    figure, axis = plt.subplots(figsize=(9, 5))
    labels = [f"{r.modelo} · {names[r.régimen]} · {r.fotos_por_clase}" for r in grouped.itertuples(index=False)]
    axis.errorbar(grouped["media_f1"], range(len(grouped)), xerr=grouped["desviación_muestral"],
                  fmt="o", capsize=3, color="#285f91")
    axis.set_yticks(range(len(grouped)), labels)
    axis.set(xlabel="F1 macro de validación (media ± desviación muestral)",
             title="Comparación por arquitectura, ajuste y fotos por clase", xlim=(0, 1))
    axis.invert_yaxis()
    figure.tight_layout()
    figure.savefig(destination / "comparacion_validacion.png", dpi=160)
    plt.close(figure)
    return chosen, grouped


def validate_selection_records(selection: dict, records: list[dict]) -> None:
    """Verify rates, seed groups, candidate statistics and the selected seed-42 run."""
    if (len(records) != 32 or len({record.get("run_id") for record in records}) != 32
            or any(record.get("status") != "FINISHED" for record in records)):
        raise ValueError("The selection requires 32 distinct completed runs")
    initial = [record for record in records if record["config"]["seed"] == 42]
    if len(initial) != 16 or not selection.get("parent_run_id"):
        raise ValueError("The selection lacks its parent or 16 initial runs")
    expected = {}
    for family, regime, budget in itertools.product(("resnet18", "yolo26m"), ("head", "full"), (10, 40)):
        group = [record for record in initial if record["config"]["family"] == family
                 and record["config"]["regime"] == regime and record["config"]["budget"] == budget]
        if len(group) != 2 or {record["config"]["lr"] for record in group} != {0.0001, 0.0003}:
            raise ValueError("Initial learning rates do not match the protocol")
        chosen = sorted(group, key=lambda item: (-item["best_f1"], item["config"]["lr"]))[0]
        rate = chosen["config"]["lr"]
        seeds = [record for record in records if record["config"]["family"] == family
                 and record["config"]["regime"] == regime and record["config"]["budget"] == budget
                 and record["config"]["lr"] == rate]
        if sorted(record["config"]["seed"] for record in seeds) != [42, 43, 44]:
            raise ValueError("The selected rate does not have exactly three seeds")
        if budget == 40:
            expected[(family, regime)] = (chosen, statistics.mean(record["best_f1"] for record in seeds),
                                          statistics.stdev(record["best_f1"] for record in seeds))
    candidates = selection["candidates"]
    if len(candidates) != 4 or len({(item["modelo"], item["régimen"]) for item in candidates}) != 4:
        raise ValueError("The selection must contain four distinct 40-image candidates")
    for candidate in candidates:
        key = (candidate["modelo"], candidate["régimen"])
        if key not in expected:
            raise ValueError("Unexpected candidate")
        chosen, mean, deviation = expected[key]
        if (candidate["fotos_por_clase"] != 40 or candidate["tasa"] != chosen["config"]["lr"]
                or candidate["run_id_seed42"] != chosen["run_id"] or candidate["model_uri"] != chosen["model_uri"]
                or not math.isclose(candidate["media_f1"], mean, rel_tol=1e-10, abs_tol=1e-12)
                or not math.isclose(candidate["desviación_muestral"], deviation, rel_tol=1e-10, abs_tol=1e-12)
                or not math.isfinite(candidate["latency_ms"]) or candidate["latency_ms"] < 0):
            raise ValueError("Candidate configuration or statistics do not match the validation index")
    if selection["winner"] != sorted(candidates, key=lambda item: (-item["media_f1"], item["latency_ms"]))[0]:
        raise ValueError("The winner does not match the recorded candidates")


def validate_test_provenance(source: Path, selection: dict) -> None:
    """Reject test predictions that belong to another selection before producing outputs."""
    predictions = source / "test/predicciones.csv"
    if not predictions.exists():
        return
    metrics_path = source / "test/metricas.json"
    if not metrics_path.is_file():
        raise ValueError("Test predictions lack provenance in test/metricas.json")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8-sig"))
    winner = selection["winner"]
    if (metrics.get("winner_run_id") != winner["run_id_seed42"]
            or metrics.get("model_uri") != winner["model_uri"]):
        raise ValueError("Test predictions belong to a different winner or model")


def ensure_separate_from_official(destination: Path) -> None:
    """Keep the project's frozen official evidence read-only for new output."""
    destination = destination.resolve()
    official = (PROJECT / "results").resolve()
    if destination == official or destination.is_relative_to(official) or official.is_relative_to(destination):
        raise ValueError("Choose an output directory separate from the project's frozen official results")


def create_reports(records: list[dict], destination: Path, output: Path, parent_id: str, current: str) -> dict:
    """Produce tables, figures and the selection checkpoint from 32 completed runs."""
    if len(records) != 32 or any(r["status"] != "FINISHED" for r in records):
        raise ValueError("Selection requires exactly 32 completed runs")
    output = output.resolve()
    destination = destination.resolve()
    if destination != (output / "results").resolve():
        raise ValueError("Campaign results must be written to its own output/results directory")
    ensure_separate_from_official(destination)
    if any(Path(record["config"]["output"]).resolve() != output for record in records):
        raise ValueError("Training records belong to a different campaign output")
    if len({record["run_id"] for record in records}) != 32:
        raise ValueError("Campaign run identifiers must be unique")
    if (destination.exists() and any(destination.iterdir())) or (output / "perufoodnet_best.pt").exists():
        raise FileExistsError("Campaign selection or checkpoint already exists; preserve it")
    destination.mkdir(parents=True, exist_ok=True)
    histories = destination / "histories"
    histories.mkdir(exist_ok=True)
    for record in records:
        shutil.copy2(output / "runs" / record["run_id"] / "history.csv", histories / f"{record['run_id']}.csv")
    selected, grouped = validation_tables(records, destination)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidates = []
    for row in grouped[grouped["fotos_por_clase"] == 40].to_dict("records"):
        match = selected[(selected["modelo"] == row["modelo"]) & (selected["régimen"] == row["régimen"])
                         & (selected["fotos_por_clase"] == 40) & (selected["semilla"] == 42)].iloc[0]
        checkpoint = output / "runs" / match["run_id"] / "best.pt"
        row.update({"run_id_seed42": match["run_id"], "model_uri": match["modelo_uri"],
                    "latency_ms": latency_ms(checkpoint, device), "checkpoint_source": str(checkpoint)})
        candidates.append(row)
    winner = sorted(candidates, key=lambda row: (-row["media_f1"], row["latency_ms"]))[0]
    official = output / "perufoodnet_best.pt"
    shutil.copy2(winner["checkpoint_source"], official)
    source_commit = MlflowClient().get_run(winner["run_id_seed42"]).data.params["commit"]
    selection = {
        "rule": "Maximize mean validation macro F1 among 40-image configurations; exact ties use lower latency.",
        "rate_rule": "Best validation macro F1 with seed 42; exact ties use 0.0001.",
        "standard_deviation": "sample, ddof=1", "latency": {"device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "input": "one synthetic 224x224 image", "warmup": 30, "iterations": 200},
        "parent_run_id": parent_id, "training_source_commit": source_commit,
        "report_source_commit": current, "candidates": candidates, "winner": winner,
        "checkpoint": str(official), "classes": json.loads((PROJECT / "data/classes.json").read_text(encoding="utf-8-sig")),
    }
    (destination / "matrix_index.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (destination / "winner.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    return selection


def regenerate_reports(source: Path, destination: Path, classes_path: Path, ingredients_path: Path) -> None:
    """Regenerate figures and tables from saved evidence without models or MLflow."""
    from curves import render_curves
    from evaluation_reports import generate_test_reports

    ensure_separate_from_official(destination)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError("Report output must be a new or empty directory")
    records = json.loads((source / "matrix_index.json").read_text(encoding="utf-8"))
    selection = json.loads((source / "winner.json").read_text(encoding="utf-8"))
    if len(records) != 32 or any(record["status"] != "FINISHED" for record in records):
        raise ValueError("Expected 32 completed training records")
    validate_selection_records(selection, records)
    validate_test_provenance(source, selection)
    validation_tables(records, destination)
    winner_id = selection["winner"]["run_id_seed42"]
    render_curves(pd.read_csv(source / "histories" / f"{winner_id}.csv"), destination / "curva_ganadora.png")
    predictions = source / "test" / "predicciones.csv"
    if predictions.exists():
        labels = json.loads(classes_path.read_text(encoding="utf-8-sig"))
        catalog = json.loads(ingredients_path.read_text(encoding="utf-8-sig"))
        generate_test_reports(pd.read_csv(predictions), labels, catalog,
                              source / "test/ejemplos/error.jpg", destination / "test")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=PROJECT / "results")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--classes", type=Path, default=PROJECT / "data/classes.json")
    parser.add_argument("--ingredients", type=Path, default=PROJECT / "data/ingredients.json")
    args = parser.parse_args()
    regenerate_reports(args.results, args.output or args.results, args.classes, args.ingredients)


if __name__ == "__main__":
    main()
