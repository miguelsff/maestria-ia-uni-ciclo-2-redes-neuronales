"""Create reproducible validation tables and select a model without using test data."""
import argparse
import json
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


def create_reports(records: list[dict], destination: Path, output: Path, parent_id: str, current: str) -> dict:
    """Produce tables, figures and the selection checkpoint from 32 completed runs."""
    if len(records) != 32 or any(r["status"] != "FINISHED" for r in records):
        raise ValueError("Selection requires exactly 32 completed runs")
    destination.mkdir(parents=True, exist_ok=True)
    histories = destination / "histories"
    histories.mkdir(exist_ok=True)
    rows = []
    for record in records:
        config = record["config"]
        rows.append({"modelo": config["family"], "régimen": config["regime"], "fotos_por_clase": config["budget"],
                     "tasa": config["lr"], "semilla": config["seed"], "f1_validación": record["best_f1"],
                     "mejor_época": record["best_epoch"], "épocas_completadas": record["epochs_completed"],
                     "motivo_parada": record["stopping_reason"], "segundos": record["seconds"],
                     "memoria_gpu_mib": record["peak_gpu_mib"], "run_id": record["run_id"],
                     "modelo_uri": record["model_uri"]})
        shutil.copy2(output / "runs" / record["run_id"] / "history.csv", histories / f"{record['run_id']}.csv")
    frame = pd.DataFrame(rows)
    frame.to_csv(destination / "corridas.csv", index=False)
    initial = frame[frame["semilla"] == 42].copy()
    initial.sort_values(["modelo", "régimen", "fotos_por_clase", "tasa"]).to_csv(
        destination / "comparacion_misma_tasa.csv", index=False)
    best_rates = (initial.sort_values(["f1_validación", "tasa"], ascending=[False, True])
                  .drop_duplicates(["modelo", "régimen", "fotos_por_clase"]))
    selected = frame.merge(best_rates[["modelo", "régimen", "fotos_por_clase", "tasa"]],
                           on=["modelo", "régimen", "fotos_por_clase", "tasa"])
    grouped = (selected.groupby(["modelo", "régimen", "fotos_por_clase", "tasa"], as_index=False)
               .agg(media_f1=("f1_validación", "mean"), desviación_muestral=("f1_validación", "std"),
                    semillas=("semilla", "count")))
    if len(grouped) != 8 or not (grouped["semillas"] == 3).all() or len(frame) != 32:
        raise ValueError("Each of the eight chosen configurations must have three seeds")
    grouped.to_csv(destination / "resumen_validacion.csv", index=False)
    figure, axis = plt.subplots(figsize=(9, 5))
    labels = [f"{r.modelo} · {r.régimen} · {r.fotos_por_clase}" for r in grouped.itertuples(index=False)]
    axis.errorbar(grouped["media_f1"], range(len(grouped)), xerr=grouped["desviación_muestral"],
                  fmt="o", capsize=3, color="#285f91")
    axis.set_yticks(range(len(grouped)), labels)
    axis.set_xlabel("F1 macro de validación (media ± desviación muestral)")
    axis.set_title("Comparación por arquitectura, ajuste y fotos por clase")
    axis.set_xlim(0, 1)
    axis.invert_yaxis()
    figure.tight_layout()
    figure.savefig(destination / "comparacion_validacion.png", dpi=160)
    plt.close(figure)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--parent-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import subprocess
    import mlflow
    records = json.loads(args.index.read_text(encoding="utf-8"))
    mlflow.set_tracking_uri(records[0]["config"]["tracking_uri"])
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    create_reports(records, PROJECT / "results", args.output, args.parent_id, source)


if __name__ == "__main__":
    main()
