"""Run and resume the two-stage validation campaign."""
import argparse
import itertools
import json
import subprocess
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from tracking import setup
from train import PROJECT, train, update_best, run_name
from reports import create_reports

FAMILIES = ("resnet18", "yolo26m")
REGIMES = ("head", "full")
BUDGETS = (10, 40)
RATES = (0.0001, 0.0003)
SCOPED = [f":(top)examen-final/perufoodnet/{path}" for path in (
    "src/train.py", "src/experiments.py", "src/datasets.py", "src/models.py", "src/evaluate.py",
    "src/tracking.py", "configs", "data/manifest.csv", "data/classes.json", "requirements.txt",
)]


def source_compatible(source: str, current: str) -> bool:
    """Accept commits whose executable inputs match without generating fingerprints."""
    result = subprocess.run(["git", "diff", "--quiet", source, current, "--", *SCOPED], cwd=PROJECT)
    if result.returncode not in (0, 1):
        raise RuntimeError(f"Cannot compare source commit {source}")
    return result.returncode == 0


def history_consistent(history: pd.DataFrame, config: dict, summary: dict, checkpoint: dict) -> bool:
    """Validate epochs, patience, first best, stop reason and saved checkpoint."""
    count = len(history)
    if not 1 <= count <= config["epochs"] or history["epoch"].tolist() != list(range(1, count + 1)):
        return False
    if summary.get("epochs_completed") != count or checkpoint.get("config") != config:
        return False
    best, best_epoch, stale = -1.0, 0, 0
    for index, score in enumerate(history["val/f1_macro"], start=1):
        if pd.isna(score) or not 0 <= score <= 1:
            return False
        best, best_epoch, stale, _ = update_best(float(score), index, best, best_epoch, stale)
        if stale >= config["patience"] and index < count:
            return False
    reason = "early_stopping" if stale >= config["patience"] else "max_epochs"
    if reason == "max_epochs" and count != config["epochs"]:
        return False
    return (summary.get("stopping_reason") == reason and summary.get("best_epoch") == best_epoch
            and checkpoint.get("epoch") == best_epoch and abs(summary["best_f1"] - best) < 1e-10
            and abs(checkpoint["best_f1"] - best) < 1e-10)


def compatible(record: dict, config: dict, current: str, client: MlflowClient) -> bool:
    """Verify configuration, remote completion, source and local evidence."""
    if record.get("status") != "FINISHED" or record.get("config") != config:
        return False
    try:
        run = client.get_run(record["run_id"])
        if run.info.status != "FINISHED" or not source_compatible(run.data.params["commit"], current):
            return False
        root = Path(config["output"]) / "runs" / record["run_id"]
        if not all((root / name).is_file() for name in ("best.pt", "history.csv", "model_uri.json", "summary.json")):
            return False
        saved_config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        history = pd.read_csv(root / "history.csv")
        model = json.loads((root / "model_uri.json").read_text(encoding="utf-8"))
        import torch
        checkpoint = torch.load(root / "best.pt", map_location="cpu", weights_only=True)
        if (saved_config != config or summary.get("run_id") != record["run_id"]
                or summary.get("model_uri") != model.get("uri")
                or not model.get("independent_process") or not model["uri"].startswith("models:/")
                or not history_consistent(history, config, summary, checkpoint)
                or record.get("epochs_completed") != summary.get("epochs_completed")
                or record.get("stopping_reason") != summary.get("stopping_reason")
                or record.get("best_epoch") != summary.get("best_epoch")
                or abs(record["best_f1"] - summary["best_f1"]) > 1e-10
                or run.data.tags.get("stopping_reason") != summary["stopping_reason"]
                or run.data.metrics.get("epochs_completed") != summary["epochs_completed"]
                or run.data.metrics.get("best_epoch") != summary["best_epoch"]
                or abs(run.data.metrics.get("best_f1", -1) - summary["best_f1"]) > 1e-10):
            return False
        logged_model = client.get_logged_model(model["uri"].split("/", 1)[1])
        return str(logged_model.status) == "READY"
    except (KeyError, OSError, ValueError, RuntimeError, mlflow.exceptions.MlflowException):
        return False


def configuration(base: dict, family: str, regime: str, budget: int, rate: float, seed: int) -> dict:
    """Build one explicit configuration from the campaign settings."""
    return dict(base, family=family, regime=regime, budget=budget, lr=rate, seed=seed, phase="matrix")


def selected_initial(records: list[dict]) -> list[dict]:
    """Select a rate per model/regime/budget; the lower rate breaks a tie."""
    chosen = []
    for family, regime, budget in itertools.product(FAMILIES, REGIMES, BUDGETS):
        group = [r for r in records if r["config"]["family"] == family
                 and r["config"]["regime"] == regime and r["config"]["budget"] == budget
                 and r["config"]["seed"] == 42]
        if len(group) != 2:
            raise ValueError(f"Expected two learning rates for {(family, regime, budget)}")
        chosen.append(sorted(group, key=lambda r: (-r["best_f1"], r["config"]["lr"]))[0])
    return chosen


def run_campaign(base: dict) -> None:
    """Run 16 initial configurations and 16 selected-rate repetitions."""
    if (base["epochs"] != 200 or base["patience"] != 10 or base["weight_decay"] != 0.01
            or base["image_size"] != 224 or base["batch_size"] != 16):
        raise ValueError("R03.1 requires max 200 epochs, patience 10, weight decay 0.01, size 224 and batch 16")
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", *SCOPED], cwd=PROJECT, text=True)
    if dirty.strip():
        raise RuntimeError("Commit all changes to code, configuration and data before running the campaign")
    current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    setup(base)
    client = MlflowClient()
    output = Path(base["output"])
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / "matrix_index.json"
    campaign_path = output / "matrix_campaign.json"
    records = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else []
    if index_path.exists() and not campaign_path.exists():
        raise RuntimeError("Existing campaign index has no campaign metadata; preserve it and use a new directory")
    if campaign_path.exists():
        campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
        if (campaign.get("base_config") != base or not campaign.get("source_commit")
                or not source_compatible(campaign["source_commit"], current)):
            raise RuntimeError("Existing campaign has incompatible configuration or training source; use a new directory")
        for record in records:
            record_config = record.get("config", {})
            if any(record_config.get(key) != value for key, value in base.items() if key != "seed"):
                raise RuntimeError("Existing campaign index contains an incompatible configuration")
            if record.get("status") == "FINISHED" and not compatible(record, record_config, current, client):
                raise RuntimeError(f"Completed run {record.get('run_id')} is incompatible or incomplete; preserve it")

    def find(config: dict) -> dict | None:
        return next((r for r in reversed(records) if compatible(r, config, current, client)), None)

    initial = [configuration(base, f, r, b, lr, 42)
               for f, r, b, lr in itertools.product(FAMILIES, REGIMES, BUDGETS, RATES)]
    # If already complete, do not create or reopen a parent merely to resume.
    initial_records = [find(c) for c in initial]
    if all(initial_records):
        repeat = [dict(r["config"], seed=seed) for r in selected_initial(initial_records) for seed in (43, 44)]
        if all(find(c) for c in repeat) and (PROJECT / "results/winner.json").is_file():
            print("Campaign is already complete; no new run created.")
            return
    if campaign_path.exists():
        parent_id = json.loads(campaign_path.read_text(encoding="utf-8"))["parent_run_id"]
        parent_context = mlflow.start_run(run_id=parent_id)
    else:
        parent_context = mlflow.start_run(run_name="Matriz R03.1 · máximo 200 épocas · paciencia 10")
        parent_id = parent_context.info.run_id
        campaign_path.write_text(json.dumps({"parent_run_id": parent_id, "base_config": base,
                                              "source_commit": current}, indent=2), encoding="utf-8")
    print(f"Campaign parent: {parent_id}", flush=True)
    with parent_context:
        mlflow.set_tag("phase", "matrix")

        def ensure(config: dict) -> dict:
            existing = find(config)
            if existing:
                print(f"Reuse {existing['run_id']}", flush=True)
                return existing
            print(f"Starting {config['family']} {config['regime']} {config['budget']} "
                  f"lr={config['lr']} seed={config['seed']}", flush=True)
            try:
                summary = train(config, parent_id)
                record = {"config": config, "status": "FINISHED", **summary}
                records.append(record)
                return record
            except Exception as exc:
                name = run_name(config)
                matches = client.search_runs([mlflow.active_run().info.experiment_id],
                    filter_string=f"tags.mlflow.parentRunId = '{parent_id}' and tags.mlflow.runName = '{name}'")
                records.append({"config": config, "status": "FAILED", "run_id": matches[0].info.run_id if matches else None,
                                "error": repr(exc)})
                raise
            finally:
                index_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

        initial_records = [ensure(config) for config in initial]
        chosen = selected_initial(initial_records)
        repeated = [ensure(dict(record["config"], seed=seed)) for record in chosen for seed in (43, 44)]
        valid = initial_records + repeated
        create_reports(valid, PROJECT / "results", output, parent_id, current)
        mlflow.log_artifacts(str(PROJECT / "results"), artifact_path="selection")
        mlflow.set_tag("selection_status", "selected_before_test")
        print("Selection saved; test split was not evaluated.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/default.json")
    args = parser.parse_args()
    run_campaign(json.loads(args.config.read_text(encoding="utf-8-sig")))


if __name__ == "__main__":
    main()
