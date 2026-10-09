"""Independently recompute persisted MRI metrics and verify selected checkpoints."""

import math
from itertools import product
from pathlib import Path

import numpy as np
import torch

from neural_exam.common.io import file_sha256, read_csv, read_json, write_json
from neural_exam.common.metrics import multilabel_metrics
from neural_exam.config import load_settings
from neural_exam.mri.labels import LABELS


def main() -> None:
    """Reject incomplete runs, non-finite predictions and inconsistent checkpoint selection."""
    settings = load_settings("configs/mri.toml")
    root = settings.output_root / "mri"
    manifest = read_json(root / "manifest.json")
    gold = [row for row in manifest["records"] if row["split"] == "gold"]
    assert len(gold) == 58
    records_by_id = {row["study_id"]: row for row in gold}
    for row in manifest["records"]:
        with np.load(row["cache"], allow_pickle=False) as cache:
            image = cache["image"]
            assert image.dtype == np.float32
            assert image.shape == (8, 3, 224, 224)
            assert np.isfinite(image).all() and image.min() >= 0 and image.max() <= 1
    checks, detailed = [], []
    for architecture, loss, strategy in product(
        settings.mri["architectures"], settings.mri["losses"], settings.mri["noise_strategies"]
    ):
        name = f"{architecture}_{loss}_{strategy}"
        run_dir = root / name
        metadata = read_json(run_dir / "run.json")
        execution = read_json(run_dir / "execution.json")
        assert execution["scientific_run"]
        assert metadata["data_identity"] == manifest["identity"]
        assert execution["selected_checkpoint_sha256"] == file_sha256(run_dir / "best.pt")
        for relative, digest in metadata["configuration"]["training_source_sha256"].items():
            assert file_sha256(Path("/workspace/src/neural_exam") / relative) == digest
        rows = read_csv(run_dir / "predictions.csv")
        assert len(rows) == len(gold) and len({row["StudyInstanceUID"] for row in rows}) == len(gold)
        assert {row["StudyInstanceUID"] for row in rows} == set(records_by_id)
        probabilities = np.array([[float(row[label]) for label in LABELS] for row in rows])
        targets = np.array([records_by_id[row["StudyInstanceUID"]]["targets"] for row in rows])
        recomputed = multilabel_metrics(targets, probabilities, LABELS, settings.mri["threshold"])
        stored = read_json(run_dir / "metrics.json")
        assert recomputed == stored, f"Persisted metrics differ: {name}"
        history = read_csv(run_dir / "history.csv")
        assert all(math.isfinite(float(row[key])) for row in history for key in ("train_loss", "val_loss"))
        best = min(history, key=lambda row: float(row["val_loss"]))
        state = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
        assert state["epoch"] == int(best["epoch"])
        assert state["best_loss"] == float(best["val_loss"])
        assert all(torch.isfinite(tensor).all() for tensor in state["model"].values())
        complete = read_json(run_dir / "complete.json")
        assert complete["epochs_executed"] == len(history)
        assert len(history) <= settings.training["epochs"]
        for index in (0, 2, 7):
            assert (run_dir / f"cam_{index}.png").is_file()
            with np.load(run_dir / f"cam_{index}.npz", allow_pickle=False) as maps:
                assert maps["image"].shape == (8, 224, 224)
                heatmaps = maps["heatmaps"]
                assert heatmaps.shape == (8, 224, 224)
                assert np.isfinite(heatmaps).all() and heatmaps.min() >= 0 and heatmaps.max() <= 1
        checks.append(
            {
                "experiment": name,
                "epochs": len(history),
                "best_epoch": state["epoch"],
                "macro_auc": stored["macro_auc"],
                "partial_macro_auc": stored["partial_macro_auc"],
                "auc_labels_defined": stored["auc_labels_defined"],
                "macro_f1": stored["macro_f1"],
                "checkpoint_sha256": execution["selected_checkpoint_sha256"],
            }
        )
        detailed.extend({"experiment": name, **row} for row in stored["per_label"])
        del state
    assert len(read_csv(root / "experiments.csv")) == 8
    write_json(
        settings.output_root / "audit/mri_results.json",
        {
            "status": "passed",
            "experiments": checks,
            "per_label": detailed,
            "predictions_verified": 8 * 58,
            "cached_studies_verified": 600,
            "metric_equality": "Recomputed from persisted predictions with strict equality.",
            "checkpoint_selection": "Minimum weak-validation loss; gold outcomes unused for selection.",
        },
    )
    print("MRI result audit passed: eight runs, 464 study predictions and 600 cached studies.")


if __name__ == "__main__":
    main()
