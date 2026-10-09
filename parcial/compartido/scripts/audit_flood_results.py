"""Verify persisted scientific predictions, metrics and selected checkpoints."""

import argparse
import math
from itertools import product
from pathlib import Path

import numpy as np
import rasterio
import torch

from neural_exam.common.io import file_sha256, read_csv, read_json, write_json
from neural_exam.config import load_settings
from neural_exam.flood.data import read_pair


def close(actual, expected, label):
    if expected is None:
        if actual is not None:
            raise AssertionError(f"{label}: expected undefined metric")
    elif actual is None or not math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError(f"{label}: {actual} differs from {expected}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/flood.toml")
    parser.add_argument("--allow-partial", action="store_true")
    arguments = parser.parse_args()
    settings = load_settings(arguments.config)
    root = settings.output_root / "flood"
    manifest = read_json(root / "manifest.json")
    rows = []
    for supervision, loss in product(settings.flood["supervision"], settings.flood["losses"]):
        name = f"{supervision}_{loss}"
        run = root / name
        if not (run / "execution.json").is_file() and arguments.allow_partial:
            continue
        execution = read_json(run / "execution.json")
        metadata = read_json(run / "run.json")
        assert execution["scientific_run"], name
        assert metadata["data_identity"] == manifest["identity"], name
        for filename, digest in execution["configuration"]["training_source_sha256"].items():
            path = Path("/workspace/src/neural_exam") / ("common" if filename == "training.py" else "flood") / filename
            assert file_sha256(path) == digest, path
        history = read_csv(run / "history.csv")
        completed = read_json(run / "complete.json")
        assert len(history) == completed["epochs_executed"], name
        best_row = min(history, key=lambda row: float(row["val_loss"]))
        checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
        assert checkpoint["epoch"] == int(best_row["epoch"]), name
        close(checkpoint["best_loss"], float(best_row["val_loss"]), name)
        close(completed["best_val_loss"], float(best_row["val_loss"]), name)
        assert all(torch.isfinite(value).all() for value in checkpoint["model"].values()), name
        del checkpoint
        for split in ("test", "bolivia"):
            scores = read_json(run / split / "metrics.json")
            per_chip = {row["chip_id"]: row for row in read_csv(run / split / "per_chip.csv")}
            assert set(per_chip) == {record["id"] for record in manifest["records"][split]}, name
            totals = dict.fromkeys(("tp", "fp", "fn", "tn"), 0)
            chip_ious, reference_ious = [], []
            for record in manifest["records"][split]:
                _, target, profile = read_pair(record)
                path = run / split / f"{record['id']}_probability.tif"
                with rasterio.open(path) as raster:
                    assert (raster.crs, raster.transform) == (profile["crs"], profile["transform"]), path
                    assert raster.count == 1 and raster.dtypes == ("float32",) and raster.nodata == -1, path
                    probability = raster.read(1)
                assert probability.shape == target.shape, path
                valid = target >= 0
                assert (probability[~valid] == -1).all(), path
                assert np.isfinite(probability[valid]).all(), path
                assert ((probability[valid] >= 0) & (probability[valid] <= 1)).all(), path
                predicted = probability >= scores["threshold"]
                counts = {
                    "tp": int(np.count_nonzero((target == 1) & predicted)),
                    "fp": int(np.count_nonzero((target == 0) & predicted)),
                    "fn": int(np.count_nonzero((target == 1) & ~predicted)),
                    "tn": int(np.count_nonzero((target == 0) & ~predicted)),
                }
                assert sum(counts.values()) == record["valid_pixels"], path
                for key, value in counts.items():
                    assert int(per_chip[record["id"]][key]) == value, path
                    totals[key] += value
                union = counts["tp"] + counts["fp"] + counts["fn"]
                if union:
                    chip_ious.append(counts["tp"] / union)
                reference_ious.append((counts["tp"] + 1e-7) / (union + 1e-7))
            assert len(per_chip) == scores["chips"], name
            assert len(chip_ious) == scores["defined_iou_chips"], name
            assert len(per_chip) - len(chip_ious) == scores["empty_union_chips"], name
            close(scores["mean_chip_iou"], float(np.mean(chip_ious)) if chip_ious else None, name)
            close(scores["reference_mean_chip_iou"], float(np.mean(reference_ious)), name)
            pooled = scores["pooled_pixels"]
            assert all(totals[key] == pooled[key] for key in totals), name
            tp, fp, fn = (totals[key] for key in ("tp", "fp", "fn"))
            for key, numerator, denominator in (
                ("iou", tp, tp + fp + fn),
                ("precision", tp, tp + fp),
                ("recall", tp, tp + fn),
                ("f1", 2 * tp, 2 * tp + fp + fn),
                ("omission", fn, tp + fn),
                ("commission", fp, tp + fp),
            ):
                close(pooled[key], numerator / denominator if denominator else None, f"{name}/{split}/{key}")
            rows.append(
                {
                    "experiment": name,
                    "split": split,
                    "geotiffs_verified": len(per_chip),
                    "mean_chip_iou": scores["mean_chip_iou"],
                    "best_epoch": int(best_row["epoch"]),
                    "best_checkpoint_sha256": file_sha256(run / "best.pt"),
                }
            )
            print(f"Verified {name}/{split}: {len(per_chip)} GeoTIFFs", flush=True)
    if not arguments.allow_partial:
        assert len(rows) == 12, "The six-configuration matrix is incomplete"
    receipt = {
        "status": "passed",
        "complete_matrix": len(rows) == 12,
        "geotiffs_verified": sum(row["geotiffs_verified"] for row in rows),
        "results": rows,
    }
    write_json(
        settings.output_root / "audit" / f"flood-results{'-partial' if arguments.allow_partial else ''}.json", receipt
    )
    print(receipt)


if __name__ == "__main__":
    main()
