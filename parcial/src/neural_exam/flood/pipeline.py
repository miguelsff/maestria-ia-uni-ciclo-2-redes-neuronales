"""Strong/weak supervision and loss ablations with an untouched Bolivia holdout."""

import platform
import sys
import time
from functools import partial
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch

from neural_exam.common.figures import save_figure
from neural_exam.common.io import file_sha256, read_json, write_csv, write_json
from neural_exam.common.metrics import binary_counts, metrics_from_counts
from neural_exam.common.training import fit, make_loader, resolve_device, restore_model, seed_everything
from neural_exam.flood.data import FloodDataset, prepare, read_pair
from neural_exam.flood.loss import segmentation_loss
from neural_exam.flood.model import UNet


def evaluate(model, dataset, device, threshold, output) -> dict:
    """Export georeferenced probabilities and both chip-mean and pooled-pixel metrics."""
    model.eval()
    rows, totals = [], dict.fromkeys(("tp", "fp", "fn", "tn"), 0)
    output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for index, record in enumerate(dataset.records):
            image, target, _ = dataset[index]
            probability = model(image[None].to(device)).sigmoid()[0, 0].cpu().numpy()
            if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
                raise FloatingPointError(f"Invalid probability map: {record['id']}")
            counts = binary_counts(target[0].numpy(), probability, threshold)
            union = counts["tp"] + counts["fp"] + counts["fn"]
            rows.append(
                {
                    "chip_id": record["id"],
                    "country": record["country"],
                    **metrics_from_counts(counts),
                    "reference_iou": (counts["tp"] + 1e-7) / (union + 1e-7),
                }
            )
            totals = {key: totals[key] + counts[key] for key in totals}
            _, raw_target, profile = read_pair(record)
            profile.update(count=1, dtype="float32", nodata=-1, compress="deflate")
            probability[raw_target < 0] = -1
            with rasterio.open(output / f"{record['id']}_probability.tif", "w", **profile) as raster:
                raster.write(probability.astype(np.float32), 1)
    defined = [row["iou"] for row in rows if row["iou"] is not None]
    result = {
        "pooled_pixels": metrics_from_counts(totals),
        "mean_chip_iou": float(np.mean(defined)) if defined else None,
        "defined_iou_chips": len(defined),
        "empty_union_chips": len(rows) - len(defined),
        "reference_mean_chip_iou": float(np.mean([row["reference_iou"] for row in rows])) if rows else None,
        "reference_policy": "Official Train.ipynb epsilon=1e-7; empty union receives one",
        "chips": len(rows),
        "threshold": threshold,
        "empty_union_policy": "undefined chip IoU excluded; counts still contribute to pooled metrics",
    }
    write_csv(output / "per_chip.csv", rows)
    write_json(output / "metrics.json", result)
    # These are candidates for human error interpretation, never automatic causal labels.
    worst = sorted((row for row in rows if row["iou"] is not None), key=lambda row: row["iou"])[:6]
    by_id = {record["id"]: record for record in dataset.records}
    for row in worst:
        record = by_id[row["chip_id"]]
        raw, target, _ = read_pair(record)
        with rasterio.open(output / f"{record['id']}_probability.tif") as raster:
            probability = raster.read(1)
        prediction = probability >= threshold
        valid = target >= 0
        errors = np.zeros((*target.shape, 3), dtype=float)
        errors[(target == 0) & prediction & valid] = (0.9, 0.6, 0)
        errors[(target == 1) & ~prediction & valid] = (0, 0.45, 0.7)
        figure, axes = plt.subplots(2, 3, figsize=(10, 6))
        figure.suptitle(f"{record['id']} | umbral {threshold:.2f} | IoU {row['iou']:.3f}", fontsize=11)
        for axis, display, title, cmap in zip(
            axes.flat,
            (
                raw[0],
                raw[1],
                np.ma.masked_where(~valid, target),
                np.ma.masked_where(~valid, probability),
                np.ma.masked_where(~valid, prediction),
                errors,
            ),
            ("VV (dB)", "VH (dB)", "Referencia", "Probabilidad", "Predicción", "FP naranja / FN azul"),
            ("gray", "gray", "Blues", "viridis", "Blues", None),
            strict=True,
        ):
            limits = (
                {"vmin": -30, "vmax": 0}
                if title.endswith("(dB)")
                else ({"vmin": 0, "vmax": 1} if cmap is not None else {})
            )
            plotted = axis.imshow(display, cmap=cmap, **limits)
            if title == "Probabilidad":
                figure.colorbar(plotted, ax=axis, fraction=0.035, pad=0.02, ticks=[0, 0.5, 1])
            axis.set_title(title)
            axis.axis("off")
        save_figure(figure, output / f"{record['id']}_errors.png")
    if not (output / "qualitative_review.csv").exists():
        write_csv(
            output / "qualitative_review.csv",
            [{"chip_id": row["chip_id"], "hypothesis": "", "visual_evidence": ""} for row in worst],
            ["chip_id", "hypothesis", "visual_evidence"],
        )
    figure, axis = plt.subplots(figsize=(4, 4))
    matrix = np.array([[totals["tn"], totals["fp"]], [totals["fn"], totals["tp"]]])
    axis.imshow(matrix, cmap="Blues")
    for (row, column), value in np.ndenumerate(matrix):
        axis.text(column, row, f"{value:,}", ha="center", va="center")
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["No agua", "Agua"],
        yticklabels=["No agua", "Agua"],
        xlabel="Predicción",
        ylabel="Referencia",
    )
    save_figure(figure, output / "confusion.png")
    return result


def run(settings, resume=False, quick=False, manifest=None) -> dict:
    """Execute the six experiments with the same gold normalization and partitions."""
    manifest = manifest if manifest is not None else prepare(settings)
    output = settings.output_root / "flood"
    matrix = list(product(settings.flood["supervision"], settings.flood["losses"]))
    if any(supervision == "gold_weak" for supervision, _ in matrix) and not manifest["records"]["weak"]:
        raise FileNotFoundError("The full ablation requires weak SAR images and masks before training starts.")
    results = []
    for supervision, loss_name in matrix:
        seed_everything(settings.training["seed"])
        training_records = list(manifest["records"]["train"])
        if supervision == "gold_weak":
            if not manifest["records"]["weak"]:
                raise FileNotFoundError("The gold+weak ablation requires downloaded weak SAR images and masks.")
            training_records += manifest["records"]["weak"]
        if quick:
            training_records = list(manifest["records"]["train"][:4])
            if supervision == "gold_weak":
                training_records += manifest["records"]["weak"][:4]
        train = FloodDataset(training_records, manifest["mean"], manifest["std"], True)
        validation_records = manifest["records"]["val"][:2] if quick else manifest["records"]["val"]
        validation = FloodDataset(validation_records, manifest["mean"], manifest["std"])
        model = UNet(settings.flood["base_channels"])
        run_dir = output / f"{supervision}_{loss_name}"
        experiment_config = {
            **settings.training,
            "supervision": supervision,
            "loss": loss_name,
            "flood": settings.flood,
            "training_source_sha256": {
                path.name: file_sha256(path)
                for path in [Path(__file__).with_name(name) for name in ("data.py", "model.py", "loss.py")]
                + [Path(__file__).parents[1] / "common/training.py"]
            },
        }
        device = resolve_device(settings.training["device"])
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        checkpoint = fit(
            model,
            make_loader(train, settings.flood["batch_size"], settings.training, True),
            make_loader(validation, settings.flood["batch_size"], settings.training),
            partial(segmentation_loss, kind=loss_name),
            experiment_config,
            run_dir,
            manifest["identity"],
            resume,
        )
        training_seconds = time.monotonic() - started
        previous_execution = read_json(run_dir / "execution.json") if (run_dir / "execution.json").is_file() else None
        if resume and previous_execution:
            training_seconds = previous_execution["training_wall_seconds"]
        restore_model(model, checkpoint, device)
        scores = {}
        for split in ("test", "bolivia"):
            records = manifest["records"][split][:2] if quick else manifest["records"][split]
            scores[split] = evaluate(
                model,
                FloodDataset(records, manifest["mean"], manifest["std"]),
                device,
                settings.flood["threshold"],
                run_dir / split,
            )
        inside, outside = (scores[split]["mean_chip_iou"] for split in ("test", "bolivia"))
        results.append(
            {
                "experiment": run_dir.name,
                "test_iou": inside,
                "bolivia_iou": outside,
                "geographic_gap": inside - outside if inside is not None and outside is not None else None,
            }
        )
        write_json(
            run_dir / "execution.json",
            {
                "arguments": sys.argv,
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "configuration": experiment_config,
                "training_chips": len(training_records),
                "validation_chips": len(validation_records),
                "steps_per_epoch": len(make_loader(train, settings.flood["batch_size"], settings.training)),
                "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
                "training_wall_seconds": training_seconds,
                "scientific_run": not quick,
                "reused_completed_training_metadata": bool(resume and previous_execution),
                "gpu_name": torch.cuda.get_device_name() if device.type == "cuda" else None,
                "peak_cuda_allocated_bytes": max(
                    torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
                    previous_execution["peak_cuda_allocated_bytes"] if resume and previous_execution else 0,
                ),
                "peak_cuda_reserved_bytes": max(
                    torch.cuda.max_memory_reserved() if device.type == "cuda" else 0,
                    previous_execution["peak_cuda_reserved_bytes"] if resume and previous_execution else 0,
                ),
            },
        )
        write_csv(output / "experiments.csv", results)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_csv(output / "experiments.csv", results)
    return {"task": "flood", "experiments": results, "scientific_run": not quick}
