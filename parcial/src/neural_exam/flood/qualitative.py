"""Recreate georeferenced error panels and persist explicitly reviewed cases."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy.ndimage import binary_dilation, binary_erosion

from neural_exam.common.figures import save_figure
from neural_exam.common.io import read_csv, read_json, write_csv, write_json
from neural_exam.config import load_settings
from neural_exam.flood.data import read_pair


def render_error_figure(record, probability_path, destination, threshold, score):
    """Visualize observed FP/FN; never assign a physical cause automatically."""
    raw, target, profile = read_pair(record)
    with rasterio.open(probability_path) as raster:
        if raster.crs != profile["crs"] or raster.transform != profile["transform"]:
            raise ValueError("Exported probability geometry differs from the source.")
        probability = raster.read(1)
    valid = target >= 0
    prediction = probability >= threshold
    errors = np.zeros((*target.shape, 3), dtype=float)
    errors[(target == 0) & prediction & valid] = (0.9, 0.6, 0)
    errors[(target == 1) & ~prediction & valid] = (0, 0.45, 0.7)
    figure, axes = plt.subplots(2, 3, figsize=(10, 6))
    iou = "ND" if score["iou"] in (None, "") else f"{float(score['iou']):.3f}"
    figure.suptitle(
        f"{record['id']} | umbral {threshold:.2f} | IoU {iou} | FP {int(score['fp']):,} / FN {int(score['fn']):,}",
        fontsize=14,
    )
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
            {"vmin": -30, "vmax": 0} if title.endswith("(dB)") else ({"vmin": 0, "vmax": 1} if cmap is not None else {})
        )
        plotted = axis.imshow(display, cmap=cmap, **limits)
        if title == "Probabilidad":
            figure.colorbar(plotted, ax=axis, fraction=0.035, pad=0.02, ticks=[0, 0.5, 1])
        axis.set_title(title, fontsize=14)
        axis.axis("off")
    save_figure(figure, destination)
    masks = {
        "fp": valid & (target == 0) & prediction,
        "fn": valid & (target == 1) & ~prediction,
        "tp": valid & (target == 1) & prediction,
        "tn": valid & (target == 0) & ~prediction,
    }
    boundary = (binary_dilation(target == 1, iterations=3) ^ binary_erosion(target == 1, iterations=3)) & valid
    error = masks["fp"] | masks["fn"]
    error_count = int(error.sum())
    near_boundary = int((error & boundary).sum())
    return {
        "chip_id": record["id"],
        "median_db": {
            key: [float(np.median(band[mask])) for band in raw] if mask.any() else None for key, mask in masks.items()
        },
        "errors_within_three_pixels_of_water_boundary": near_boundary,
        "error_pixels": error_count,
        "boundary_fraction": near_boundary / error_count if error_count else None,
        "boundary_definition": "3 iterations of cross-shaped binary dilation XOR erosion; valid pixels only",
    }


def render_case(settings, split, identifier):
    """Use the prespecified gold+BCE baseline and its persisted predictions."""
    source = settings.output_root / "flood"
    baseline = source / "gold_bce"
    if not read_json(baseline / "execution.json")["scientific_run"]:
        raise ValueError("Qualitative scientific cases require full training.")
    manifest = read_json(source / "manifest.json")
    if read_json(baseline / "run.json")["data_identity"] != manifest["identity"]:
        raise ValueError("Baseline and current manifest differ.")
    record = next(record for record in manifest["records"][split] if record["id"] == identifier)
    score = next(row for row in read_csv(baseline / split / "per_chip.csv") if row["chip_id"] == identifier)
    statistics = render_error_figure(
        record,
        baseline / split / f"{identifier}_probability.tif",
        baseline / split / f"{identifier}_errors.png",
        settings.flood["threshold"],
        score,
    )
    return {"score": score, "statistics": statistics, "split": split}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/flood.toml")
    parser.add_argument("--cases", type=Path, default=Path("reports/flood/qualitative_cases.csv"))
    parser.add_argument("--split", choices=["test", "bolivia"])
    parser.add_argument("--chip-id")
    arguments = parser.parse_args()
    settings = load_settings(arguments.config)
    if arguments.chip_id:
        if not arguments.split:
            parser.error("--chip-id requires --split")
        print(render_case(settings, arguments.split, arguments.chip_id))
        return
    statistics = []
    for case in read_csv(arguments.cases):
        statistics.append(render_case(settings, case["split"], case["chip_id"]))
        path = settings.output_root / "flood/gold_bce" / case["split"] / "qualitative_review.csv"
        reviews = {row["chip_id"]: row for row in read_csv(path)}
        reviews[case["chip_id"]] = {key: case[key] for key in ("chip_id", "hypothesis", "visual_evidence")}
        write_csv(path, list(reviews.values()), ["chip_id", "hypothesis", "visual_evidence"])
    write_json(settings.output_root / "audit/qualitative-statistics.json", statistics)


if __name__ == "__main__":
    main()
