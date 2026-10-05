"""Official Sen1Floods11 manifests, nodata masks and train-only normalization."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from neural_exam.common.io import file_sha256, fingerprint, write_csv, write_json

ALIGNMENT_ATOL = 1e-12

SPLITS = {
    "train": "flood_train_data.csv",
    "val": "flood_valid_data.csv",
    "test": "flood_test_data.csv",
    "bolivia": "flood_bolivia_data.csv",
}


def chip_id(name: str) -> str:
    """Use event and chip identifier; layer suffixes never distinguish split membership."""
    parts = Path(name).stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected Sen1Floods11 filename: {name}")
    return "_".join(parts[:2])


def read_pair(record: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Validate image/mask alignment and preserve nodata across both SAR bands."""
    with rasterio.open(record["image"]) as image, rasterio.open(record["label"]) as label:
        if image.count != 2 or label.count != 1:
            raise ValueError("Expected two SAR bands (VV,VH) and one label band.")
        if (
            image.crs is None
            or (image.width, image.height, image.crs)
            != (
                label.width,
                label.height,
                label.crs,
            )
            or not np.allclose(tuple(image.transform), tuple(label.transform), rtol=0, atol=ALIGNMENT_ATOL)
        ):
            raise ValueError(f"SAR and ground truth are not spatially aligned: {record['id']}")
        pixels = image.read(masked=True)
        labels = label.read(1, masked=True)
        target = labels.astype(np.float32).filled(-1)
        invalid = np.ma.getmaskarray(pixels).any(axis=0) | np.ma.getmaskarray(labels)
        raw = pixels.astype(np.float32).filled(np.nan)
        invalid |= ~np.isfinite(raw).all(axis=0)
        target[invalid] = -1
        if not np.isin(target, (-1, 0, 1)).all():
            raise ValueError("Unexpected water-mask values; do not silently remap classes.")
        profile = image.profile.copy()
    return np.nan_to_num(raw, nan=-50.0), target, profile


def build_records(root: Path, weak_max: int) -> tuple[dict[str, list[dict]], dict]:
    """Keep the official gold partitions and prohibit Bolivia in weak training."""
    if weak_max < 0:
        raise ValueError("weak_max must be nonnegative.")
    base = root / "v1.1"
    gold_root = base / "data/flood_events/HandLabeled"
    records = {}
    seen = set()
    for split, filename in SPLITS.items():
        rows = []
        with (base / "splits/flood_handlabeled" / filename).open(encoding="utf-8", newline="") as stream:
            for row in csv.reader(stream):
                if len(row) != 2 or not all(value.strip().endswith(".tif") for value in row):
                    raise ValueError(f"Unexpected official split row in {filename}.")
                image_name, label_name = (Path(value.strip()).name for value in row)
                identifier = chip_id(image_name)
                if identifier != chip_id(label_name) or identifier in seen:
                    raise ValueError("Mismatched image/label or overlapping official chip IDs.")
                seen.add(identifier)
                country = identifier.split("_")[0]
                if (country == "Bolivia") != (split == "bolivia"):
                    raise ValueError("Bolivia must belong exclusively to the external evaluation partition.")
                record = {
                    "id": identifier,
                    "image": str(gold_root / "S1Hand" / image_name),
                    "label": str(gold_root / "LabelHand" / label_name),
                    "country": country,
                    "weight": 1.0,
                }
                if not Path(record["image"]).is_file() or not Path(record["label"]).is_file():
                    raise FileNotFoundError(f"Missing gold SAR image or mask: {identifier}")
                rows.append(record)
        records[split] = rows
    weak_root = base / "data/flood_events/WeaklyLabeled"
    weak, excluded = [], {"bolivia": 0, "gold_overlap": 0}
    weak_images = sorted((weak_root / "S1Weak").glob("*.tif"))
    weak_labels = sorted((weak_root / "S1OtsuLabelWeak").glob("*.tif"))
    if {chip_id(path.name) for path in weak_images} != {chip_id(path.name) for path in weak_labels}:
        raise ValueError("Weak SAR images and masks must have matching chip IDs.")
    for image in weak_images:
        identifier = chip_id(image.name)
        if identifier.startswith("Bolivia_"):
            excluded["bolivia"] += 1
            continue
        if identifier in seen:
            excluded["gold_overlap"] += 1
            continue
        label = weak_root / "S1OtsuLabelWeak" / f"{identifier}_S1OtsuLabelWeak.tif"
        if not label.is_file():
            raise FileNotFoundError(f"Missing weak water mask: {label}")
        weak.append(
            {
                "id": identifier,
                "image": str(image),
                "label": str(label),
                "country": identifier.split("_")[0],
                "weight": None,
            }
        )
    records["weak"] = weak[:weak_max]
    excluded["weak_limit"] = max(0, len(weak) - weak_max)
    return records, excluded


def prepare(settings) -> dict:
    """Validate rasters and fit band statistics using gold training chips only."""
    records, excluded = build_records(settings.flood_root, settings.flood["weak_max"])
    original_counts = {split: len(rows) for split, rows in records.items()}
    empty_chips = {split: [] for split in records}
    coverage = []
    total_pairs = sum(original_counts.values())
    count = np.zeros(2)
    sums = np.zeros(2)
    squares = np.zeros(2)
    for split, rows in records.items():
        for record in rows:
            record["source_stamps"] = {
                key: [Path(record[key]).stat().st_size, Path(record[key]).stat().st_mtime_ns]
                for key in ("image", "label")
            }
            record["source_sha256"] = {key: file_sha256(Path(record[key])) for key in ("image", "label")}
            image, target, _ = read_pair(record)
            record["valid_pixels"] = int((target >= 0).sum())
            record["water_pixels"] = int((target == 1).sum())
            record["image_size"] = list(target.shape)
            if record["valid_pixels"] == 0:
                empty_chips[split].append(record["id"])
            coverage.append(
                {
                    "split": split,
                    "chip_id": record["id"],
                    "country": record["country"],
                    "valid_pixels": record["valid_pixels"],
                    "water_pixels": record["water_pixels"],
                }
            )
            if len(coverage) % 500 == 0:
                print(f"Flood rasters validated: {len(coverage)}/{total_pairs}", flush=True)
            if split == "train":
                valid = target >= 0
                values = np.clip(image[:, valid], -50, 10).astype(np.float64)
                count += values.shape[1]
                sums += values.sum(axis=1)
                squares += np.square(values).sum(axis=1)
    if (count == 0).any():
        raise ValueError("No valid pixels in gold training chips.")
    means = sums / count
    standard_deviations = np.sqrt(np.maximum(squares / count - means**2, 1e-6))
    records = {split: [record for record in rows if record["valid_pixels"] > 0] for split, rows in records.items()}
    if not all(records[split] for split in SPLITS):
        raise ValueError("Every official gold partition must contain valid chips.")
    if "gold_weak" in settings.flood["supervision"] and not records["weak"]:
        raise FileNotFoundError("Download weak SAR images and masks before preparing the full experiment matrix.")
    for record in records["weak"]:
        record["weight"] = settings.flood["weak_weight"]
    manifest = {
        "records": records,
        "mean": means.tolist(),
        "std": standard_deviations.tolist(),
        "excluded": excluded,
        "normalization_source": "gold training chips only",
        "clip_db": [-50, 10],
        "band_order": ["VV", "VH"],
        "units": "dB",
        "alignment_absolute_tolerance": ALIGNMENT_ATOL,
        "counts_before_empty_exclusion": original_counts,
        "empty_chips": empty_chips,
        "split_sha256": {
            split: file_sha256(settings.flood_root / "v1.1/splits/flood_handlabeled" / name)
            for split, name in SPLITS.items()
        },
    }
    manifest["identity"] = fingerprint(manifest)
    write_json(settings.output_root / "flood/manifest.json", manifest)
    write_csv(settings.output_root / "flood/coverage.csv", coverage)
    return manifest


class FloodDataset(Dataset):
    """Read two-channel SAR chips with synchronized spatial augmentation."""

    def __init__(self, records, mean, std, augment=False):
        self.records = records
        self.mean = np.asarray(mean, dtype=np.float32)[:, None, None]
        self.std = np.asarray(std, dtype=np.float32)[:, None, None]
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image, target, _ = read_pair(record)
        image = (np.clip(image, -50, 10) - self.mean) / self.std
        image[:, target < 0] = 0
        image = torch.from_numpy(image.copy())
        target = torch.from_numpy(target[None].copy())
        if self.augment:
            for axis in (-1, -2):
                if torch.rand(()) < 0.5:
                    image, target = image.flip(axis), target.flip(axis)
        return image, target, torch.full_like(target, record["weight"])
