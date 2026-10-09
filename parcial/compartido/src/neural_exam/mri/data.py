"""Study selection and DICOM preprocessing; slices never define independent splits."""

from __future__ import annotations

import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pydicom
import torch
from pydicom.pixels import apply_modality_lut
from torch.nn import functional as F
from torch.utils.data import Dataset

from neural_exam.common.io import fingerprint, read_csv, write_csv, write_json
from neural_exam.mri.labels import LABELS, extract_labels


def find_dicom_root(root: Path) -> Path:
    """Resolve direct or one-directory-deep Kaggle extraction without walking all slices."""
    candidates = [root] if root.name == "train_series" else [root / "train_series"]
    if root.is_dir():
        candidates += [child / "train_series" for child in root.iterdir() if child.is_dir()]
    matches = list(dict.fromkeys(path.resolve() for path in candidates if path.is_dir()))
    if len(matches) != 1:
        raise FileNotFoundError(
            "Set RSNA_DATA_DIR to the extracted directory containing exactly one train_series folder."
        )
    return matches[0]


def is_gold(row: dict) -> bool:
    """Require all twelve expert labels; blank values are never interpreted as zero."""
    try:
        return all(float(row[label]) in (0.0, 1.0) for label in LABELS)
    except (ValueError, KeyError, TypeError):
        return False


def select_studies(studies: list[dict], maximum: int, seed: int, validation_fraction: float) -> list[dict]:
    """Include every gold study and greedily cover rare report-derived positive labels."""
    ids = [row["StudyInstanceUID"] for row in studies]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate StudyInstanceUID in metadata.")
    gold, weak = [], []
    for row in studies:
        extracted = extract_labels(row.get("Report", ""))
        record = {"study_id": row["StudyInstanceUID"], "weak": extracted}
        if is_gold(row):
            record.update(split="gold", targets=[float(row[label]) for label in LABELS])
            gold.append(record)
        elif extracted["supported"]:
            record.update(targets=extracted["values"])
            weak.append(record)
    if maximum <= len(gold) + 2:
        raise ValueError("Study budget must include all expert studies and at least three weak studies.")
    generator = np.random.default_rng(seed)
    generator.shuffle(weak)
    positives = np.array([[value == 1 for value in row["targets"]] for row in weak], dtype=int)
    chosen, coverage = [], np.zeros(12)
    # A deterministic rarity score handles multilabel data without pretending single-label stratification.
    frequencies = positives.sum(axis=0).clip(min=1) if len(weak) else np.ones(12)
    available = list(range(len(weak)))
    while available and len(chosen) < maximum - len(gold):
        index = max(available, key=lambda item: float((positives[item] / (frequencies * (coverage + 1))).sum()))
        chosen.append(weak[index])
        coverage += positives[index]
        available.remove(index)
    if len(chosen) < 3:
        raise ValueError("Insufficient supported reports for weak training and validation.")
    # Match weak-label validation quotas while retaining positive training evidence.
    validation_size = max(1, min(len(chosen) - 1, round(len(chosen) * validation_fraction)))
    validation = set()
    val_coverage = np.zeros(12)
    selected_positives = np.array([[value == 1 for value in row["targets"]] for row in chosen])
    totals = selected_positives.sum(0)
    quotas = np.minimum(np.maximum(1, np.round(totals * validation_fraction)), np.maximum(0, totals - 1))
    scale = np.maximum(1, quotas)
    for _ in range(validation_size):
        eligible = [
            i
            for i in range(len(chosen))
            if i not in validation and np.all(val_coverage + selected_positives[i] <= np.maximum(0, totals - 1))
        ]
        if not eligible:
            raise ValueError("Cannot construct the requested validation size while retaining training positives.")
        index = max(
            eligible,
            key=lambda i: float((selected_positives[i] * (2 * (quotas - val_coverage) - 1) / scale).sum()),
        )
        validation.add(index)
        val_coverage += selected_positives[index]
    if np.any((totals > 0) & (totals - val_coverage < 1)):
        raise ValueError("Positive weak-label support was lost from training.")
    for index, record in enumerate(chosen):
        record["split"] = "val" if index in validation else "train"
    return gold + chosen


def choose_series(rows: list[dict]) -> dict:
    """Prefer sagittal fluid-sensitive fat-suppressed series; record any fallback."""
    if not rows:
        raise ValueError("Study has no series descriptor.")
    return min(
        rows,
        key=lambda row: (
            row.get("Anatomical_Plane") != "Sagittal",
            row.get("Fluid_Sensitive") not in {"1", "1.0"},
            row.get("Fat_Suppression") not in {"1", "1.0"},
            row["SeriesInstanceUID"],
        ),
    )


def decode_stacks(series_root: Path, stacks: int, image_size: int) -> tuple[np.ndarray, dict]:
    """Decode ordered MRI slices, including JPEG lossless/JPEG 2000 when plugins exist."""
    files = list(series_root.glob("*.dcm"))
    if len(files) < 3:
        raise FileNotFoundError(f"Expected at least three extracted DICOM slices in {series_root}.")
    headers = [(path, pydicom.dcmread(path, stop_before_pixels=True)) for path in files]
    instance_ids = [str(header.get("SOPInstanceUID", path.name)) for path, header in headers]
    if len(instance_ids) != len(set(instance_ids)):
        raise ValueError(f"Duplicate DICOM instance in {series_root}.")
    geometric = all("ImagePositionPatient" in header and "ImageOrientationPatient" in header for _, header in headers)
    if geometric:
        orientation = np.asarray(headers[0][1].ImageOrientationPatient, dtype=float)
        normal = np.cross(orientation[:3], orientation[3:])
        if np.linalg.norm(normal) < 0.99:
            raise ValueError("Invalid DICOM orientation vectors.")
        if any(
            not np.allclose(np.asarray(header.ImageOrientationPatient, dtype=float), orientation, atol=0.01)
            for _, header in headers
        ):
            raise ValueError("Mixed slice orientations in a single MRI series.")
        headers.sort(key=lambda item: float(np.dot(np.asarray(item[1].ImagePositionPatient, dtype=float), normal)))
        positions = np.array(
            [np.dot(np.asarray(header.ImagePositionPatient, dtype=float), normal) for _, header in headers]
        )
        distances = np.diff(positions)
        if (distances <= 1e-5).any():
            raise ValueError("Repeated slice position; use an unambiguous single-stack series.")
        if distances.max() > 2.5 * np.median(distances):
            raise ValueError("Large slice gap; verify the extraction and acquisition before using the series.")
    else:
        warnings.warn("Missing DICOM geometry: falling back to InstanceNumber.", stacklevel=2)
        if any("InstanceNumber" not in header for _, header in headers):
            raise ValueError("DICOM slices cannot be ordered reliably.")
        headers.sort(key=lambda item: int(item[1].InstanceNumber))
    arrays = []
    for path, _ in headers:
        dataset = pydicom.dcmread(path)
        pixels = apply_modality_lut(dataset.pixel_array, dataset).astype(np.float32)
        if pixels.ndim != 2:
            raise ValueError(f"Expected a single 2D DICOM instance: {path}")
        if str(dataset.get("PhotometricInterpretation", "")) == "MONOCHROME1":
            pixels = pixels.max() + pixels.min() - pixels
        arrays.append(pixels)
    volume = np.stack(arrays)
    if not np.isfinite(volume).all():
        raise ValueError("DICOM intensities contain non-finite values.")
    lower, upper = np.percentile(volume, (0.5, 99.5))
    volume = np.clip((volume - float(lower)) / max(float(upper - lower), 1e-6), 0, 1).astype(np.float32)
    centers = np.linspace(1, len(volume) - 2, stacks).round().astype(int)
    sampled = np.stack([volume[center - 1 : center + 2] for center in centers])
    resized = F.interpolate(
        torch.from_numpy(sampled), size=(image_size, image_size), mode="bilinear", align_corners=False
    )
    return resized.numpy(), {
        "slice_count": len(files),
        "geometric_sort": geometric,
        "transfer_syntaxes": sorted({str(header.file_meta.TransferSyntaxUID) for _, header in headers}),
    }


def prepare(settings) -> dict:
    """Validate selected studies and cache MRI tensors outside the source repository."""
    studies = read_csv(settings.metadata_root / "train.csv")
    descriptors = read_csv(settings.metadata_root / "train_series.csv")
    if not set(["StudyInstanceUID", "Report", *LABELS]).issubset(studies[0] if studies else {}):
        raise ValueError("Unexpected RSNA train.csv schema.")
    root = find_dicom_root(settings.mri_root)
    selected = select_studies(
        studies, settings.mri["max_studies"], settings.training["seed"], settings.mri["validation_fraction"]
    )
    series_by_study = {}
    for descriptor in descriptors:
        series_by_study.setdefault(descriptor["StudyInstanceUID"], []).append(descriptor)
    output = settings.output_root / "mri"
    cache = output / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(selected, 1):
        descriptor = choose_series(series_by_study.get(record["study_id"], []))
        source = root / record["study_id"] / descriptor["SeriesInstanceUID"]
        stamps = sorted((path.name, path.stat().st_size, path.stat().st_mtime_ns) for path in source.glob("*.dcm"))
        identity = fingerprint(
            {
                "files": stamps,
                "size": settings.mri["image_size"],
                "stacks": settings.mri["stacks"],
                "preprocessing": "percentile-float32-v2",
            }
        )
        cache_file = cache / f"{record['study_id']}_{identity[:12]}.npz"
        if not cache_file.exists():
            image, details = decode_stacks(source, settings.mri["stacks"], settings.mri["image_size"])
            np.savez_compressed(cache_file, image=image, details=np.array(str(details)))
        record.update(
            cache=str(cache_file),
            series_id=descriptor["SeriesInstanceUID"],
            plane=descriptor["Anatomical_Plane"],
            fluid_sensitive=descriptor["Fluid_Sensitive"],
            fat_suppression=descriptor["Fat_Suppression"],
            source_identity=identity,
        )
        if index % 25 == 0 or index == len(selected):
            print(f"Validated and cached MRI series {index}/{len(selected)}", flush=True)
    manifest = {
        "records": selected,
        "labels": LABELS,
        "identity": fingerprint(selected),
        "split_counts": dict(Counter(record["split"] for record in selected)),
        "report_script_counts": dict(Counter(record["weak"]["script"] for record in selected)),
    }
    write_json(output / "manifest.json", manifest)
    write_csv(
        output / "prevalence.csv",
        [
            {
                "split": split,
                "label": label,
                "positive": sum(record["targets"][index] == 1 for record in selected if record["split"] == split),
                "observed": sum(
                    record["targets"][index] is not None for record in selected if record["split"] == split
                ),
            }
            for split in ("train", "val", "gold")
            for index, label in enumerate(LABELS)
        ],
    )
    return manifest


class MRIDataset(Dataset):
    """Load cached adjacent-slice stacks and independent label confidence masks."""

    def __init__(self, records, architecture="scratch", noise_strategy="weighted", confidence_threshold=0.8):
        self.records = records
        self.architecture = architecture
        self.noise_strategy = noise_strategy
        self.confidence_threshold = confidence_threshold

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with np.load(record["cache"], allow_pickle=False) as cache:
            image = torch.from_numpy(cache["image"].copy())
        if self.architecture == "resnet50":
            mean = image.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
            std = image.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
            image = (image - mean) / std
        target = torch.tensor(
            [float("nan") if value is None else value for value in record["targets"]], dtype=torch.float32
        )
        confidence = torch.ones(12) if record["split"] == "gold" else torch.tensor(record["weak"]["confidence"])
        if self.noise_strategy == "high_confidence" and record["split"] != "gold":
            confidence = (confidence >= self.confidence_threshold).float()
        return image, target, confidence
