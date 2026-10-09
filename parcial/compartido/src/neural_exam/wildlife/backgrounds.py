"""Select genuine training backgrounds without reusing any ECCV holdout frame."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from neural_exam.common.io import file_sha256, fingerprint, read_json

METADATA_URL = "https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/labels/caltech_camera_traps.json.zip"
SPLITS_URL = "https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/CaltechCameraTrapsSplits_v0.json"
IMAGE_BASE = "https://storage.googleapis.com/public-datasets-lila/caltech-unzipped/cct_images/"


def select_backgrounds(
    metadata: dict, splits: dict, documents: dict, cameras: int = 8, per_camera: int = 125, seed: int = 42
) -> list[dict]:
    """Select one empty frame per new sequence from official full-collection train cameras.

    Camera and frame ranking use only metadata and the fixed seed. All raw ECCV
    images and sequences, including excluded annotations, remain protected.
    """
    if cameras < 1 or per_camera < 1:
        raise ValueError("Background camera and frame counts must be positive.")
    empty = {int(row["id"]) for row in metadata["categories"] if row["name"].casefold() == "empty"}
    if len(empty) != 1:
        raise ValueError("Expected one explicit empty category in full-collection metadata.")
    labels = defaultdict(set)
    for row in metadata["annotations"]:
        labels[str(row["image_id"])].add(int(row["category_id"]))
    reserved_images = {str(row["id"]) for doc in documents.values() for row in doc["images"]}
    reserved_sequences = {str(row["seq_id"]) for doc in documents.values() for row in doc["images"]}
    benchmark_cameras = {str(row["location"]) for doc in documents.values() for row in doc["images"]}
    official = splits["splits"]
    allowed = set(map(str, official["train"])) - set(map(str, official["val"]))
    allowed -= set(map(str, official["eccv_val"])) | benchmark_cameras
    grouped = defaultdict(dict)
    seen = set()
    for row in metadata["images"]:
        identifier, camera, sequence = str(row["id"]), str(row["location"]), str(row["seq_id"])
        if identifier in seen:
            raise ValueError("Duplicate full-collection image IDs.")
        seen.add(identifier)
        if (
            camera not in allowed
            or identifier in reserved_images
            or sequence in reserved_sequences
            or labels[identifier] != empty
        ):
            continue
        filename = row["file_name"]
        if Path(filename).name != filename or "\\" in filename:
            raise ValueError("Unexpected background image path.")
        if min(int(row["width"]), int(row["height"])) <= 0:
            raise ValueError("Background metadata contains invalid dimensions.")
        rank = fingerprint([seed, identifier])
        prior = grouped[camera].get(sequence)
        if prior is None or rank < prior[0]:
            grouped[camera][sequence] = (rank, row)
    eligible = [camera for camera, sequences in grouped.items() if len(sequences) >= per_camera]
    eligible.sort(key=lambda camera: fingerprint([seed, "camera", camera]))
    if len(eligible) < cameras:
        raise ValueError("Not enough independent official training cameras for the background cohort.")
    chosen = []
    for camera in eligible[:cameras]:
        ranked = sorted(grouped[camera].values(), key=lambda item: item[0])
        chosen.extend(dict(row) for _, row in ranked[:per_camera])
    return sorted(chosen, key=lambda row: str(row["id"]))


def load_background_cohort(path: Path, documents: dict, annotation_hashes: dict) -> tuple[list[dict], dict]:
    """Reconstruct and verify a downloaded cohort before it can enter training."""
    cohort = read_json(path)
    if cohort["annotation_sha256"] != annotation_hashes:
        raise ValueError("Background cohort belongs to different ECCV annotations.")
    root = path.parent
    for name, expected in cohort["metadata_sha256"].items():
        if Path(name).name != name or file_sha256(root / name) != expected:
            raise ValueError("Background source metadata changed.")
    selected = select_backgrounds(
        read_json(root / "full_metadata.json"),
        read_json(root / "CaltechCameraTrapsSplits_v0.json"),
        documents,
        **cohort["selection"],
    )
    expected = {str(row["id"]): row for row in selected}
    recorded = {str(row["id"]): row for row in cohort["images"]}
    if len(recorded) != len(cohort["images"]) or expected.keys() != recorded.keys():
        raise ValueError("Background cohort differs from its deterministic selection.")
    for identifier, row in recorded.items():
        original = expected[identifier]
        if any(row[key] != value for key, value in original.items()):
            raise ValueError("Background image metadata differs from the official source.")
        image = root / "images" / row["file_name"]
        if not image.is_file() or file_sha256(image) != row["sha256"]:
            raise ValueError("Missing or changed background photograph.")
    return cohort["images"], {
        "manifest_sha256": file_sha256(path),
        "metadata_sha256": cohort["metadata_sha256"],
        "selection": cohort["selection"],
        "images": len(recorded),
        "cameras": sorted({str(row["location"]) for row in selected}),
        "policy": "Official full-collection train cameras outside all ECCV cameras; no ECCV image or sequence reused.",
        "source": METADATA_URL,
    }
