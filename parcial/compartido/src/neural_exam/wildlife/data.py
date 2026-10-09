"""Audit official CCT20 data and build YOLO labels without copying photographs."""

from __future__ import annotations

import hashlib
import io
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from neural_exam.common.io import file_sha256, fingerprint, read_json, write_csv, write_json
from neural_exam.wildlife.backgrounds import load_background_cohort

PARTITIONS = {
    "train": "train_annotations.json",
    "val": "cis_val_annotations.json",
    "cis_test": "cis_test_annotations.json",
    "trans_val": "trans_val_annotations.json",
    "trans_test": "trans_test_annotations.json",
}


def normalize_box(box: list, width: int, height: int, units: str) -> list[float]:
    """Convert COCO xywh to YOLO centers using the original metadata dimensions."""
    if width <= 0 or height <= 0:
        raise ValueError("Invalid source image dimensions.")
    x, y, w, h = map(float, box)
    if units == "pixels":
        x, w, y, h = x / width, w / width, y / height, h / height
    elif units != "normalized":
        raise ValueError("Bounding-box units must be pixels or normalized.")
    if not np.isfinite([x, y, w, h]).all() or min(x, y) < 0 or min(w, h) <= 0 or x + w > 1.00001 or y + h > 1.00001:
        raise ValueError("Invalid source bounding box; verify it rather than clipping it.")
    return [x + w / 2, y + h / 2, w, h]


def _link_image(source: Path, target: Path) -> None:
    """Use symbolic links only; a Linux volume avoids NTFS link limitations."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise ValueError("Existing image link points to another source; use a fresh output.")
        return
    if target.exists():
        raise ValueError("A regular file occupies the derived image path; no image copies are allowed.")
    try:
        target.symlink_to(source.resolve())
    except OSError as error:
        raise RuntimeError(
            "Cannot link images. Use the persistent Linux dataset volume; copying is disabled."
        ) from error


def _find_annotations(root: Path) -> Path:
    matches = [path for path in (root, root / "eccv_18_annotation_files") if (path / PARTITIONS["train"]).is_file()]
    if len(matches) != 1:
        raise FileNotFoundError("Expected one official annotation directory in CCT_ANNOTATIONS_DIR or CCT_DATA_DIR.")
    return matches[0]


def _decode_source(item: tuple[Path, dict | None]) -> dict:
    path, previous = item
    stamp = {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
    if previous and all(previous.get(key) == value for key, value in stamp.items()):
        return previous
    payload = path.read_bytes()
    with Image.open(io.BytesIO(payload)) as image:
        image.verify()
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        if image.getexif().get(274, 1) != 1:
            raise ValueError(f"Unsupported EXIF rotation in {path.name}; source images remain unchanged.")
        width, height = image.size
    return {**stamp, "width": width, "height": height, "sha256": hashlib.sha256(payload).hexdigest()}


def _audit_sources(image_index: dict, cache_path: Path) -> dict:
    previous = read_json(cache_path) if cache_path.exists() else {}
    entries = [(matches[0], previous.get(name)) for name, matches in sorted(image_index.items())]
    results = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, (entry, decoded) in enumerate(zip(entries, pool.map(_decode_source, entries), strict=True), 1):
            results[entry[0].name] = decoded
            if index % 2000 == 0 or index == len(entries):
                print(f"CCT20 integrity: {index}/{len(entries)}", flush=True)
    write_json(cache_path, results)
    return results


def _summary(records: list[dict], names: list[str]) -> dict:
    counts = Counter(box["class_id"] for record in records for box in record["boxes"])
    return {
        "images": len(records),
        "backgrounds": sum(not record["boxes"] for record in records),
        "cameras": sorted({record["camera_id"] for record in records}),
        "sequences": len({record["sequence_id"] for record in records}),
        "class_boxes": {name: counts[index] for index, name in enumerate(names)},
    }


def _smoke_sample(records: list[dict], class_count: int, per_class: int, backgrounds: int, seed: int) -> list[dict]:
    generator = np.random.default_rng(seed)
    ordered = [records[index] for index in generator.permutation(len(records))]
    chosen = {}
    for class_id in range(class_count):
        candidates = [record for record in ordered if any(box["class_id"] == class_id for box in record["boxes"])]
        chosen.update({record["image_id"]: record for record in candidates[:per_class]})
    chosen.update({record["image_id"]: record for record in [row for row in ordered if not row["boxes"]][:backgrounds]})
    return sorted(chosen.values(), key=lambda record: record["image_id"])


def _materialize(records: dict, names: list[str], dataset_root: Path, source_overrides: dict | None = None) -> dict:
    result = {}
    for split, rows in records.items():
        converted = []
        for row in rows:
            source = Path(row["source"])
            stem = fingerprint(row["image_id"])[:24]
            image = dataset_root / "images" / split / f"{stem}{source.suffix.lower()}"
            _link_image((source_overrides or {}).get(str(source), source), image)
            label = dataset_root / "labels" / split / f"{stem}.txt"
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text(
                "".join(
                    f"{box['class_id']} " + " ".join(f"{value:.8f}" for value in box["box"]) + "\n"
                    for box in row["boxes"]
                ),
                encoding="utf-8",
            )
            converted.append({**row, "image": str(image.absolute())})
        result[split] = converted
        image_list = dataset_root / "lists" / f"{split}.txt"
        image_list.parent.mkdir(parents=True, exist_ok=True)
        image_list.write_text("".join(record["image"] + "\n" for record in converted), encoding="utf-8")
    base = {"path": str(dataset_root.absolute()), "train": "lists/train.txt", "val": "lists/val.txt", "names": names}
    (dataset_root / "data.yaml").write_text(yaml.safe_dump(base), encoding="utf-8")
    for split in records.keys() - {"train", "val"}:
        (dataset_root / f"{split}.yaml").write_text(
            yaml.safe_dump({**base, "test": f"lists/{split}.txt"}), encoding="utf-8"
        )
    return result


def prepare(settings, quick: bool = False) -> dict:
    """Validate images, annotations and partitions, then produce linked YOLO data."""
    image_root = getattr(settings, "wildlife_images_root", None) or settings.wildlife_root
    annotation_root = _find_annotations(getattr(settings, "wildlife_annotations_root", None) or settings.wildlife_root)
    documents = {split: read_json(annotation_root / filename) for split, filename in PARTITIONS.items()}
    hashes = {split: file_sha256(annotation_root / filename) for split, filename in PARTITIONS.items()}
    background_path = settings.wildlife.get("background_manifest")
    extra_backgrounds, background_provenance = (
        load_background_cohort(Path(background_path), documents, hashes) if background_path else ([], None)
    )
    schema = {int(category["id"]): category["name"] for category in documents["train"]["categories"]}
    if any(
        {int(category["id"]): category["name"] for category in doc["categories"]} != schema
        for doc in documents.values()
    ):
        raise ValueError("Category schemas differ between partitions.")
    categories = {identifier: name for identifier, name in schema.items() if name.casefold() not in {"empty", "car"}}
    class_map = {identifier: index for index, identifier in enumerate(sorted(categories))}
    names = [categories[identifier] for identifier in sorted(categories)]
    expected = settings.wildlife.get("expected_classes")
    if expected is not None and len(names) != expected:
        raise ValueError(f"Expected {expected} animal categories, found {len(names)}.")
    boxes = [
        a["bbox"]
        for doc in documents.values()
        for a in doc["annotations"]
        if int(a["category_id"]) in class_map and isinstance(a.get("bbox"), list) and len(a["bbox"]) == 4
    ]
    if not boxes:
        raise ValueError("No animal bounding boxes in the annotations.")
    declared = settings.wildlife.get("bbox_units", "auto")
    units = (
        ("normalized" if max(float(value) for box in boxes for value in box) <= 1.00001 else "pixels")
        if declared == "auto"
        else declared
    )
    image_index = {}
    for path in image_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            image_index.setdefault(path.name, []).append(path)
    for row in extra_backgrounds:
        path = Path(background_path).parent / "images" / row["file_name"]
        image_index.setdefault(path.name, []).append(path)
    if any(len(matches) != 1 for matches in image_index.values()):
        raise ValueError("Duplicate filenames in the configured image root.")
    referenced = {Path(image["file_name"]).name for doc in documents.values() for image in doc["images"]}
    referenced.update(row["file_name"] for row in extra_backgrounds)
    missing = referenced - image_index.keys()
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} referenced images; first: {sorted(missing)[:3]}.")
    cache_root = settings.weights_root.parent if hasattr(settings, "weights_root") else settings.output_root / "cache"
    source_audit = _audit_sources(
        {name: image_index[name] for name in referenced}, cache_root / "wildlife-integrity.json"
    )
    train_sequences = {str(image["seq_id"]) for image in documents["train"]["images"]}
    cameras = {split: {str(image["location"]) for image in doc["images"]} for split, doc in documents.items()}
    if cameras["train"] & (cameras["trans_val"] | cameras["trans_test"]):
        raise ValueError("Unseen cameras overlap training cameras.")
    if not (cameras["val"] | cameras["cis_test"]) <= cameras["train"]:
        raise ValueError("A seen-camera partition contains an unseen camera.")
    records, exclusions, monitoring_frames = {}, [], []
    seen_images, seen_annotations = set(), set()
    for split, document in documents.items():
        by_image = {}
        split_ids = {str(image["id"]) for image in document["images"]}
        if len(split_ids) != len(document["images"]):
            raise ValueError(f"Duplicate image IDs in {split}.")
        for annotation in document["annotations"]:
            identifier = str(annotation["id"])
            if identifier in seen_annotations or str(annotation["image_id"]) not in split_ids:
                raise ValueError("Duplicate annotation ID or reference to an unknown image.")
            if int(annotation["category_id"]) not in schema:
                raise ValueError("Unknown annotation category.")
            seen_annotations.add(identifier)
            by_image.setdefault(str(annotation["image_id"]), []).append(annotation)
        rows = []
        for image in document["images"]:
            identifier = str(image["id"])
            if identifier in seen_images:
                raise ValueError("Image shared between official partitions.")
            seen_images.add(identifier)
            filename = Path(image["file_name"]).name
            source = image_index[filename][0]
            actual = source_audit[filename]
            width, height = int(image["width"]), int(image["height"])
            if (
                min(width, height) <= 0
                or abs(width / height - actual["width"] / actual["height"]) > 2 / actual["height"]
            ):
                raise ValueError(f"Metadata and reduced-image aspect ratios differ for {filename}.")
            base = {
                "image_id": identifier,
                "source": str(source.absolute()),
                "image": str(source.absolute()),
                "width": actual["width"],
                "height": actual["height"],
                "original_width": width,
                "original_height": height,
                "camera_id": str(image["location"]),
                "sequence_id": str(image["seq_id"]),
                "timestamp": image.get("date_captured", ""),
                "frame_number": image.get("frame_num"),
                "sequence_frames": image.get("seq_num_frames"),
                "source_size": actual["bytes"],
                "source_mtime_ns": actual["mtime_ns"],
                "source_sha256": actual["sha256"],
            }
            if split == "trans_test":
                monitoring_frames.append({**base, "boxes": []})
            if split == "val" and base["sequence_id"] in train_sequences:
                exclusions.append({"split": split, "image_id": identifier, "reason": "sequence_shared_with_train"})
                continue
            annotations = by_image.get(identifier, [])
            animals = [a for a in annotations if int(a["category_id"]) in class_map]
            empty = any(schema[int(a["category_id"])].casefold() == "empty" for a in annotations)
            if animals and empty:
                raise ValueError("An image is marked empty and contains an animal.")
            reason = None
            if not animals and not empty:
                reason = "no_animal_box_or_explicit_empty"
            elif any(not isinstance(a.get("bbox"), list) or len(a["bbox"]) != 4 for a in animals):
                reason = "incomplete_animal_boxes"
            converted = []
            if reason is None:
                try:
                    converted = [
                        {
                            "class_id": class_map[int(a["category_id"])],
                            "box": normalize_box(a["bbox"], width, height, units),
                            "annotation_id": str(a["id"]),
                        }
                        for a in animals
                    ]
                except (ValueError, TypeError):
                    reason = "invalid_bbox"
            if reason:
                exclusions.append({"split": split, "image_id": identifier, "reason": reason})
            else:
                rows.append({**base, "boxes": converted})
        if not rows:
            raise ValueError(f"No eligible images in {split}.")
        records[split] = rows
    for row in extra_backgrounds:
        source = image_index[row["file_name"]][0]
        actual = source_audit[row["file_name"]]
        if (actual["width"], actual["height"]) != (int(row["width"]), int(row["height"])):
            raise ValueError("Background dimensions changed after cohort verification.")
        records["train"].append(
            {
                "image_id": str(row["id"]),
                "source": str(source.absolute()),
                "image": str(source.absolute()),
                "width": actual["width"],
                "height": actual["height"],
                "original_width": int(row["width"]),
                "original_height": int(row["height"]),
                "camera_id": str(row["location"]),
                "sequence_id": str(row["seq_id"]),
                "timestamp": row.get("date_captured", ""),
                "frame_number": row.get("frame_num"),
                "sequence_frames": row.get("seq_num_frames"),
                "source_size": actual["bytes"],
                "source_mtime_ns": actual["mtime_ns"],
                "source_sha256": actual["sha256"],
                "origin": "full_collection_training_background",
                "boxes": [],
            }
        )
    correction_provenance = None
    if settings.wildlife.get("annotation_corrections"):
        from neural_exam.wildlife.corrections import apply_corrections

        correction_provenance = apply_corrections(records, names, Path(settings.wildlife["annotation_corrections"]))
    sequences = {split: {row["sequence_id"] for row in rows} for split, rows in records.items()}
    overlaps = {f"{a}:{b}": len(sequences[a] & sequences[b]) for a, b in combinations(records, 2)}
    if any(overlaps.values()):
        raise ValueError(f"Sequence leakage remains after depuration: {overlaps}.")
    accepted_cameras = {row["camera_id"] for row in records["train"]}
    if any(row["camera_id"] not in accepted_cameras for split in ("val", "cis_test") for row in records[split]):
        raise ValueError("A seen camera has no eligible training images.")
    summaries = {split: _summary(rows, names) for split, rows in records.items()}
    output = settings.output_root / "wildlife"
    if quick:
        seed = settings.training["seed"]
        records = {
            "train": _smoke_sample(records["train"], len(names), 4, 20, seed),
            "val": _smoke_sample(records["val"], len(names), 2, 8, seed + 1),
        }
        monitoring_frames = []
    jpeg_provenance, source_overrides = None, None
    if settings.wildlife.get("canonical_jpeg_eoi", False):
        from neural_exam.wildlife.jpeg import prepare_jpeg_sources

        source_overrides, jpeg_provenance = prepare_jpeg_sources(records, output / "dataset/jpeg_sources")
        write_json(output / "jpeg_materialization.json", jpeg_provenance)
    records = _materialize(records, names, output / "dataset", source_overrides)
    manifest = {
        "schema_version": 2,
        "records": records,
        "names": names,
        "bbox_units": units,
        "class_map": [
            {"source_id": identifier, "class_id": class_map[identifier], "name": categories[identifier]}
            for identifier in sorted(categories)
        ],
        "annotation_sha256": hashes,
        "images_root": str(image_root),
        "annotations_root": str(annotation_root),
        "exclusions": exclusions,
        "train_class_boxes": summaries["train"]["class_boxes"],
        "split_summary": summaries,
        "sequence_overlaps": overlaps,
        "raw_split_images": {split: len(doc["images"]) for split, doc in documents.items()},
        "integrity": {
            "referenced_images": len(referenced),
            "unreferenced_files": len(image_index.keys() - referenced),
            "fully_decoded": True,
            "cached_by_size_mtime": True,
        },
        "monitoring_frames": monitoring_frames,
        "scientific_run": not quick,
        "dataset": "CCT20 ECCV benchmark; empty and car are not animal classes",
    }
    if background_provenance:
        manifest["background_cohort"] = background_provenance
    if correction_provenance:
        manifest["annotation_corrections"] = correction_provenance
    if jpeg_provenance:
        manifest["jpeg_materialization"] = {key: value for key, value in jpeg_provenance.items() if key != "copies"}
    manifest["identity"] = fingerprint(manifest)
    write_json(output / "manifest.json", manifest)
    write_csv(output / "exclusions.csv", exclusions, ["split", "image_id", "reason"])
    write_csv(output / "class_map.csv", manifest["class_map"])
    write_csv(
        output / "class_balance.csv",
        [{"species": name, "training_boxes": summaries["train"]["class_boxes"][name]} for name in names],
    )
    write_csv(
        output / "split_audit.csv",
        [
            {
                "split": split,
                "raw_images": len(documents[split]["images"]),
                "eligible_images": summary["images"],
                "backgrounds": summary["backgrounds"],
                "cameras": len(summary["cameras"]),
                "sequences": summary["sequences"],
                "additional_training_backgrounds": len(extra_backgrounds) if split == "train" else 0,
            }
            for split, summary in summaries.items()
        ],
    )
    return manifest
