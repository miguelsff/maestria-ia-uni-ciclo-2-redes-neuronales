"""Verify the prepared v2 cohort and record pending human work without approving it."""

from __future__ import annotations

from collections import Counter
from itertools import combinations

from neural_exam.common.io import file_sha256, fingerprint, read_csv, read_json, write_json
from neural_exam.config import load_settings
from neural_exam.wildlife.data import PARTITIONS, _find_annotations
from neural_exam.wildlife.review import label_review_status


def main() -> None:
    settings = load_settings(device="cpu")
    root = settings.output_root
    manifest = read_json(root / "wildlife/manifest.json")
    payload = {key: value for key, value in manifest.items() if key != "identity"}
    if fingerprint(payload) != manifest["identity"]:
        raise ValueError("Prepared manifest identity differs from its content.")
    annotations = _find_annotations(settings.wildlife_annotations_root)
    documents = {split: read_json(annotations / filename) for split, filename in PARTITIONS.items()}
    if manifest["annotation_sha256"] != {
        split: file_sha256(annotations / filename) for split, filename in PARTITIONS.items()
    }:
        raise ValueError("Official ECCV annotations changed after preparation.")
    records = manifest["records"]
    for left, right in combinations(records, 2):
        for field in ("image_id", "sequence_id"):
            if {row[field] for row in records[left]} & {row[field] for row in records[right]}:
                raise ValueError("Images or sequences overlap between prepared partitions.")
    added = [row for row in records["train"] if row.get("origin") == "full_collection_training_background"]
    reserved_ids = {str(row["id"]) for document in documents.values() for row in document["images"]}
    reserved_sequences = {str(row["seq_id"]) for document in documents.values() for row in document["images"]}
    reserved_cameras = {str(row["location"]) for document in documents.values() for row in document["images"]}
    cohort = manifest["background_cohort"]
    if len(added) != cohort["images"] or len(added) != 1000 or any(row["boxes"] for row in added):
        raise ValueError("Expected exactly 1,000 explicitly empty added training frames.")
    if len({row["sequence_id"] for row in added}) != len(added):
        raise ValueError("Added background sequences are not independent.")
    for field, reserved in (
        ("image_id", reserved_ids),
        ("sequence_id", reserved_sequences),
        ("camera_id", reserved_cameras),
    ):
        if {str(row[field]) for row in added} & reserved:
            raise ValueError("Backgrounds reuse a raw ECCV image, sequence or camera.")
    camera_counts = Counter(row["camera_id"] for row in added)
    if len(camera_counts) != 8 or set(camera_counts.values()) != {125}:
        raise ValueError("Background cohort is not balanced across its eight selected cameras.")
    baseline_path = root / "baseline-manifest.json"
    baseline_check = "not available in this independent reproduction"
    if baseline_path.exists():
        baseline = read_json(baseline_path)
        if manifest.get("annotation_corrections"):
            from neural_exam.wildlife.corrections import verify_parent

            changed = verify_parent(manifest, baseline)
            baseline_check = {
                "status": "only registered train boxes changed; holdouts and monitoring identical",
                "changed_images": changed,
            }
        else:
            for split, original_rows in baseline["records"].items():
                current_rows = [
                    row for row in records[split] if row.get("origin") != "full_collection_training_background"
                ]
                if current_rows != original_rows:
                    raise ValueError("Original eligible records differ from the preserved baseline.")
            if baseline["monitoring_frames"] != manifest["monitoring_frames"]:
                raise ValueError("Ecological demonstration cohort changed.")
            baseline_check = {"status": "identical original records and monitoring frames"}
        baseline_check["sha256"] = file_sha256(baseline_path)
    labels = read_csv(root / "wildlife/label_review.csv")
    lookup = {row["image_id"]: row for rows in records.values() for row in rows}
    sampled_backgrounds = [lookup[row["image_id"]] for row in labels if not lookup[row["image_id"]]["boxes"]]
    if len(sampled_backgrounds) != 20 or any(
        row.get("origin") != "full_collection_training_background" for row in sampled_backgrounds
    ):
        raise ValueError("Label-review backgrounds do not come from the new training cohort.")
    object_ids = {box["annotation_id"] for row in records["trans_test"] for box in row["boxes"]}
    occlusion_rows = read_csv(root / "wildlife/occlusion_review.csv")
    if len(occlusion_rows) != len(object_ids) or {row["annotation_id"] for row in occlusion_rows} != object_ids:
        raise ValueError("Occlusion review does not contain the full eligible held-out object cohort.")
    receipt = {
        "status": "prepared; human reviews remain independent requirements",
        "manifest_identity": manifest["identity"],
        "baseline_comparison": baseline_check,
        "official_annotation_sha256": manifest["annotation_sha256"],
        "original_holdouts_preserved": True,
        "all_image_and_sequence_overlaps": 0,
        "additional_backgrounds": len(added),
        "background_cameras": dict(sorted(camera_counts.items())),
        "sampled_training_backgrounds": len(sampled_backgrounds),
        "sampled_background_cameras": dict(Counter(row["camera_id"] for row in sampled_backgrounds)),
        "label_review": label_review_status(root / "wildlife/label_review.csv"),
        "occlusion_objects": len(object_ids),
        "occlusion_level_counts": dict(Counter(row["level"] or "pending" for row in occlusion_rows)),
        "scientific_checkpoint_exists": (root / "wildlife/yolov8s/training_complete.json").exists(),
    }
    write_json(root / "receipts/preflight.json", receipt)
    print(receipt)


if __name__ == "__main__":
    main()
