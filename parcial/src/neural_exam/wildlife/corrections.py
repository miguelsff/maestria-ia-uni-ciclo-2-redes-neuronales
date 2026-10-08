"""Apply explicit train-only annotation proposals and migrate unchanged reviews."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from neural_exam.common.io import file_sha256, fingerprint, read_csv, read_json, write_csv, write_json


def apply_corrections(records: dict, names: list[str], path: Path) -> dict:
    """Verify original targets and apply proposed boxes without approving a review."""
    document = read_json(path)
    patches = document["patches"]
    if document.get("schema_version") != 1 or not patches:
        raise ValueError("Expected a nonempty version-one annotation correction registry.")
    identifiers = [patch["image_id"] for patch in patches]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate correction image IDs.")
    lookup = {row["image_id"]: row for row in records["train"]}
    annotation_ids = {box["annotation_id"] for rows in records.values() for row in rows for box in row["boxes"]}
    for patch in patches:
        if patch.get("split") != "train" or patch["image_id"] not in lookup:
            raise ValueError("Corrections may only reference eligible training images.")
        record = lookup[patch["image_id"]]
        if record["source_sha256"] != patch["source_sha256"] or record["boxes"] != patch["original_boxes"]:
            raise ValueError("Correction source photograph or original targets changed.")
        if not patch.get("reason", "").strip() or patch.get("proposal_kind") != "agent_proposal_requires_human_review":
            raise ValueError("Corrections require a reason and explicit unapproved proposal provenance.")
        previous_ids = {box["annotation_id"] for box in record["boxes"]}
        converted = []
        for box in patch["replacement_boxes"]:
            class_id = names.index(box["category"])
            cx, cy, width, height = box["box"]
            if (
                not np.isfinite([cx, cy, width, height]).all()
                or min(width, height) <= 0
                or min(cx - width / 2, cy - height / 2) < 0
                or max(cx + width / 2, cy + height / 2) > 1
            ):
                raise ValueError("Proposed boxes must remain inside the photograph.")
            identifier = box["annotation_id"]
            if identifier in annotation_ids and identifier not in previous_ids:
                raise ValueError("Correction annotation ID belongs to another object.")
            if identifier in {item["annotation_id"] for item in converted}:
                raise ValueError("Duplicate replacement annotation ID.")
            converted.append({"class_id": class_id, "box": box["box"], "annotation_id": identifier})
        if not converted:
            raise ValueError("Animal corrections cannot turn a photograph into a background.")
        annotation_ids.update(box["annotation_id"] for box in converted)
        record["boxes"] = converted
    parent_changes = document.get("parent_changed_image_ids", identifiers)
    if len(set(parent_changes)) != len(parent_changes) or not set(parent_changes) <= set(identifiers):
        raise ValueError("Parent changes must be unique registered correction images.")
    return {
        "registry_sha256": file_sha256(path),
        "parent_manifest_identity": document["parent_manifest_identity"],
        "image_ids": identifiers,
        "parent_changed_image_ids": parent_changes,
        "approval": "proposals only; current-manifest human label review is mandatory before training",
    }


def target_identity(record: dict, names: list[str]) -> str:
    """Identify visual targets independently of derived filesystem paths."""
    return fingerprint({"image_sha256": record["source_sha256"], "boxes": record["boxes"], "names": names})


def verify_parent(manifest: dict, parent: dict) -> list[str]:
    """Require identical holdouts and permit only registered training box changes."""
    provenance = manifest.get("annotation_corrections", {})
    if provenance.get("parent_manifest_identity") != parent["identity"]:
        raise ValueError("Correction registry belongs to another parent manifest.")
    changed = []
    if manifest["names"] != parent["names"] or manifest["monitoring_frames"] != parent["monitoring_frames"]:
        raise ValueError("Categories or ecological monitoring frames changed.")
    for split, rows in parent["records"].items():
        current = manifest["records"][split]
        if [row["image_id"] for row in current] != [row["image_id"] for row in rows]:
            raise ValueError("Correction changed partition membership or order.")
        for before, after in zip(rows, current, strict=True):
            left, right = copy.deepcopy(before), copy.deepcopy(after)
            left.pop("image", None)
            right.pop("image", None)
            if left == right:
                continue
            left_boxes, right_boxes = left.pop("boxes"), right.pop("boxes")
            if split != "train" or left != right or left_boxes == right_boxes:
                raise ValueError("Only training boxes may differ from the parent.")
            changed.append(before["image_id"])
    expected = provenance.get("parent_changed_image_ids", provenance.get("image_ids", []))
    if sorted(changed) != sorted(expected):
        raise ValueError("Actual changes differ from the correction registry.")
    return changed


def migrate_reviews(manifest: dict, output: Path, parent_root: Path) -> dict:
    """Carry genuine approvals only when the same sampled targets are unchanged."""
    receipt_path = output / "review_migration.json"
    parent_path = parent_root / "manifest.json"
    parent = read_json(parent_path)
    if fingerprint({key: value for key, value in parent.items() if key != "identity"}) != parent["identity"]:
        raise ValueError("Parent manifest identity is invalid.")
    changed = verify_parent(manifest, parent)
    parent_sample = read_json(parent_root / "label_review_sample.json")
    if parent_sample["manifest_identity"] != parent["identity"]:
        raise ValueError("Parent visual sample belongs to another identity.")
    previous = read_csv(parent_root / "label_review.csv")
    current = read_csv(output / "label_review.csv")
    previous_ids = sorted(row["image_id"] for row in previous)
    if previous_ids != parent_sample["image_ids"] or len(set(previous_ids)) != len(previous_ids):
        raise ValueError("Parent visual sample is incomplete or duplicated.")
    if sorted(row["image_id"] for row in current) != previous_ids:
        raise ValueError("Review migration requires identical sampled image IDs.")
    identity = {
        "parent_manifest": parent["identity"],
        "current_manifest": manifest["identity"],
        "parent_csv_sha256": file_sha256(parent_root / "label_review.csv"),
    }
    if receipt_path.exists():
        saved = read_json(receipt_path)
        if any(saved[key] != value for key, value in identity.items()):
            raise ValueError("Review migration inputs changed; use a new output.")
        return saved
    if any(row["status"] != "pending" or row["reviewer"].strip() for row in current):
        raise ValueError("Migration cannot overwrite a current human review.")
    old_records = {row["image_id"]: row for rows in parent["records"].values() for row in rows}
    new_records = {row["image_id"]: row for rows in manifest["records"].values() for row in rows}
    old_rows = {row["image_id"]: row for row in previous}
    carried = []
    for row in current:
        old = old_rows[row["image_id"]]
        same = target_identity(old_records[row["image_id"]], parent["names"]) == target_identity(
            new_records[row["image_id"]], manifest["names"]
        )
        if same:
            for key in ("status", "reviewer", "visual_evidence"):
                row[key] = old[key]
            carried.append(row["image_id"])
        # Changed targets stay pending with no assigned reviewer; old judgment is archived separately.
    write_csv(output / "label_review.csv", current)
    write_csv(output / "parent_label_review.csv", previous)
    receipt = {**identity, "unchanged_rows_carried": len(carried), "changed_images_requiring_review": changed}
    write_json(receipt_path, receipt)
    return receipt
