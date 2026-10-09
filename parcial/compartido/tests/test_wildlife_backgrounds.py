"""Synthetic fixtures verify background provenance and full occlusion coverage."""

import json
from collections import Counter

import pytest
from PIL import Image
from test_wildlife_data import fixture_settings
from test_wildlife_workflow import predictions, records

from neural_exam.common.io import file_sha256, read_csv, write_csv, write_json
from neural_exam.wildlife.backgrounds import load_background_cohort, select_backgrounds
from neural_exam.wildlife.data import PARTITIONS, prepare
from neural_exam.wildlife.pipeline import robustness_metrics
from neural_exam.wildlife.review import prepare_label_review, prepare_occlusion_review


def sources(tmp_path):
    settings = fixture_settings(tmp_path)
    documents = {key: json.loads((settings.wildlife_root / name).read_text()) for key, name in PARTITIONS.items()}
    images = [
        {"id": identifier, "file_name": f"{identifier}.jpg", "location": camera, "seq_id": sequence, "width": 20, "height": 20}
        for identifier, camera, sequence in [
            ("extra_a", "fresh_train", "extra_sequence_a"),
            ("extra_b", "fresh_train", "extra_sequence_b"),
            ("extra_test", "trans_test", "extra_sequence_test"),
            ("extra_wrong", "fresh_train", "val_regular"),
            ("extra_not_empty", "fresh_train", "extra_sequence_c"),
        ]
    ]
    metadata = {
        "images": images,
        "categories": [{"id": 30, "name": "empty"}, {"id": 1, "name": "fox"}],
        "annotations": [
            {"image_id": row["id"], "category_id": 1 if row["id"] == "extra_not_empty" else 30}
            for row in images
        ],
    }
    splits = {"splits": {"train": ["seen", "fresh_train"], "val": ["trans_val", "trans_test"], "eccv_val": ["trans_val", "trans_test"]}}
    return settings, documents, metadata, splits


def cohort(tmp_path):
    settings, documents, metadata, splits = sources(tmp_path)
    root = tmp_path / "backgrounds"
    write_json(root / "full_metadata.json", metadata)
    write_json(root / "CaltechCameraTrapsSplits_v0.json", splits)
    chosen = select_backgrounds(metadata, splits, documents, cameras=1, per_camera=1)
    for row in chosen:
        image = root / "images" / row["file_name"]
        image.parent.mkdir(exist_ok=True)
        Image.new("RGB", (20, 20), "gray").save(image)
        row["sha256"] = file_sha256(image)
    hashes = {key: file_sha256(settings.wildlife_root / name) for key, name in PARTITIONS.items()}
    receipt = {
        "selection": {"cameras": 1, "per_camera": 1, "seed": 42},
        "annotation_sha256": hashes,
        "metadata_sha256": {name: file_sha256(root / name) for name in ["full_metadata.json", "CaltechCameraTrapsSplits_v0.json"]},
        "images": chosen,
    }
    write_json(root / "cohort.json", receipt)
    settings.wildlife["background_manifest"] = str(root / "cohort.json")
    return settings, documents, hashes, root


def test_background_selection_excludes_reserved_images_sequences_and_cameras(tmp_path):
    _, documents, metadata, splits = sources(tmp_path)
    first = select_backgrounds(metadata, splits, documents, cameras=1, per_camera=2)
    assert {row["id"] for row in first} == {"extra_a", "extra_b"}
    assert first == select_backgrounds(metadata, splits, documents, cameras=1, per_camera=2)
    metadata["annotations"].append({"image_id": "extra_a", "category_id": 1})
    with pytest.raises(ValueError, match="Not enough"):
        select_backgrounds(metadata, splits, documents, cameras=1, per_camera=2)


def test_background_cohort_detects_changed_photo_and_annotation_receipt(tmp_path):
    _, documents, hashes, root = cohort(tmp_path)
    rows, provenance = load_background_cohort(root / "cohort.json", documents, hashes)
    assert provenance["images"] == 1
    with pytest.raises(ValueError, match="different ECCV"):
        load_background_cohort(root / "cohort.json", documents, {**hashes, "train": "changed"})
    (root / "images" / rows[0]["file_name"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed background"):
        load_background_cohort(root / "cohort.json", documents, hashes)


def test_prepare_adds_only_verified_empty_train_frames(tmp_path):
    settings, _, _, _ = cohort(tmp_path)
    manifest = prepare(settings)
    assert manifest["split_summary"]["train"]["images"] == 3
    assert manifest["split_summary"]["train"]["backgrounds"] == 2
    assert all(len(manifest["records"][split]) == 2 for split in PARTITIONS if split != "train")
    extra = [row for row in manifest["records"]["train"] if row.get("origin")]
    assert len(extra) == 1 and extra[0]["boxes"] == []
    assert all(value == 0 for value in manifest["sequence_overlaps"].values())


def test_label_review_prefers_actual_training_backgrounds(tmp_path):
    settings, _, _, _ = cohort(tmp_path)
    manifest = prepare(settings)
    path = prepare_label_review(manifest, settings.output_root / "wildlife")
    rows = read_csv(path)
    selected = {row["image_id"]: row for row in manifest["records"]["train"]}
    assert all(row["split"] == "train" for row in rows if row["image_id"] in selected and not selected[row["image_id"]]["boxes"])
    assert Counter(row["status"] for row in rows) == {"pending": len(rows)}


def test_full_occlusion_does_not_create_image_copies_or_accept_partial_coverage(tmp_path):
    output = tmp_path / "wildlife"
    source = output / "dataset/images/trans_test/source.jpg"
    source.parent.mkdir(parents=True)
    Image.new("RGB", (64, 64)).save(source)
    truth = [{**row, "camera_id": "new", "image": str(source)} for row in records()]
    manifest = {"records": {"trans_test": truth}, "names": ["fox", "deer"]}
    path = prepare_occlusion_review(manifest, output, full=True)
    reviews = read_csv(path)
    assert len(reviews) == 3 and not (output / "occlusion_gallery").exists()
    assert all(row["level"] == "" and row["reviewer"] == "" for row in reviews)
    assert "normalized_box" in (output / "occlusion_review.html").read_text()
    for row, level in zip(reviews, ["low", "medium", "high"], strict=True):
        row.update(level=level, reviewer="Synthetic fixture", visual_evidence="Fixture only")
    write_csv(path, reviews)
    assert robustness_metrics(truth, predictions(), manifest["names"], path, minimum=1, require_full=True)["occlusion_status"] == "reviewed"
    reviews[2].update(level="", reviewer="", visual_evidence="")
    write_csv(path, reviews)
    score = robustness_metrics(truth, predictions(), manifest["names"], path, minimum=0, require_full=True)
    assert score["occlusion_status"] == "manual_review_pending" and score["occlusion_coverage"]["unreviewed_objects"] == 1
