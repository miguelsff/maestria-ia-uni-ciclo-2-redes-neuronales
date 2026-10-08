"""Small local fixtures exercise official partition conversion without downloads."""

from types import SimpleNamespace

import pytest
from PIL import Image

from neural_exam.common.io import read_json, write_json
from neural_exam.wildlife.data import PARTITIONS, _link_image, normalize_box, prepare


def test_bbox_units_and_bounds():
    assert normalize_box([10, 20, 20, 40], 100, 200, "pixels") == pytest.approx([0.2, 0.2, 0.2, 0.2])
    assert normalize_box([0.1, 0.1, 0.2, 0.2], 100, 200, "normalized") == pytest.approx([0.2, 0.2, 0.2, 0.2])
    with pytest.raises(ValueError, match="Invalid"):
        normalize_box([0.9, 0.1, 0.2, 0.2], 100, 200, "normalized")


def fixture_settings(tmp_path):
    root = tmp_path / "CCT20"
    root.mkdir()
    categories = [{"id": 1, "name": "fox"}, {"id": 30, "name": "empty"}, {"id": 33, "name": "car"}]
    for split, filename in PARTITIONS.items():
        camera = "seen" if split in {"train", "val", "cis_test"} else split
        images, annotations = [], []
        entries = ["regular", "empty", "incomplete"]
        if split == "val":
            entries.append("shared")
        for suffix in entries:
            identifier = f"{split}_{suffix}"
            Image.new("RGB", (20, 20)).save(root / f"{identifier}.jpg")
            images.append(
                {
                    "id": identifier,
                    "file_name": f"{identifier}.jpg",
                    "width": 20,
                    "height": 20,
                    "location": camera,
                    "seq_id": "train_regular" if suffix == "shared" else identifier,
                }
            )
            annotation = {
                "id": f"annotation_{identifier}",
                "image_id": identifier,
                "category_id": 30 if suffix == "empty" else 1,
            }
            if suffix not in {"empty", "incomplete"}:
                annotation["bbox"] = [2, 2, 10, 10]
            annotations.append(annotation)
        write_json(root / filename, {"categories": categories, "images": images, "annotations": annotations})
    return SimpleNamespace(
        wildlife_root=root, output_root=tmp_path / "outputs", wildlife={"bbox_units": "pixels"}, training={"seed": 42}
    )


def test_empty_background_incomplete_boxes_and_shared_validation_sequences(tmp_path):
    settings = fixture_settings(tmp_path)
    manifest = prepare(settings)
    assert manifest["names"] == ["fox"]
    assert all(len(records) == 2 for records in manifest["records"].values())
    reasons = {row["reason"] for row in manifest["exclusions"]}
    assert reasons == {"incomplete_animal_boxes", "sequence_shared_with_train"}
    labels = list((settings.output_root / "wildlife/dataset/labels/train").glob("*.txt"))
    assert sorted(bool(path.read_text()) for path in labels) == [False, True]
    image_list = (settings.output_root / "wildlife/dataset/lists/train.txt").read_text().splitlines()
    assert all("dataset/images/train" in path.replace("\\", "/") for path in image_list)
    assert all(path.is_symlink() for path in (settings.output_root / "wildlife/dataset/images").rglob("*.jpg"))
    assert all(value == 0 for value in manifest["sequence_overlaps"].values())
    assert len(manifest["monitoring_frames"]) == 3


def test_separate_roots_and_original_resolution_boxes(tmp_path):
    settings = fixture_settings(tmp_path)
    images = tmp_path / "photographs"
    images.mkdir()
    for source in settings.wildlife_root.glob("*.jpg"):
        source.rename(images / source.name)
    settings.wildlife_images_root = images
    settings.wildlife_annotations_root = settings.wildlife_root
    document = read_json(settings.wildlife_root / PARTITIONS["train"])
    for image in document["images"]:
        image.update(width=40, height=40)
    document["annotations"][0]["bbox"] = [4, 4, 20, 20]
    write_json(settings.wildlife_root / PARTITIONS["train"], document)
    manifest = prepare(settings)
    regular = next(row for row in manifest["records"]["train"] if row["boxes"])
    assert regular["boxes"][0]["box"] == pytest.approx([0.35, 0.35, 0.5, 0.5])
    assert (regular["width"], regular["original_width"]) == (20, 40)
    assert len(regular["source_sha256"]) == 64


def test_smoke_never_materializes_test_partitions(tmp_path):
    settings = fixture_settings(tmp_path)
    manifest = prepare(settings, quick=True)
    assert set(manifest["records"]) == {"train", "val"}
    assert manifest["scientific_run"] is False
    assert manifest["monitoring_frames"] == []
    assert not (settings.output_root / "wildlife/dataset/images/trans_test").exists()


def test_source_change_invalidates_integrity_cache(tmp_path):
    settings = fixture_settings(tmp_path)
    first = prepare(settings)
    Image.new("RGB", (20, 20), "red").save(settings.wildlife_root / "train_regular.jpg")
    second = prepare(settings)
    assert first["identity"] != second["identity"]


def test_aspect_ratio_mismatch_fails_before_labels(tmp_path):
    settings = fixture_settings(tmp_path)
    Image.new("RGB", (40, 20)).save(settings.wildlife_root / "train_regular.jpg")
    with pytest.raises(ValueError, match="aspect ratios"):
        prepare(settings)
    assert not (settings.output_root / "wildlife/dataset").exists()


def test_links_never_fall_back_to_copying(tmp_path, monkeypatch):
    from pathlib import Path

    source, target = tmp_path / "original.jpg", tmp_path / "derived/image.jpg"
    source.write_bytes(b"original")

    def unsupported(*args, **kwargs):
        raise OSError("links unavailable")

    monkeypatch.setattr(Path, "symlink_to", unsupported)
    with pytest.raises(RuntimeError, match="copying is disabled"):
        _link_image(source, target)
    assert not target.exists()
    assert source.read_bytes() == b"original"


def test_regular_file_cannot_be_reused_as_a_derived_link(tmp_path):
    source, target = tmp_path / "source.jpg", tmp_path / "copy.jpg"
    source.write_bytes(b"source")
    target.write_bytes(b"copy")
    with pytest.raises(ValueError, match="regular file"):
        _link_image(source, target)
