"""Detection metrics, review integrity, bounded prediction and ecological counts."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from neural_exam.cli import parser
from neural_exam.common.io import read_csv, write_csv, write_json
from neural_exam.config import load_settings
from neural_exam.reporting import prepare_report_assets
from neural_exam.wildlife.metrics import coco_evaluate
from neural_exam.wildlife.monitoring import activity_report
from neural_exam.wildlife.pipeline import _select_cam_records, predict_records, robustness_metrics
from neural_exam.wildlife.review import label_review_status, prepare_label_review, prepare_occlusion_review


def records():
    return [
        {
            "image_id": "animal",
            "boxes": [
                {"class_id": 0, "annotation_id": "small", "box": [0.3, 0.3, 0.1, 0.05]},
                {"class_id": 0, "annotation_id": "medium", "box": [0.6, 0.3, 0.2, 0.25]},
                {"class_id": 1, "annotation_id": "large", "box": [0.5, 0.7, 0.4, 0.2]},
            ],
        },
        {"image_id": "empty", "boxes": []},
    ]


def predictions():
    return [
        {
            "image_id": "animal",
            "class_id": box["class_id"],
            "confidence": 0.8,
            "bbox": [box["box"][0] - box["box"][2] / 2, box["box"][1] - box["box"][3] / 2, *box["box"][2:]],
        }
        for box in records()[0]["boxes"]
    ]


def test_coco_background_false_positive_and_absent_category():
    truth, predicted = records(), predictions()
    scores = coco_evaluate(truth, predicted, ["fox", "deer", "bird"])
    assert scores["map50"] == pytest.approx(1)
    assert scores["per_species"][2]["support"] == 0
    assert scores["per_species"][2]["map50_95"] is None
    predicted.append({"image_id": "empty", "class_id": 0, "confidence": 0.99, "bbox": [0.1, 0.1, 0.2, 0.2]})
    assert coco_evaluate(truth, predicted, ["fox", "deer", "bird"])["map50"] < scores["map50"]


def test_coco_rejects_predictions_outside_fixed_categories():
    predicted = predictions()
    predicted[0]["class_id"] = 4
    with pytest.raises(ValueError, match="unknown image or category"):
        coco_evaluate(records(), predicted, ["fox", "deer"])


def test_zero_area_clipped_prediction_is_retained_as_false_positive():
    predicted = predictions()
    predicted.append({"image_id": "empty", "class_id": 0, "confidence": 0.99, "bbox": [0, 0, 0, 0.1]})
    assert coco_evaluate(records(), predicted, ["fox", "deer"])["map50"] < 1


def test_cam_selection_keeps_split_and_covers_rare_day_and_common_night():
    samples = []
    detected = []
    for index in range(6):
        row = {
            "image_id": str(index),
            "timestamp": "2020-01-01 12:00:00" if index % 2 == 0 else "2020-01-01 23:00:00",
            "boxes": [] if index < 2 else [{"class_id": index % 2, "box": [0.5, 0.5, 0.2, 0.2]}],
        }
        samples.append(row)
        if index < 2 or index >= 4:
            detected.append(
                {"image_id": str(index), "class_id": index % 2, "confidence": 0.9, "bbox": [0.4, 0.4, 0.2, 0.2]}
            )
    manifest = {
        "records": {"cis_test": samples},
        "names": ["rare", "common"],
        "train_class_boxes": {"rare": 5, "common": 1000},
    }
    selected = _select_cam_records(manifest, detected, "cis_test", 0.25)
    assert len(selected) == 6 and len({row[0]["image_id"] for row in selected}) == 6
    assert [item[1] for item in selected] == ["false_positive"] * 2 + ["missed_or_wrong_class"] * 2 + [
        "detected_iou50"
    ] * 2
    assert [item[2] for item in selected] == [0, 1, 0, 1, 0, 1]


def test_relative_area_strata_include_each_object_once():
    sizes = robustness_metrics(records(), predictions(), ["fox", "deer"])["size"]
    assert [group["eligible_objects"] for group in sizes.values()] == [1, 1, 1]
    assert all(group["map50_95"] == pytest.approx(1) for group in sizes.values())


def test_other_occlusion_levels_are_ignored_and_review_needs_evidence(tmp_path):
    path = tmp_path / "review.csv"
    rows = [
        {
            "annotation_id": box["annotation_id"],
            "image_id": "animal",
            "species": ["fox", "deer"][box["class_id"]],
            "level": level,
            "visual_evidence": "body contour visible",
            "reviewer": "Human fixture",
        }
        for box, level in zip(records()[0]["boxes"], ["low", "medium", "high"], strict=True)
    ]
    write_csv(path, rows)
    scores = robustness_metrics(records(), predictions(), ["fox", "deer"], path, minimum=1)
    assert scores["occlusion_status"] == "reviewed"
    assert all(
        group["map50_95"] == pytest.approx(1) and group["eligible_objects"] == 1
        for group in scores["occlusion"].values()
    )
    assert (
        robustness_metrics(records(), predictions(), ["fox", "deer"], path, minimum=30)["occlusion_status"]
        == "manual_review_pending"
    )
    rows[0]["visual_evidence"] = ""
    write_csv(path, rows)
    with pytest.raises(ValueError, match="visual evidence"):
        robustness_metrics(records(), predictions(), ["fox", "deer"], path)


def review_manifest(tmp_path):
    image = tmp_path / "source.jpg"
    Image.new("RGB", (64, 64), "green").save(image)
    rows = [{**row, "image": str(image), "camera_id": "new"} for row in records()]
    return {"identity": "fixture", "names": ["fox", "deer"], "records": {"train": rows, "val": [], "trans_test": rows}}


def test_label_sample_cannot_be_shortened_to_skip_review(tmp_path):
    manifest = review_manifest(tmp_path)
    path = prepare_label_review(manifest, tmp_path)
    assert label_review_status(path)["status"] == "human_review_pending"
    rows = read_csv(path)
    for row in rows:
        row.update(status="reviewed", reviewer="Human fixture")
    write_csv(path, rows)
    assert label_review_status(path)["status"] == "reviewed"
    write_csv(path, rows[:1])
    with pytest.raises(ValueError, match="original visual sample"):
        label_review_status(path)


def test_full_training_is_blocked_before_loading_model_when_review_is_pending(tmp_path, monkeypatch):
    from neural_exam.wildlife.pipeline import run

    directory = tmp_path / "ultralytics"
    directory.mkdir()
    monkeypatch.setenv("YOLO_CONFIG_DIR", str(directory))
    import ultralytics

    def forbidden(*args, **kwargs):
        pytest.fail("A model must not load before the human label review.")

    monkeypatch.setattr(ultralytics, "YOLO", forbidden)
    settings = SimpleNamespace(output_root=tmp_path, weights_root=tmp_path / "weights", training={"seed": 42})
    manifest = review_manifest(tmp_path)
    with pytest.raises(RuntimeError, match="Human label review is pending"):
        run(settings, manifest=manifest)


def test_resume_rejects_checkpoints_without_provenance(tmp_path, monkeypatch):
    from neural_exam.wildlife.pipeline import run

    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("PRETRAINED_WEIGHTS_DIR", str(tmp_path / "weights"))
    settings = load_settings(device="cpu")
    settings.weights_root.mkdir()
    (settings.weights_root / settings.wildlife["model"]).write_bytes(b"synthetic fixture")
    last = tmp_path / "wildlife/yolov8s/weights/last.pt"
    last.parent.mkdir(parents=True)
    last.write_bytes(b"unknown source")
    with pytest.raises(ValueError, match="no matching run metadata"):
        run(settings, resume=True, quick=True, manifest=review_manifest(tmp_path))


def test_occlusion_gallery_preserves_human_rows_and_rejects_wrong_image(tmp_path):
    manifest = review_manifest(tmp_path)
    path = prepare_occlusion_review(manifest, tmp_path)
    rows = read_csv(path)
    rows[0].update(level="high", visual_evidence="only head visible", reviewer="Human fixture")
    write_csv(path, rows)
    prepare_occlusion_review(manifest, tmp_path, extend=5)
    assert read_csv(path)[0]["visual_evidence"] == "only head visible"
    rows[0]["image_id"] = "another"
    write_csv(path, rows)
    with pytest.raises(ValueError, match="object/image/species"):
        prepare_occlusion_review(manifest, tmp_path)


def test_ecological_counts_include_empty_photos_and_do_not_count_individuals():
    frames = [
        {"image_id": str(index), "camera_id": "A", "timestamp": "2020-01-02 03:00:00" if index < 3 else "bad-date"}
        for index in range(4)
    ]
    detected = [{"image_id": "0", "class_id": 0, "confidence": confidence} for confidence in [0.9, 0.8, 0.1]]
    result = activity_report(frames, detected, ["fox"])
    day = next(row for row in result["camera_days"] if row["date"] != "unknown")
    assert day["photos"] == 3 and day["photos_with_fauna"] == 1
    assert day["empty_photo_rate"] == pytest.approx(2 / 3)
    assert result["species_days"][0]["detections"] == 2
    assert result["species_days"][0]["photos_with_species"] == 1
    assert result["camera_days"][1]["date"] == "unknown"


def test_prediction_batches_are_bounded_and_resume_cached_chunks(tmp_path):
    calls = []

    class Detector:
        def predict(self, images, **kwargs):
            calls.append(images)
            assert kwargs["stream"] is True and len(images) <= 2
            for _ in images:
                yield SimpleNamespace(
                    boxes=SimpleNamespace(
                        xyxyn=torch.tensor([[0.1, 0.2, 0.3, 0.4]]), conf=torch.tensor([0.9]), cls=torch.tensor([0])
                    )
                )

    settings = SimpleNamespace(
        wildlife={"prediction_batch": 2, "image_size": 64, "nms_iou": 0.7}, training={"device": "cpu"}
    )
    rows = [{"image_id": str(index), "image": f"{index}.jpg", "source_sha256": f"hash{index}"} for index in range(5)]
    first = predict_records(Detector(), rows, settings, tmp_path)
    second = predict_records(Detector(), rows, settings, tmp_path)
    assert [len(chunk) for chunk in calls] == [2, 2, 1]
    assert first == second and len(first) == 5


def test_narrow_commands_and_new_roots_keep_legacy_fallback(monkeypatch):
    for name in ["CCT_IMAGES_DIR", "CCT_ANNOTATIONS_DIR", "CCT_DATA_DIR", "KAGGLE_ROOT"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CCT_DATA_DIR", "/legacy")
    assert load_settings().wildlife_images_root == Path("/legacy")
    monkeypatch.setenv("CCT_IMAGES_DIR", "/photos")
    monkeypatch.setenv("CCT_ANNOTATIONS_DIR", "/labels")
    settings = load_settings()
    assert settings.wildlife_images_root == Path("/photos")
    assert settings.wildlife_annotations_root == Path("/labels")
    assert parser().parse_args(["download", "--dataset", "wildlife-weights"]).dataset == "wildlife-weights"
    assert parser().parse_args(["review-occlusion", "--extend", "60"]).extend == 60


def test_wildlife_report_exports_both_maps_and_rejects_mixed_checkpoints(tmp_path):
    source = tmp_path / "wildlife"
    manifest = review_manifest(tmp_path)
    manifest["split_summary"] = {
        split: {
            "images": len(rows),
            "backgrounds": sum(not row["boxes"] for row in rows),
            "cameras": ["A"],
            "sequences": len(rows),
        }
        for split, rows in manifest["records"].items()
    }
    review = prepare_label_review(manifest, source)
    human_rows = read_csv(review)
    for row in human_rows:
        row.update(status="reviewed", reviewer="Synthetic human fixture")
    write_csv(review, human_rows)
    write_json(source / "manifest.json", manifest)
    write_json(source / "yolov8s/run.json", {"scientific_run": True, "scope": "synthetic test fixture"})
    write_json(source / "yolov8s/evaluation_complete.json", {"checkpoint_sha256": "fixture"})
    write_csv(
        source / "experiments.csv",
        [
            {
                "experiment": "fixture",
                "cis_map50": 1,
                "trans_map50": 1,
                "cis_map50_95": 1,
                "trans_map50_95": 1,
                "geographic_gap": 0,
            }
        ],
    )
    for split in ["cis_test", "trans_test"]:
        score = coco_evaluate(records(), predictions(), ["fox", "deer", "absent"])
        score["protocol"] = {"checkpoint_sha256": "fixture", "categories_evaluated": 2}
        write_json(source / split / "metrics.json", score)
    review = source / "occlusion_review.csv"
    write_csv(review, [], ["annotation_id", "image_id", "species", "level", "visual_evidence", "reviewer"])
    robustness = robustness_metrics(records(), predictions(), ["fox", "deer"], review)
    robustness["size_by_split"] = {split: robustness["size"] for split in ["cis_test", "trans_test"]}
    write_json(source / "robustness.json", robustness)
    activity = activity_report(
        [
            {"image_id": "animal", "camera_id": "A", "timestamp": "2020-01-01"},
            {"image_id": "empty", "camera_id": "A", "timestamp": "2020-01-01"},
        ],
        predictions(),
        ["fox", "deer"],
    )
    write_json(source / "activity.json", {**activity, "camera_id": "A"})
    write_csv(
        source / "activity_sequence.csv",
        [{"frame_number": 0, "timestamp": "2020-01-01", "detections": 3, "species": "fox, deer"}],
    )
    write_json(source / "cam_cases.json", [{"scope": "synthetic fixture"}] * 12)
    for name in ["class_balance", "training_curves", "cam_cis_test_grid", "cam_trans_test_grid"]:
        Image.new("RGB", (8, 8)).save(source / f"{name}.png")
    settings = SimpleNamespace(output_root=tmp_path)
    write_json(
        source / "generalization.json",
        {
            "common_categories": ["fox"],
            "common_category_scores": {split: {"map50": 1.0, "map50_95": 1.0} for split in ("cis_test", "trans_test")},
        },
    )
    interval_fixture = {
        "clusters": 2,
        "intervals": {"map50_95": {"lower": 0.75, "upper": 1.0, "defined_replicates": 4}},
    }
    write_json(source / "uncertainty.json", {"global": {"cis_test": interval_fixture}, "size": {}, "occlusion": {}})
    status = prepare_report_assets(settings, "wildlife")["wildlife"]
    tables = (tmp_path / "reports/wildlife/tables.tex").read_text()
    assert status["missing_figures"] == [] and status["label_review_status"] == "reviewed"
    assert status["occlusion_status"] == "manual_review_pending" and status["cam_cases"] == 12
    assert status["occlusion_coverage"]["unreviewed_objects"] == 3
    assert "Cobertura visual: 0/3" in tables
    assert "mAP50" in tables and "mAP50:95" in tables and "ND" in tables
    assert "IC95 inferior" in tables and "0.7500" in tables
    assert "mismo conjunto de clases" in tables and "entre semillas" in tables
    assert "cis\\_test" in tables and "1 \\%" in tables
    score["protocol"]["checkpoint_sha256"] = "another checkpoint"
    write_json(source / "trans_test/metrics.json", score)
    with pytest.raises(ValueError, match="same selected checkpoint"):
        prepare_report_assets(settings, "wildlife")
