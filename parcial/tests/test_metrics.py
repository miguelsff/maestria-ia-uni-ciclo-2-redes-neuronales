"""Regression cases for missing classes and masked pixels."""

import numpy as np
import pytest

from neural_exam.common.metrics import binary_counts, metrics_from_counts, multilabel_metrics
from neural_exam.wildlife.metrics import coco_evaluate
from neural_exam.wildlife.monitoring import activity_report


def test_macro_auc_does_not_silently_drop_single_class_labels():
    target = np.array([[0, 0], [1, 0]])
    result = multilabel_metrics(target, np.array([[0.1, 0.1], [0.9, 0.2]]), ["tear", "fracture"], 0.5)
    assert result["macro_auc"] is None
    assert result["partial_macro_auc"] == 1
    assert result["auc_labels_defined"] == 1


def test_nodata_does_not_enter_confusion_counts():
    counts = binary_counts(np.array([-1, 0, 1, 1]), np.array([1, 1, 0, 1]))
    assert counts == {"tp": 1, "fp": 1, "fn": 1, "tn": 0}
    assert metrics_from_counts(counts)["iou"] == pytest.approx(1 / 3)
    assert metrics_from_counts(dict.fromkeys(counts, 0))["iou"] is None


def test_probability_validation():
    with pytest.raises(ValueError, match="Probabilities"):
        multilabel_metrics(np.zeros((1, 1)), np.array([[np.nan]]), ["a"], 0.5)


def test_coco_empty_predictions_and_perfect_prediction():
    records = [{"image_id": "frame", "boxes": [{"class_id": 0, "annotation_id": "box", "box": [0.5, 0.5, 0.2, 0.2]}]}]
    empty = coco_evaluate(records, [], ["fox", "deer"])
    assert empty["map50_95"] == 0
    assert empty["per_species"][1]["map50_95"] is None
    prediction = [{"image_id": "frame", "class_id": 0, "confidence": 0.9, "bbox": [0.4, 0.4, 0.2, 0.2]}]
    assert coco_evaluate(records, prediction, ["fox", "deer"])["map50_95"] == pytest.approx(1)


def test_activity_counts_photos_once_but_all_detections():
    frames = [
        {"image_id": "a", "camera_id": "new", "timestamp": "2026-10-01T12:00:00"},
        {"image_id": "b", "camera_id": "new", "timestamp": "2026-10-01T13:00:00"},
    ]
    detections = [{"image_id": "a", "class_id": 0, "confidence": 0.9}] * 2
    report = activity_report(frames, detections, ["fox"])
    assert report["camera_days"][0]["empty_photo_rate"] == 0.5
    assert report["species_days"][0]["detections"] == 2
    assert report["species_days"][0]["photos_with_species"] == 1
    frames[0]["timestamp"] = None
    assert any(row["date"] == "unknown" for row in activity_report(frames, detections, ["fox"])["camera_days"])


def test_duplicate_frames_are_rejected():
    frame = {"image_id": "a", "camera_id": "new", "timestamp": ""}
    with pytest.raises(ValueError, match="Duplicate"):
        activity_report([frame, frame], [], ["fox"])
