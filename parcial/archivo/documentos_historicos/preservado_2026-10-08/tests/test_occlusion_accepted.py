"""Verify exclusions and note waiver on synthetic data, plus frozen COCO background behavior."""

import csv
import sys
from pathlib import Path

import pytest
from test_occlusion_sample import SAMPLE, fixture_population

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_occlusion_accepted as evaluator  # noqa: E402
import occlusion_accepted_review as accepted  # noqa: E402


def accepted_fixture(tmp_path):
    """All human fields are explicitly synthetic; 18 remain ungraded."""
    _, output, _, _ = fixture_population(tmp_path)
    protocol = SAMPLE.read_protocol(output)
    rows = SAMPLE.read_reviews(output)
    for index, row in enumerate(rows[:-18]):
        row.update(level=SAMPLE.LEVELS[index % 4], reviewer="Synthetic reviewer", visual_evidence="")
    SAMPLE.write_reviews(output, rows)
    exported = output / "occlusion_review_accepted_582.csv"
    with exported.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SAMPLE.FIELDS)
        writer.writeheader()
        writer.writerows(rows[:-18])
    receipt = {
        "protocol_sha256": SAMPLE.digest(output / "protocol.json"),
        "reviews_sha256": SAMPLE.digest(output / "occlusion_review_sample_600.csv"),
        "accepted_csv_sha256": SAMPLE.digest(exported),
        "notes_required": False,
        "human_notes_fabricated": False,
        "excluded_cases": 18,
        "excluded_annotation_ids": [row["annotation_id"] for row in rows[-18:]],
        "accepted_cases": 582,
        "resolved_case_121": {
            "annotation_id": rows[120]["annotation_id"],
            **{field: rows[120][field] for field in SAMPLE.HUMAN_FIELDS},
        },
        "user_instruction": "Synthetic fixture instruction; no genuine human approval.",
    }
    SAMPLE.write_json(output / "review_acceptance.json", receipt)
    return output, protocol, rows, receipt


def test_notes_waiver_does_not_fill_notes_or_change_original_protocol(tmp_path):
    output, protocol, rows, _ = accepted_fixture(tmp_path)
    before = SAMPLE.digest(output / "protocol.json")
    _, selected, _, status = accepted.accepted_inputs(output)
    assert len(selected) == 582 and status["excluded"] == 18
    assert status["accepted_cases_without_notes"] == 582
    assert all(row["visual_evidence"] == "" for row in selected)
    assert SAMPLE.review_status(rows, protocol)["complete"] == 0  # The prior strict protocol remains intact.
    descriptive = accepted.describe_accepted_reviews(output)
    assert sum(level["observed_case_proportion"] for level in descriptive["levels"].values()) == pytest.approx(1)
    assert descriptive["original_design_weight_sum"] == pytest.approx(3300)
    assert descriptive["accepted_original_design_weight_sum"] + descriptive[
        "excluded_original_design_weight_sum"
    ] == pytest.approx(3300)
    assert descriptive["confidence_intervals_for_population_proportions"] is None
    assert descriptive["original_95pct_plus_minus_5pp_planning_guarantee_claimed"] is False
    assert SAMPLE.digest(output / "protocol.json") == before


def test_case_resolution_exclusions_and_hashes_are_binding(tmp_path):
    output, _, rows, receipt = accepted_fixture(tmp_path)
    receipt["excluded_annotation_ids"][0] = rows[0]["annotation_id"]
    SAMPLE.write_json(output / "review_acceptance.json", receipt)
    with pytest.raises(ValueError, match="ungraded"):
        accepted.accepted_inputs(output)
    receipt["excluded_annotation_ids"] = [row["annotation_id"] for row in rows[-18:]]
    receipt["resolved_case_121"]["level"] = "high"
    SAMPLE.write_json(output / "review_acceptance.json", receipt)
    with pytest.raises(ValueError, match="Case 121"):
        accepted.accepted_inputs(output)
    rows[0]["level"] = "high"
    SAMPLE.write_reviews(output, rows)
    with pytest.raises(ValueError, match="human review changed"):
        accepted.accepted_inputs(output)


def test_excluded_detection_is_ignored_but_background_false_positive_is_counted():
    records = [
        {
            "image_id": "a",
            "sequence_id": "s1",
            "boxes": [{"annotation_id": "kept", "class_id": 0, "box": [0.5, 0.5, 0.2, 0.2]}],
        },
        {
            "image_id": "b",
            "sequence_id": "s2",
            "boxes": [{"annotation_id": "excluded", "class_id": 0, "box": [0.5, 0.5, 0.2, 0.2]}],
        },
        {"image_id": "c", "sequence_id": "s3", "boxes": []},
    ]
    predictions = [
        {"image_id": "a", "class_id": 0, "bbox": [0.4, 0.4, 0.2, 0.2], "confidence": 0.8},
        {"image_id": "b", "class_id": 0, "bbox": [0.4, 0.4, 0.2, 0.2], "confidence": 0.95},
        {"image_id": "c", "class_id": 0, "bbox": [0.4, 0.4, 0.2, 0.2], "confidence": 0.99},
    ]
    rows = [{"annotation_id": "kept", "level": "low"}]
    result = evaluator.compute(records, predictions, ["fixture species"], rows, "low", 200, 42)
    assert result["metrics"]["eligible_objects"] == 1
    assert result["metrics"]["map50"] == pytest.approx(0.5)
    assert result["uncertainty"]["replicates"] == 200
    assert result["uncertainty"]["clusters"] == 3
    assert 0 < result["uncertainty"]["intervals"]["map50"]["defined_replicates"] < 200
    without_false_positive = evaluator.compute(records, predictions[:2], ["fixture species"], rows, "low", 2, 42)
    assert without_false_positive["metrics"]["map50"] == pytest.approx(1)
