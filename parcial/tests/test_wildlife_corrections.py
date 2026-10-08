"""Check correction scope, authentic review migration and patience recovery."""

import copy
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from neural_exam.common.io import fingerprint, read_csv, read_json, write_csv, write_json
from neural_exam.wildlife.corrections import apply_corrections, migrate_reviews, verify_parent
from neural_exam.wildlife.resume import attach_resume_callbacks
from neural_exam.wildlife.review import architecture_review_status, label_review_status
from neural_exam.wildlife.uncertainty import bootstrap_metrics


def fixture_record(identifier):
    return {
        "image_id": identifier,
        "source_sha256": "photo-hash",
        "image": "/old/image.jpg",
        "sequence_id": identifier,
        "boxes": [{"class_id": 0, "box": [0.5, 0.5, 0.2, 0.2], "annotation_id": identifier}],
    }


def correction_fixture(tmp_path):
    parent = {
        "records": {
            "train": [fixture_record("changed"), fixture_record("same")],
            "trans_test": [fixture_record("test")],
        },
        "names": ["bird"],
        "monitoring_frames": [],
    }
    parent["identity"] = fingerprint(parent)
    current = copy.deepcopy(parent)
    patch = {
        "image_id": "changed",
        "split": "train",
        "source_sha256": "photo-hash",
        "original_boxes": copy.deepcopy(parent["records"]["train"][0]["boxes"]),
        "reason": "visible head outside original box",
        "proposal_kind": "agent_proposal_requires_human_review",
        "replacement_boxes": [{"category": "bird", "box": [0.5, 0.5, 0.3, 0.3], "annotation_id": "changed"}],
    }
    path = tmp_path / "patch.json"
    write_json(path, {"schema_version": 1, "parent_manifest_identity": parent["identity"], "patches": [patch]})
    current["annotation_corrections"] = apply_corrections(current["records"], current["names"], path)
    current["identity"] = fingerprint({key: value for key, value in current.items() if key != "identity"})
    return parent, current, path


def test_corrections_only_change_registered_train_boxes(tmp_path):
    parent, current, path = correction_fixture(tmp_path)
    assert verify_parent(current, parent) == ["changed"]
    assert current["records"]["trans_test"] == parent["records"]["trans_test"]
    current["records"]["trans_test"][0]["boxes"][0]["box"][0] = 0.6
    with pytest.raises(ValueError, match="Only training"):
        verify_parent(current, parent)
    document = read_json(path)
    document["patches"][0]["split"] = "trans_test"
    write_json(path, document)
    with pytest.raises(ValueError, match="only reference eligible training"):
        apply_corrections(parent["records"], parent["names"], path)


def test_migration_invalidates_changed_targets_without_inventing_authorship(tmp_path):
    parent, current, _ = correction_fixture(tmp_path)
    old, new = tmp_path / "old", tmp_path / "new"
    write_json(old / "manifest.json", parent)
    write_json(
        old / "label_review_sample.json", {"manifest_identity": parent["identity"], "image_ids": ["changed", "same"]}
    )
    original = [
        {
            "image_id": identifier,
            "status": "reviewed",
            "reviewer": "Human",
            "visual_evidence": "visible bird",
            "split": "train",
            "gallery": "old.png",
        }
        for identifier in ("changed", "same")
    ]
    write_csv(old / "label_review.csv", original)
    write_csv(
        new / "label_review.csv",
        [{**row, "status": "pending", "reviewer": "", "visual_evidence": "", "gallery": "new.png"} for row in original],
    )
    receipt = migrate_reviews(current, new, old)
    changed, same = read_csv(new / "label_review.csv")
    assert changed["status"] == "pending" and changed["reviewer"] == changed["visual_evidence"] == ""
    assert same["status"] == "reviewed" and same["reviewer"] == "Human"
    assert read_csv(new / "parent_label_review.csv") == original
    assert receipt["unchanged_rows_carried"] == 1
    # Repeated preparation must not overwrite a new genuine reviewer or judgment.
    changed.update(status="reviewed", reviewer="New human", visual_evidence="corrected box inspected")
    write_csv(new / "label_review.csv", [changed, same])
    migrate_reviews(current, new, old)
    assert read_csv(new / "label_review.csv")[0]["reviewer"] == "New human"


def test_retained_corrections_preserve_reviews_when_only_image_materialization_changes(tmp_path):
    _, parent, path = correction_fixture(tmp_path)
    parent["identity"] = fingerprint({key: value for key, value in parent.items() if key != "identity"})
    current = copy.deepcopy(parent)
    document = read_json(path)
    document.update(parent_manifest_identity=parent["identity"], parent_changed_image_ids=[])
    write_json(path, document)
    # Every build starts from the official targets, then reapplies the existing correction.
    current["records"]["train"][0]["boxes"] = copy.deepcopy(document["patches"][0]["original_boxes"])
    current["annotation_corrections"] = apply_corrections(current["records"], current["names"], path)
    for rows in current["records"].values():
        for row in rows:
            row["image"] = "/new/lossless-working-copy.jpg"
    current["identity"] = fingerprint({key: value for key, value in current.items() if key != "identity"})
    assert verify_parent(current, parent) == []
    old, new = tmp_path / "old", tmp_path / "new"
    write_json(old / "manifest.json", parent)
    write_json(
        old / "label_review_sample.json", {"manifest_identity": parent["identity"], "image_ids": ["changed", "same"]}
    )
    reviews = [
        {"image_id": identifier, "status": "reviewed", "reviewer": "Human fixture", "visual_evidence": ""}
        for identifier in ("changed", "same")
    ]
    write_csv(old / "label_review.csv", reviews)
    write_csv(new / "label_review.csv", [{**row, "status": "pending", "reviewer": ""} for row in reviews])
    receipt = migrate_reviews(current, new, old)
    assert receipt["unchanged_rows_carried"] == 2 and receipt["changed_images_requiring_review"] == []
    assert read_csv(new / "label_review.csv") == reviews


def test_current_sample_requires_real_written_evidence(tmp_path):
    path = tmp_path / "label_review.csv"
    write_json(tmp_path / "label_review_sample.json", {"image_ids": ["a"], "require_evidence": True})
    write_csv(path, [{"image_id": "a", "status": "reviewed", "reviewer": "Human", "visual_evidence": ""}])
    assert label_review_status(path)["status"] == "human_review_pending"
    assert label_review_status(path)["missing_visual_evidence"] == 1


def test_resume_restores_original_patience_and_rejects_changed_checkpoint(tmp_path):
    class Model:
        def __init__(self):
            self.callbacks = {}

        def add_callback(self, event, callback):
            self.callbacks[event] = callback

    last = tmp_path / "weights/last.pt"
    last.parent.mkdir()
    last.write_bytes(b"checkpoint")
    trainer = SimpleNamespace(
        epoch=3,
        start_epoch=4,
        args=SimpleNamespace(patience=10),
        last=last,
        stopper=SimpleNamespace(best_fitness=0.7, best_epoch=2, patience=10, possible_stop=False),
    )
    model = Model()
    attach_resume_callbacks(model, tmp_path, resume=False)
    model.callbacks["on_model_save"](trainer)
    recovered = Model()
    attach_resume_callbacks(recovered, tmp_path, resume=True)
    trainer.stopper.best_epoch = 0
    recovered.callbacks["on_train_start"](trainer)
    assert trainer.stopper.best_epoch == 2 and trainer.stopper.best_fitness == 0.7
    last.write_bytes(b"changed")
    with pytest.raises(ValueError, match="does not match"):
        attach_resume_callbacks(Model(), tmp_path, resume=True)


def test_architecture_decision_requires_a_real_reviewer_and_evidence(tmp_path):
    path = tmp_path / "architecture-review.json"
    assert architecture_review_status(path)["status"] == "human_review_pending"
    write_json(path, {"reviewed": True, "decision": "pretrained_detector_permitted", "reviewer": "", "evidence": ""})
    assert architecture_review_status(path)["status"] == "human_review_pending"
    write_json(
        path,
        {
            "reviewed": True,
            "decision": "pretrained_detector_permitted",
            "reviewer": "Human fixture",
            "evidence": "Synthetic test only",
        },
    )
    assert architecture_review_status(path)["status"] == "reviewed"


def test_bootstrap_keeps_sequence_frames_and_perfect_predictions():
    records = [fixture_record("a"), fixture_record("b")]
    predictions = [
        {"image_id": row["image_id"], "class_id": 0, "bbox": [0.4, 0.4, 0.2, 0.2], "confidence": 0.9} for row in records
    ]
    scores = bootstrap_metrics(records, predictions, ["bird"], replicates=4, seed=42)
    assert scores["clusters"] == 2 and scores["intervals"]["map50"]["defined_replicates"] == 4
    assert scores["intervals"]["map50"]["lower"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="outside this split"):
        bootstrap_metrics(records, [{**predictions[0], "image_id": "outside"}], ["bird"], replicates=4, seed=42)


@pytest.mark.parametrize("parent_version", ["v2", "v3", "v4", "v5"])
def test_bootstrap_stages_a_fresh_output_and_refuses_changed_parent(tmp_path, parent_version):
    parent, output = tmp_path / "v2", tmp_path / "v3"
    for name in (
        "wildlife/manifest.json",
        "wildlife/label_review.csv",
        "wildlife/label_review_sample.json",
        "wildlife/occlusion_review.csv",
        "cache/wildlife-integrity.json",
        "cache/weights/yolov8s.pt",
        "cache/weights/yolo26n.pt",
        "report-assets/neurips_2026.sty",
    ):
        path = parent / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
    log = output / "receipts/bootstrap.log"
    log.parent.mkdir(parents=True)
    log.write_text("", encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts/bootstrap_wildlife_v3.py"
    command = [
        sys.executable,
        str(script),
        "--parent",
        str(parent),
        "--output",
        str(output),
        "--parent-version",
        parent_version,
    ]
    first = subprocess.run(command, capture_output=True, text=True, check=False)
    assert first.returncode == 0, first.stderr
    receipt = (output / "receipts/bootstrap.json").read_bytes()
    assert subprocess.run(command, capture_output=True, check=False).returncode == 0
    assert (output / "receipts/bootstrap.json").read_bytes() == receipt
    (parent / "wildlife/label_review.csv").write_text("changed", encoding="utf-8")
    changed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert changed.returncode != 0 and "differs" in changed.stderr
    assert (output / f"inputs/parent-{parent_version}/label_review.csv").read_text() == "fixture"


@pytest.mark.parametrize("require_notes", [False, True])
def test_scientific_gate_respects_the_configured_note_policy(tmp_path, require_notes):
    project = Path(__file__).resolve().parents[1]
    config = tmp_path / "config.toml"
    config.write_text(
        (project / "configs/wildlife_v4.toml")
        .read_text()
        .replace("require_label_evidence = false", f"require_label_evidence = {str(require_notes).lower()}")
    )
    review = tmp_path / "wildlife"
    write_json(review / "label_review_sample.json", {"image_ids": ["fixture"], "require_evidence": require_notes})
    write_csv(
        review / "label_review.csv",
        [{"image_id": "fixture", "status": "reviewed", "reviewer": "Synthetic human fixture", "visual_evidence": ""}],
    )
    result = subprocess.run(
        [sys.executable, str(project / "scripts/check_wildlife_gate.py")],
        env={**os.environ, "OUTPUT_DIR": str(tmp_path), "NEURAL_EXAM_CONFIG": str(config)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1  # The separate architecture decision is deliberately still absent.
    receipt = read_json(tmp_path / "receipts/scientific-gate.json")
    assert receipt["labels"]["status"] == ("human_review_pending" if require_notes else "reviewed")
    assert receipt["labels"]["missing_visual_evidence"] == 1
    assert receipt["model_loaded"] is False


def test_bootstrap_resamples_whole_sequences_including_empty_frames(monkeypatch):
    import neural_exam.wildlife.uncertainty as module

    records = [fixture_record(identifier) for identifier in ("a1", "a2", "b1", "b2")]
    for record in records:
        record["sequence_id"] = record["image_id"][0]
    records[1]["boxes"] = []
    observed = []

    def evaluate(sampled, detected, names, **kwargs):
        identifiers = [row["image_id"].split("-")[-1] for row in sampled]
        observed.append(identifiers)
        assert len(identifiers) == 4
        assert identifiers.count("a1") == identifiers.count("a2")
        assert identifiers.count("b1") == identifiers.count("b2")
        assert sum(not row["boxes"] for row in sampled) == identifiers.count("a2")
        return {"map50": None, "map50_95": 0.5}

    monkeypatch.setattr(module, "coco_evaluate", evaluate)
    result = module.bootstrap_metrics(records, [], ["bird"], replicates=4, seed=42)
    assert len(observed) == 4 and result["clusters"] == 2
    assert result["intervals"]["map50"]["lower"] is None
    assert result["intervals"]["map50"]["defined_replicates"] == 0
