"""Check isolated recovery reproduces the unchanged bootstrap on real COCO fixtures."""

import os
import runpy
import subprocess
import sys
from dataclasses import replace
from importlib import metadata
from pathlib import Path

import pytest

from neural_exam.common.io import file_sha256, fingerprint, read_json, write_csv, write_json
from neural_exam.config import load_settings
from neural_exam.wildlife.uncertainty import assess_uncertainty

PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts/recover_wildlife_evaluation.py"
RECOVERY = runpy.run_path(str(SCRIPT))


def fixture_inputs():
    records, predictions, reviews = [], [], []
    for sequence in ("a", "b"):
        for frame in range(3):
            identifier = f"{sequence}{frame}"
            boxes = []
            if frame < 2:
                side = (0.05 if sequence == "a" else 0.5) if frame == 0 else 0.2
                boxes = [{"class_id": frame, "box": [0.5, 0.5, side, side], "annotation_id": identifier}]
                predictions.append(
                    {
                        "image_id": identifier,
                        "class_id": frame,
                        "bbox": [0.5 - side / 2, 0.5 - side / 2, side, side],
                        "confidence": 0.8,
                    }
                )
                reviews.append({"annotation_id": identifier, "level": "low" if frame == 0 else "medium"})
            else:
                predictions.append(
                    {"image_id": identifier, "class_id": 0, "bbox": [0.1, 0.1, 0.05, 0.05], "confidence": 0.9}
                )
            records.append({"image_id": identifier, "sequence_id": sequence, "boxes": boxes})
    manifest = {"records": {split: records for split in ("cis_test", "trans_test")}, "names": ["bird", "fox", "badger"]}
    manifest["identity"] = fingerprint(manifest)
    return manifest, {split: predictions for split in manifest["records"]}, reviews


@pytest.mark.parametrize("reviewed", [False, True])
def test_isolated_group_assembly_exactly_matches_frozen_assessment(reviewed):
    manifest, predictions, reviews = fixture_inputs()
    robustness = {"occlusion_status": "reviewed" if reviewed else "manual_review_pending"}
    original = assess_uncertainty(manifest, predictions, robustness, 4, 42, reviews)
    groups = {
        key: RECOVERY["evaluate_group"](key, manifest, predictions[key.split("--")[0]], robustness, reviews, 4, 42)
        for key in RECOVERY["bootstrap_jobs"](robustness)
    }
    assert RECOVERY["assemble_results"](robustness, groups) == original
    with pytest.raises(ValueError, match="coverage is incomplete"):
        RECOVERY["assemble_results"](robustness, {})
    if not reviewed:
        with pytest.raises(ValueError, match="declared protocol"):
            RECOVERY["evaluate_group"](
                "trans_test--occlusion_low", manifest, predictions["trans_test"], robustness, reviews, 4, 42
            )


def test_fresh_worker_validates_inputs_and_saves_bound_results(tmp_path):
    manifest, predictions, reviews = fixture_inputs()
    config = tmp_path / "fixture.toml"
    config.write_text(
        (PROJECT / "configs/wildlife_v6.toml")
        .read_text()
        .replace("uncertainty_replicates = 200", "uncertainty_replicates = 4")
    )
    root = tmp_path / "outputs"
    settings = replace(load_settings(config), output_root=root)
    package = PROJECT / "src/neural_exam"
    paths = [
        *(package / "wildlife").glob("*.py"),
        *(package / "common").glob("*.py"),
        package / "config.py",
        package / "cli.py",
    ]
    sources = {path.relative_to(package).as_posix(): file_sha256(path) for path in paths}
    checkpoint = root / "wildlife/yolov8s/weights/best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"synthetic test checkpoint; never used for inference")
    digest = file_sha256(checkpoint)
    write_json(root / "wildlife/yolov8s/training_complete.json", {"sha256": digest})
    write_json(root / "wildlife/yolov8s/run.json", {"scientific_run": True, "source_sha256": sources})
    write_json(
        root / "receipts/source-freeze.json",
        {
            "sha256": {"configs/wildlife_v6.toml": file_sha256(PROJECT / "configs/wildlife_v6.toml")},
            "versions": {"numpy": metadata.version("numpy")},
        },
    )
    write_json(root / "wildlife/manifest.json", manifest)
    robustness = {"occlusion_status": "manual_review_pending"}
    write_json(root / "wildlife/robustness.json", robustness)
    write_csv(root / "wildlife/occlusion_review.csv", reviews)
    for split in predictions:
        write_json(root / "wildlife" / split / "predictions.json", predictions[split])
        write_json(
            root / "wildlife" / split / "metrics.json",
            {"protocol": {"checkpoint_sha256": digest, "conf": 0.001, "nms_iou": 0.7, "max_detections": 300}},
        )
    _, identity, _ = RECOVERY["verify_inputs"](settings)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--worker", "cis_test--medium_1_to_5pct"],
        env={**os.environ, "OUTPUT_DIR": str(root), "NEURAL_EXAM_CONFIG": str(config)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    saved = read_json(root / "wildlife/bootstrap_cache" / fingerprint(identity) / "cis_test--medium_1_to_5pct.json")
    assert saved["identity"] == identity
    assert saved["metrics"] == RECOVERY["evaluate_group"](
        saved["group"], manifest, predictions["cis_test"], robustness, reviews, 4, 42
    )
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint changed"):
        RECOVERY["verify_inputs"](settings)
