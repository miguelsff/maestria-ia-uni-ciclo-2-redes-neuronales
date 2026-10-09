"""Check sampling, real persistence and evaluation with synthetic temporary data."""

import importlib.util
import json
import sys
import threading
from collections import Counter
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts/wildlife_occlusion_sample.py"
SPEC = importlib.util.spec_from_file_location("wildlife_occlusion_sample", SCRIPT)
SAMPLE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SAMPLE
SPEC.loader.exec_module(SAMPLE)


def fixture_population(tmp_path):
    """Ten synthetic strata, deliberately unequal populations and equal quotas."""
    parent, images, output = tmp_path / "parent", tmp_path / "images", tmp_path / "sample"
    (parent / "wildlife").mkdir(parents=True)
    images.mkdir()
    records, quotas = [], []
    for index in range(10):
        filename = f"fixture-{index}.jpg"
        source = images / filename
        source.write_bytes(f"Synthetic image bytes {index}; not scientific data".encode())
        population = 60 * (index + 1)
        records.append(
            {
                "image_id": f"frame-{index}",
                "source": f"/datasets/{filename}",
                "camera_id": str(index % 3),
                "sequence_id": f"sequence-{index}",
                "source_sha256": SAMPLE.digest(source),
                "boxes": [
                    {"annotation_id": f"{index}-{box}", "class_id": index, "box": [0.5, 0.5, 0.2, 0.2]}
                    for box in range(population)
                ],
            }
        )
        quotas.append(
            {
                "camera_id": str(index % 3),
                "species": f"species-{index}",
                "size": "medium",
                "population": population,
                "sample": 60,
            }
        )
    manifest = {
        "identity": "synthetic-manifest",
        "names": [f"species-{i}" for i in range(10)],
        "records": {"trans_test": records},
    }
    SAMPLE.write_json(parent / "wildlife/manifest.json", manifest)
    for name in ("label_review.csv", "occlusion_review.csv"):
        (parent / "wildlife" / name).write_text("Synthetic immutable parent review\n")
    analysis = {
        "population": {"manifest_identity": manifest["identity"], "objects": 3300},
        "source_sha256": {
            name: SAMPLE.digest(parent / "wildlife" / name)
            for name in ("manifest.json", "label_review.csv", "occlusion_review.csv")
        },
        "proposed_allocation": quotas,
        "assumptions": {"planned_random_seed": 42},
        "limits": [],
    }
    path = tmp_path / "analysis.json"
    SAMPLE.write_json(path, analysis)
    with (
        patch.object(SAMPLE, "verify_parent", return_value={}),
        patch.object(
            SAMPLE,
            "read_json",
            side_effect=lambda path: {"sha256": {}}
            if path.name == "source-freeze.json"
            else json.loads(path.read_text()),
        ),
    ):
        SAMPLE.prepare(parent, output, images, path)
    return parent, output, manifest, analysis


def test_random_selection_matches_quotas_is_stable_and_preserves_parent(tmp_path):
    parent, output, manifest, analysis = fixture_population(tmp_path)
    selected = SAMPLE.select_objects(manifest, analysis)
    assert selected == SAMPLE.select_objects(manifest, analysis)
    assert len({row["annotation_id"] for row in selected}) == 600
    actual = Counter((row["camera_id"], row["species"], row["size"]) for row in selected)
    assert actual == {
        (row["camera_id"], row["species"], row["size"]): row["sample"] for row in analysis["proposed_allocation"]
    }
    assert sum(float(row["sample_weight"]) for row in selected) == pytest.approx(3300)
    assert SAMPLE.read_protocol(output)["selected"] == selected
    assert all(
        not row["level"] and not row["reviewer"] and not row["visual_evidence"] for row in SAMPLE.read_reviews(output)
    )
    assert all(SAMPLE.digest(parent / "wildlife" / name) == value for name, value in analysis["source_sha256"].items())
    (output / "protocol.json").write_text("{}")
    with pytest.raises(ValueError, match="receipt changed"):
        SAMPLE.read_protocol(output)


def test_weighted_proportions_and_human_completion_gate(tmp_path):
    _, output, _, _ = fixture_population(tmp_path)
    protocol, rows = SAMPLE.read_protocol(output), SAMPLE.read_reviews(output)
    with pytest.raises(ValueError, match="All 600"):
        SAMPLE.weighted_proportions(rows, protocol)
    for row in rows:
        row.update(
            level="high" if row["species"] == "species-0" else "low",
            visual_evidence="Synthetic fixture evidence",
            reviewer="Synthetic fixture reviewer",
        )
    result = SAMPLE.weighted_proportions(rows, protocol)
    assert result["levels"]["high"]["proportion"] == pytest.approx(60 / 3300)
    assert result["levels"]["high"]["proportion"] != pytest.approx(60 / 600)
    assert sum(value["proportion"] for value in result["levels"].values()) == pytest.approx(1)
    assert result["levels"]["medium"]["upper"] > 0  # A homogeneous sample must not give a zero-width interval.
    status = SAMPLE.review_status(rows, protocol)
    assert status["complete"] == 600 and status["missing_support"]["medium"] == 30
    rows[0]["visual_evidence"] = ""
    assert SAMPLE.review_status(rows, protocol)["complete"] == 599
    rows[0]["sample_weight"] = "99"
    with pytest.raises(ValueError, match="metadata changed"):
        SAMPLE.validate_reviews(rows, protocol)


def test_preparation_again_preserves_existing_human_rows(tmp_path):
    parent, output, _, _ = fixture_population(tmp_path)
    rows = SAMPLE.read_reviews(output)
    rows[0].update(
        level="ambiguous", visual_evidence="Synthetic fixture evidence", reviewer="Synthetic fixture reviewer"
    )
    SAMPLE.write_reviews(output, rows)
    before = SAMPLE.digest(output / "occlusion_review_sample_600.csv")
    with patch.object(SAMPLE, "verify_parent", return_value={}):
        status = SAMPLE.prepare(parent, output, tmp_path / "images", tmp_path / "analysis.json")
    assert status["complete"] == 1
    assert SAMPLE.digest(output / "occlusion_review_sample_600.csv") == before


def test_http_autosave_conflict_security_and_restart(tmp_path):
    _, output, _, _ = fixture_population(tmp_path)
    server = SAMPLE.ReviewServer(output, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(url + "/api/state") as response:
            state = json.load(response)
        assert state["status"]["complete"] == 0
        change = {
            "annotation_id": state["rows"][0]["annotation_id"],
            "level": "medium",
            "visual_evidence": "Synthetic fixture evidence",
            "reviewer": "Synthetic fixture reviewer",
        }

        def post(revision, changes, token=None):
            request = Request(
                url + "/api/reviews",
                data=json.dumps({"revision": revision, "changes": changes}).encode(),
                headers={"Content-Type": "application/json", "X-Review-Token": token or server.token},
            )
            with urlopen(request) as response:
                return json.load(response)

        with pytest.raises(HTTPError) as caught:
            post(state["revision"], [change], token="wrong-token")
        assert caught.value.code == 403
        saved = post(state["revision"], [change])
        assert saved["status"]["complete"] == 1
        with pytest.raises(HTTPError) as caught:
            post(state["revision"], [change])
        assert caught.value.code == 409
        with pytest.raises(HTTPError) as caught:
            post(saved["revision"], [{**change, "sample_weight": "99"}])
        assert caught.value.code == 400
        assert SAMPLE.digest(output / "occlusion_review_sample_600.csv") == saved["revision"]
        with urlopen(url + "/api/download") as response:
            assert "attachment" in response.headers["Content-Disposition"]
            assert b"Synthetic fixture evidence" in response.read()
        with pytest.raises(HTTPError) as caught:
            urlopen(url + "/../protocol.json")
        assert caught.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert SAMPLE.review_status(SAMPLE.read_reviews(output), SAMPLE.read_protocol(output))["complete"] == 1


@pytest.mark.parametrize("level", ["low", "medium", "high"])
def test_worker_exactly_delegates_to_frozen_metrics_and_bootstrap(tmp_path, level):
    evaluator_spec = importlib.util.spec_from_file_location(
        "sample_evaluator", PROJECT / "scripts/evaluate_occlusion_sample.py"
    )
    evaluator = importlib.util.module_from_spec(evaluator_spec)
    evaluator_spec.loader.exec_module(evaluator)
    from neural_exam.wildlife.metrics import coco_evaluate
    from neural_exam.wildlife.uncertainty import bootstrap_metrics

    parent, output = tmp_path / "parent", tmp_path / "sample"
    (parent / "wildlife/trans_test").mkdir(parents=True)
    records, predictions, rows = [], [], []
    for index, grade in enumerate(SAMPLE.LEVELS[:3]):
        identifier = str(index)
        records.append(
            {
                "image_id": identifier,
                "sequence_id": identifier,
                "boxes": [{"annotation_id": identifier, "class_id": 0, "box": [0.5, 0.5, 0.2, 0.2]}],
            }
        )
        predictions.append({"image_id": identifier, "class_id": 0, "bbox": [0.4, 0.4, 0.2, 0.2], "confidence": 0.8})
        rows.append({"annotation_id": identifier, "level": grade})
    SAMPLE.write_json(parent / "wildlife/manifest.json", {"names": ["bird"], "records": {"trans_test": records}})
    SAMPLE.write_json(parent / "wildlife/trans_test/predictions.json", predictions)
    protocol = {"bootstrap_replicates": 4, "seed": 42}
    status = {"missing_support": {grade: 0 for grade in SAMPLE.LEVELS[:3]}}
    with patch.object(evaluator, "inputs", return_value=(protocol, rows, status, "fixture-signature")):
        evaluator.worker(parent, output, level, "fixture-signature")
    result = SAMPLE.read_json(output / "evaluation/fixture-signatur" / f"{level}.json")
    kwargs = {"occlusions": {row["annotation_id"]: row["level"] for row in rows}, "occlusion_group": level}
    assert result["metrics"] == coco_evaluate(records, predictions, ["bird"], **kwargs)
    assert result["uncertainty"] == bootstrap_metrics(records, predictions, ["bird"], 4, 42, **kwargs)
