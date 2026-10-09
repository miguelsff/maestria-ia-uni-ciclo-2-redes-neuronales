"""Exercise migration boundaries and reject corrupted or ambiguous scientific inputs."""

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/closure_runtime.py"
SPEC = importlib.util.spec_from_file_location("closure_runtime", SCRIPT)
RUNTIME = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNTIME
SPEC.loader.exec_module(RUNTIME)
RUN_SPEC = importlib.util.spec_from_file_location("run_exam", SCRIPT.with_name("run_exam.py"))
RUN = importlib.util.module_from_spec(RUN_SPEC)
RUN_SPEC.loader.exec_module(RUN)


def fixture_layout(tmp_path, shared=False):
    exam = tmp_path / "repository/parcial"
    root = exam / "compartido" if shared else exam
    for name in ("src/neural_exam/config.py", "configs/wildlife_v6.toml", "scripts/evaluate_occlusion_accepted.py",
                 "Dockerfile", "requirements.txt", "pyproject.toml"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic signed source\r\n")
    receipt = tmp_path / "freeze.json"
    receipt.write_text(json.dumps({"sha256": {name: RUNTIME.digest(root / name)
                                             for name in ("src/neural_exam/config.py", "configs/wildlife_v6.toml")}}))
    return exam, root, receipt


@pytest.mark.parametrize("shared", [False, True])
def test_virtual_workspace_preserves_bytes_and_scientific_paths(tmp_path, shared):
    exam, _, receipt = fixture_layout(tmp_path, shared)
    question = exam / "pregunta_3_fauna"
    for name in ("notebooks/03_wildlife.ipynb", "fuentes_informe/main.tex", "entregable/04_occlusion_582.ipynb"):
        path = question / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic question source\r\n")
    destination = exam / ".runtime/staged"
    result = RUNTIME.materialize(exam, destination, receipt)
    assert result["status"] == "verified"
    assert (destination / "src/neural_exam/config.py").read_bytes() == b"synthetic signed source\r\n"
    assert (destination / "reports/wildlife/main.tex").read_bytes() == b"synthetic question source\r\n"
    assert (destination / "notebooks/03_wildlife.ipynb").is_file()
    assert (destination / "deliveries/pregunta_3_fauna/04_occlusion_582.ipynb").is_file()
    with pytest.raises(ValueError, match="already exists"):
        RUNTIME.materialize(exam, destination, receipt)


def test_corruption_is_rejected_before_creating_a_workspace(tmp_path):
    exam, root, receipt = fixture_layout(tmp_path)
    (root / "configs/wildlife_v6.toml").write_bytes(b"synthetic signed source\n")
    destination = exam / ".runtime/staged"
    with pytest.raises(ValueError, match="Frozen source changed"):
        RUNTIME.materialize(exam, destination, receipt)
    assert not destination.exists()


def test_cannot_write_outside_calling_worktree(tmp_path):
    exam, _, receipt = fixture_layout(tmp_path)
    destination = tmp_path / "another-worktree"
    with pytest.raises(ValueError, match="calling worktree"):
        RUNTIME.materialize(exam, destination, receipt)
    assert not destination.exists()


def test_conflicting_legacy_and_shared_sources_are_rejected(tmp_path):
    exam, _, _ = fixture_layout(tmp_path)
    duplicate = exam / "compartido/scripts/evaluate_occlusion_accepted.py"
    duplicate.parent.mkdir(parents=True)
    duplicate.write_bytes(b"conflicting source\n")
    with pytest.raises(ValueError, match="Conflicting legacy/shared"):
        RUNTIME.workspace_inputs(exam)


def test_migrated_common_templates_resolve_report_includes_and_offline_review(tmp_path):
    exam, root, receipt = fixture_layout(tmp_path, shared=True)
    templates = root / "plantillas"
    sources = {
        "informes/preamble.tex": b"\\documentclass{article}\r\n",
        "informes/references.bib": b"@misc{fixture,title={Synthetic}}\r\n",
        "reviews/occlusion_offline.html": b'<script src="occlusion_offline.js"></script>\r\n',
        "reviews/occlusion_offline.js": b"console.log('Synthetic fixture');\r\n",
    }
    for name, content in sources.items():
        path = templates / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    report = exam / "pregunta_1_mri/fuentes_informe/main.tex"
    report.parent.mkdir(parents=True)
    report.write_bytes(b"\\input{preamble}\n\\bibliography{references}\n")
    destination = exam / ".runtime/migrated"
    RUNTIME.materialize(exam, destination, receipt)
    # The compiler copies common resources from reports/ into its task build.
    assert (destination / "reports/preamble.tex").read_bytes() == sources["informes/preamble.tex"]
    assert (destination / "reports/references.bib").read_bytes() == sources["informes/references.bib"]
    # Offline packaging resolves its HTML and JS independently from report resources.
    assert (destination / "reviews/occlusion_offline.html").read_bytes() == sources["reviews/occlusion_offline.html"]
    assert (destination / "reviews/occlusion_offline.js").read_bytes() == sources["reviews/occlusion_offline.js"]
    assert not (destination / "reports/informes").exists()
    assert not (destination / "reports/reviews").exists()


def test_one_command_closure_prepares_absolute_mounts_without_manual_environment(tmp_path):
    exam, root, receipt = fixture_layout(tmp_path, shared=True)
    parent = tmp_path / "parent"
    (parent / "receipts").mkdir(parents=True)
    (parent / "receipts/source-freeze.json").write_bytes(receipt.read_bytes())
    (root / "compose.closure.yaml").write_text("services: {}\n")
    with patch.object(RUN.subprocess, "run") as process:
        plan, environment = RUN.prepare_run(exam, parent, tmp_path / "accepted", "closure", "fixture-image")
        process.assert_not_called()
    assert environment["RUNTIME_HOST_DIR"] == plan["workspace"]
    assert environment["CLOSURE_OUTPUT_HOST_DIR"] == plan["derived"]
    assert environment["WILDLIFE_ORIGINAL_HOST_DIR"] == str(parent.resolve())
    assert plan["commands"][-1][-3:] == ["run", "--rm", "verify"]
    assert (Path(plan["workspace"]) / "src/neural_exam/config.py").is_file()
    assert plan["scientific_runner_executed"] is False


@pytest.mark.parametrize("task", ["mri", "flood", "wildlife"])
def test_scientific_plan_preserves_runner_bytes_and_redirects_output(tmp_path, task):
    exam, root, receipt = fixture_layout(tmp_path, shared=True)
    parent = tmp_path / "parent"
    (parent / "receipts").mkdir(parents=True)
    (parent / "receipts/source-freeze.json").write_bytes(receipt.read_bytes())
    runner = root / f"scripts/run_{task}.ps1"
    runner.write_bytes(b"# synthetic original runner\r\n")
    example = root / (".env.wildlife-v6.example" if task == "wildlife" else ".env.example")
    example.write_bytes(b"OUTPUT_HOST_DIR=original-output\r\nCCT_BACKGROUND_HOST_DIR=original-backgrounds\r\n")
    output = tmp_path / "fresh-scientific-output"
    with patch.dict(RUN.os.environ, {"CCT_IMAGES_HOST_DIR": "explicit-dataset", "OUTPUT_HOST_DIR": "original-output"}):
        with patch.object(RUN.subprocess, "run") as process:
            plan, environment = RUN.prepare_run(exam, parent, tmp_path / "accepted", task, "fixture-image", output)
            process.assert_not_called()
    assert (Path(plan["workspace"]) / f"scripts/run_{task}.ps1").read_bytes() == runner.read_bytes()
    assert environment["OUTPUT_HOST_DIR"] == str(output.resolve())
    assert environment["MRI_OUTPUT_DIR"] == str(output.resolve())
    assert environment["FLOOD_OUTPUT_DIR"] == str(output.resolve())
    assert environment["CCT_IMAGES_HOST_DIR"] == "explicit-dataset"
    assert not output.exists()
    assert str(Path(plan["workspace"]) / f"scripts/run_{task}.ps1") in plan["commands"][0]
    assert plan["fresh_training_reproduction_verified"] is False
    with pytest.raises(ValueError, match="fresh output"):
        RUN.prepare_run(exam, parent, tmp_path / "accepted", task, "fixture-image", parent)


@pytest.mark.parametrize("task,question", [("mri", "pregunta_1_mri"), ("flood", "pregunta_2_inundaciones")])
def test_archived_runner_report_is_flattened_without_editorial_sources(tmp_path, task, question):
    exam, _, receipt = fixture_layout(tmp_path, shared=True)
    report = exam / question / "fuentes_informe"
    sources = {"main.tex": b"\\input{preamble}\n", "discussion.tex": b"Original discussion\r\n",
               "finalized.json": b'{"reviewed": false}\r\n'}
    for name, content in sources.items():
        path = report / "latex_previo_d1" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    editorial = report / "latex_original/main.tex"
    editorial.parent.mkdir(parents=True)
    editorial.write_bytes(b"Editorial source with different content\n")
    destination = exam / ".runtime/migrated-reports"
    RUNTIME.materialize(exam, destination, receipt)
    for name, content in sources.items():
        assert (destination / "reports" / task / name).read_bytes() == content
    assert not (destination / "reports" / task / "latex_previo_d1").exists()
    assert not (destination / "reports" / task / "latex_original").exists()


def test_current_fauna_report_precedes_historical_archive(tmp_path):
    exam, _, receipt = fixture_layout(tmp_path, shared=True)
    report = exam / "pregunta_3_fauna/fuentes_informe"
    historical = report / "latex_previo_d1/main.tex"
    historical.parent.mkdir(parents=True)
    historical.write_bytes(b"Historical fauna report\n")
    (report / "main.tex").write_bytes(b"Final fauna source\r\n")
    destination = exam / ".runtime/current-fauna"
    RUNTIME.materialize(exam, destination, receipt)
    assert (destination / "reports/wildlife/main.tex").read_bytes() == b"Final fauna source\r\n"
