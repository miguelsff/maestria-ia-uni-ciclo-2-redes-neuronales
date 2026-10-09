"""Exercise migration boundaries and reject corrupted or ambiguous scientific inputs."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/closure_runtime.py"
SPEC = importlib.util.spec_from_file_location("closure_runtime", SCRIPT)
RUNTIME = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNTIME
SPEC.loader.exec_module(RUNTIME)


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
