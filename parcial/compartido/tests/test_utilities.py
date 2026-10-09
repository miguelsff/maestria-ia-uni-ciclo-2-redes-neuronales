"""Safety and reproducibility cases for local assets and report generation."""

import io
import tarfile

import pytest
import torch
from torch.utils.data import TensorDataset

from neural_exam.common.cam import GradCAM
from neural_exam.common.download import safe_extract
from neural_exam.common.io import read_json
from neural_exam.common.training import fit, make_loader, seed_everything
from neural_exam.reporting import latex_escape


def test_archive_escape_is_rejected_before_writing(tmp_path):
    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as stream:
        member = tarfile.TarInfo("../escaped.txt")
        member.size = 1
        stream.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="escapes"):
        safe_extract(archive, tmp_path / "destination")
    assert not (tmp_path / "escaped.txt").exists()


def test_latex_external_text_is_escaped():
    assert latex_escape("a_b%") == r"a\_b\%"
    assert latex_escape(r"\input{x}") == r"\textbackslash{}input\{x\}"


def test_gradcam_keeps_differentiable_scores_and_normalizes_maps():
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(4, 2),
    )
    capture = GradCAM(model[0])
    try:
        maps = capture.maps(model(torch.ones(1, 3, 8, 8))[0, 0], (8, 8))
        assert maps.shape == (1, 8, 8)
        assert torch.isfinite(maps).all()
        assert maps.min() >= 0 and maps.max() <= 1
    finally:
        capture.close()


@pytest.mark.slow
def test_resume_restores_optimizer_rng_and_loader_state(tmp_path):
    config = {
        "seed": 42,
        "workers": 0,
        "device": "cpu",
        "amp": False,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "epochs": 2,
        "patience": 10,
    }
    dataset = TensorDataset(torch.tensor([[0.0], [1.0], [2.0], [3.0]]), torch.ones(4, 1), torch.ones(4, 1))

    def criterion(logits, target, weight):
        return ((logits - target).square() * weight).mean()

    def loaders():
        return make_loader(dataset, 2, config, True), make_loader(dataset, 2, config)

    seed_everything(42)
    expected = torch.nn.Sequential(torch.nn.Linear(1, 4), torch.nn.Dropout(0.2), torch.nn.Linear(4, 1))
    fit(expected, *loaders(), criterion, config, tmp_path / "full", "fixture")
    seed_everything(42)
    interrupted = torch.nn.Sequential(torch.nn.Linear(1, 4), torch.nn.Dropout(0.2), torch.nn.Linear(4, 1))
    calls = 0

    def fail_on_second_epoch(logits, target, weight):
        nonlocal calls
        calls += 1
        if calls == 5:
            raise RuntimeError("synthetic interruption")
        return criterion(logits, target, weight)

    with pytest.raises(RuntimeError, match="synthetic"):
        fit(interrupted, *loaders(), fail_on_second_epoch, config, tmp_path / "resumed", "fixture")
    assert read_json(tmp_path / "resumed/run.json")["status"] == "running"
    assert not (tmp_path / "resumed/complete.json").exists()
    resumed = torch.nn.Sequential(torch.nn.Linear(1, 4), torch.nn.Dropout(0.2), torch.nn.Linear(4, 1))
    fit(resumed, *loaders(), criterion, config, tmp_path / "resumed", "fixture", resume=True)
    for key, tensor in expected.state_dict().items():
        assert torch.equal(tensor, resumed.state_dict()[key])
    for directory in (tmp_path / "full", tmp_path / "resumed"):
        metadata = read_json(directory / "run.json")
        completion = read_json(directory / "complete.json")
        assert metadata["status"] == "complete"
        assert metadata["epochs_executed"] == completion["epochs_executed"] == 2
        assert metadata["best_val_loss"] == completion["best_val_loss"]
        assert fit(resumed, *loaders(), criterion, config, directory, "fixture", resume=True) == directory / "best.pt"
