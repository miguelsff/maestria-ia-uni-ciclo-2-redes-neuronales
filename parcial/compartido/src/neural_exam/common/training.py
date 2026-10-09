"""Supervised training with explicit device selection and resumable state."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from neural_exam.common.figures import history_plot
from neural_exam.common.io import fingerprint, read_json, write_csv, write_json


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch before creating models or loaders."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(requested: str) -> torch.device:
    """Fail on an unavailable requested GPU instead of silently training on CPU."""
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check Docker GPU access, or explicitly select --device cpu.")
    return torch.device(requested)


def seed_worker(worker_id: int) -> None:
    """Seed each data-loading process from its reproducible PyTorch seed."""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_loader(dataset, batch_size: int, config: dict, shuffle: bool = False) -> DataLoader:
    """Build bounded loaders; new workers on each epoch allow exact seed restoration."""
    generator = torch.Generator().manual_seed(config["seed"])
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config["workers"],
        pin_memory=config["device"] != "cpu",
        persistent_workers=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def _rng_state() -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), numpy_state[2], numpy_state[3], numpy_state[4]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value[0], np.array(value[1], dtype=np.uint32), value[2], value[3], value[4]))
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _save_checkpoint(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def fit(
    model,
    train_loader,
    val_loader,
    loss_function,
    config: dict,
    run_dir: Path,
    data_identity: str,
    resume: bool = False,
) -> Path:
    """Train until validation loss stops improving; never select using final test data."""
    run_dir.mkdir(parents=True, exist_ok=True)
    signature = fingerprint({"training": config, "data_identity": data_identity})
    metadata_path = run_dir / "run.json"
    if metadata_path.exists() and read_json(metadata_path)["signature"] != signature:
        raise ValueError("Existing run has different configuration or partitions; use a new output directory.")
    if (run_dir / "complete.json").exists():
        if not resume:
            raise FileExistsError("Run already complete. Use --resume to reuse its recorded checkpoint.")
        return run_dir / "best.pt"
    if (run_dir / "last.pt").exists() and not resume:
        raise FileExistsError("Interrupted run found. Use --resume or choose a new output directory.")
    device = resolve_device(config["device"])
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    amp_enabled = config["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    start_epoch, best_loss, stale, history = 0, float("inf"), 0, []
    if resume and (run_dir / "last.pt").exists():
        state = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best_loss, stale, history = state["epoch"], state["best_loss"], state["stale"], state["history"]
        _restore_rng(state["rng"])
        train_loader.generator.set_state(state["train_generator"])
        val_loader.generator.set_state(state["val_generator"])
    write_json(
        metadata_path,
        {
            "signature": signature,
            "configuration": config,
            "data_identity": data_identity,
            "torch_version": torch.__version__,
            "device": str(device),
            "status": "running",
            "model_selection": "minimum validation loss",
        },
    )
    for epoch in range(start_epoch, config["epochs"]):
        if stale >= config["patience"]:
            break
        averages = []
        for training, loader in ((True, train_loader), (False, val_loader)):
            model.train(training)
            numerator, denominator = 0.0, 0
            for image, target, weight in loader:
                image, target, weight = (value.to(device, non_blocking=True) for value in (image, target, weight))
                optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(training), torch.amp.autocast("cuda", enabled=amp_enabled):
                    loss = loss_function(model(image), target, weight)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1}.")
                if training:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                numerator += loss.item() * image.shape[0]
                denominator += image.shape[0]
            if not denominator:
                raise ValueError("Training and validation loaders must both contain examples.")
            averages.append(numerator / denominator)
        improved = averages[1] < best_loss
        best_loss = min(best_loss, averages[1])
        stale = 0 if improved else stale + 1
        history.append({"epoch": epoch + 1, "train_loss": averages[0], "val_loss": averages[1]})
        state = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "best_loss": best_loss,
            "stale": stale,
            "history": history,
            "rng": _rng_state(),
            "train_generator": train_loader.generator.get_state(),
            "val_generator": val_loader.generator.get_state(),
        }
        _save_checkpoint(run_dir / "last.pt", state)
        if improved:
            _save_checkpoint(run_dir / "best.pt", state)
        write_csv(run_dir / "history.csv", history)
        print(f"{run_dir.name}: epoch {epoch + 1}, validation loss {averages[1]:.5f}", flush=True)
    history_plot(history, run_dir / "loss.png")
    write_json(run_dir / "complete.json", {"epochs_executed": len(history), "best_val_loss": best_loss})
    write_json(
        metadata_path,
        {
            **read_json(metadata_path),
            "status": "complete",
            "epochs_executed": len(history),
            "best_val_loss": best_loss,
        },
    )
    return run_dir / "best.pt"


def restore_model(model, checkpoint: Path, device: torch.device):
    """Load a project's own state dictionary for final evaluation."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    return model.to(device).eval()
