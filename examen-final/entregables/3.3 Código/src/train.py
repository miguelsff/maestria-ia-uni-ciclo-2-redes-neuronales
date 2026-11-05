"""Train one reproducible PeruFoodNet configuration."""
import argparse
import json
import random
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from curves import CURVES_REVISION, render_curves
from datasets import FoodDataset
from evaluate import evaluate
from models import build_model, set_training_mode
from tracking import log_context, log_native_model, setup

PROJECT = Path(__file__).resolve().parents[1]


def update_best(score: float, epoch: int, best: float, best_epoch: int, stale: int) -> tuple[float, int, int, bool]:
    """Track strict improvement; ties count towards patience."""
    if score > best:
        return score, epoch, 0, True
    return best, best_epoch, stale + 1, False


def run_name(config: dict) -> str:
    """Build a run and model name accepted by the MLflow server."""
    rate = str(config["lr"]).replace(".", "p")
    return f"{config['family']}-{config['regime']}-{config['budget']}-lr{rate}-seed{config['seed']}"


def train(config: dict, parent_id: str | None = None) -> dict:
    """Train one configuration and persist its best validation checkpoint."""
    if config["image_size"] != 224:
        raise ValueError("This protocol requires image_size=224")
    if config["epochs"] < 1 or config["patience"] < 1:
        raise ValueError("epochs and patience must be positive")
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    rows = pd.read_csv(PROJECT / "data/manifest.csv")
    train_rows = rows[(rows.split == "train") & (rows[f"subset_{config['budget']}"] == 1)]
    val_rows = rows[rows.split == "val"]
    root = Path(config["dataset"])
    start = time.perf_counter()
    train_set = FoodDataset(root, train_rows, config["family"], True, config["cache"])
    val_set = FoodDataset(root, val_rows, config["family"], False, config["cache"])
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, generator=generator)
    # Metrics use the same deterministic evaluator for both splits.
    train_eval = FoodDataset(root, train_rows, config["family"], False, False)
    train_eval.images = train_set.images
    train_eval_loader = DataLoader(train_eval, batch_size=config["batch_size"])
    val_loader = DataLoader(val_set, batch_size=config["batch_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    weights = Path(config["pretrained"]) / ("resnet18-f37072fd.pth" if config["family"] == "resnet18" else "yolo26m-cls.pt")
    model = build_model(config["family"], weights, config["regime"]).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=config["lr"], weight_decay=config["weight_decay"])
    name = run_name(config)
    setup(config)
    with mlflow.start_run(run_name=name, parent_run_id=parent_id, nested=mlflow.active_run() is not None) as run:
        output = Path(config["output"]) / "runs" / run.info.run_id
        output.mkdir(parents=True, exist_ok=False)
        (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        (output / "model.txt").write_text(str(model), encoding="utf-8")
        log_context(config, train_rows, val_rows, PROJECT)
        mlflow.set_tag("phase", config["phase"])
        mlflow.log_params({"trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                           "total_parameters": sum(p.numel() for p in model.parameters())})
        frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
        bn_stats = {name: b.detach().clone() for name, b in model.named_buffers() if "running_" in name or "num_batches_tracked" in name}
        extractor = {name: p.detach().clone() for name, p in model.named_parameters()
                     if not (name.endswith("fc.weight") or name.endswith("fc.bias")
                             or name.endswith("linear.weight") or name.endswith("linear.bias"))}
        best, best_epoch, stale = -1.0, 0, 0
        history = []
        stopping_reason = "max_epochs"
        for epoch in range(1, config["epochs"] + 1):
            set_training_mode(model, config["regime"])
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                if logits.shape != (len(labels), 40):
                    raise ValueError(f"Invalid logits: {logits.shape}")
                loss = nn.functional.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()
            train_metrics = evaluate(model, train_eval_loader, device)
            val_metrics = evaluate(model, val_loader, device)
            metrics = {"train/loss": train_metrics["loss"], "train/accuracy": train_metrics["accuracy"],
                       "val/loss": val_metrics["loss"], "val/accuracy": val_metrics["accuracy"], "val/f1_macro": val_metrics["f1_macro"]}
            mlflow.log_metrics(metrics, step=epoch)
            history.append({"epoch": epoch, **metrics})
            print(json.dumps(history[-1]), flush=True)
            best, best_epoch, stale, improved = update_best(val_metrics["f1_macro"], epoch, best, best_epoch, stale)
            if improved:
                torch.save({"state_dict": model.state_dict(), "config": config, "epoch": epoch, "best_f1": best,
                            "architecture": model.model.yaml if config["family"] == "yolo26m" else None}, output / "best.pt")
            if stale >= config["patience"]:
                stopping_reason = "early_stopping"
                break
        if config["regime"] == "head":
            assert all(torch.equal(frozen[name], p) for name, p in model.named_parameters() if name in frozen)
            assert all(torch.equal(bn_stats[name], b) for name, b in model.named_buffers() if name in bn_stats)
        changed = sum(not torch.equal(extractor[name], p) for name, p in model.named_parameters() if name in extractor)
        if config["regime"] == "full" and changed == 0:
            raise RuntimeError("No extractor parameters changed during full training")
        mlflow.log_metric("extractor_tensors_changed", changed)
        (output / "environment.json").write_text(json.dumps({"torch": str(torch.__version__), "cuda": str(torch.version.cuda), "device": str(device)}, indent=2), encoding="utf-8")
        model.load_state_dict(torch.load(output / "best.pt", map_location=device, weights_only=True)["state_dict"])
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        render_curves(pd.DataFrame(history), output / "curves.png")
        sample = next(iter(val_loader))[0][:1]
        uri = log_native_model(model, sample, output, name, config["phase"])
        duration = time.perf_counter() - start
        memory = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0
        summary = {"run_id": run.info.run_id, "model_uri": uri, "best_epoch": best_epoch,
                   "best_f1": best, "epochs_completed": len(history), "stopping_reason": stopping_reason,
                   "seconds": duration, "peak_gpu_mib": memory}
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        mlflow.log_metrics({"best_epoch": best_epoch, "best_f1": best, "epochs_completed": len(history),
                            "seconds": duration, "peak_gpu_mib": memory})
        mlflow.set_tag("stopping_reason", stopping_reason)
        mlflow.set_tag("curves_revision", CURVES_REVISION)
        mlflow.log_artifacts(str(output))
        return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT / "configs/default.json"))
    parser.add_argument("--family", choices=["resnet18", "yolo26m"], required=True)
    parser.add_argument("--regime", choices=["head", "full"], required=True)
    parser.add_argument("--budget", type=int, choices=[10, 40], default=40)
    parser.add_argument("--lr", type=float, choices=[0.0001, 0.0003], default=0.0001)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--parent-id")
    parser.add_argument("--phase", default="smoke")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    config.update(family=args.family, regime=args.regime, budget=args.budget, lr=args.lr, phase=args.phase)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.seed is not None:
        config["seed"] = args.seed
    if args.parent_id:
        result = train(config, args.parent_id)
    else:
        setup(config)
        with mlflow.start_run(run_name=f"campaign-{args.phase}") as parent:
            mlflow.set_tag("phase", args.phase)
            result = train(config, parent.info.run_id)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
