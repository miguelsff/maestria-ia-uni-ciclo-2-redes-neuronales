"""Full MRI factorial ablation, held-out evaluation and study-level explanations."""

from functools import partial
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from neural_exam.common.cam import GradCAM
from neural_exam.common.figures import bar_plot, cam_plot
from neural_exam.common.io import file_sha256, read_json, write_csv, write_json
from neural_exam.common.metrics import multilabel_metrics
from neural_exam.common.training import fit, make_loader, resolve_device, restore_model, seed_everything
from neural_exam.mri.data import MRIDataset, prepare
from neural_exam.mri.labels import LABELS
from neural_exam.mri.loss import classification_loss
from neural_exam.mri.model import StudyClassifier


def label_agreement(records: list[dict]) -> dict:
    """Estimate observed rule disagreement separately from abstention coverage."""
    rows = []
    for index, label in enumerate(LABELS):
        for confidence_filter in (0.0, 0.8):
            eligible = [
                record
                for record in records
                if record["split"] == "gold"
                and record["weak"]["values"][index] is not None
                and record["weak"]["confidence"][index] >= confidence_filter
            ]
            agreement = sum(record["targets"][index] == record["weak"]["values"][index] for record in eligible)
            rows.append(
                {
                    "label": label,
                    "minimum_confidence": confidence_filter,
                    "observed": len(eligible),
                    "agreement": agreement / len(eligible) if eligible else None,
                    "estimated_disagreement": 1 - agreement / len(eligible) if eligible else None,
                }
            )
    return {"per_label": rows, "confidence_is": "heuristic rule weight, not a calibrated probability"}


def training_alpha(records: list[dict], strategy: str, threshold: float) -> torch.Tensor:
    """Estimate focal balance from labels actually retained by the training strategy."""
    targets = np.array(
        [[np.nan if value is None else value for value in record["targets"]] for record in records], dtype=float
    )
    if strategy == "high_confidence":
        confidence = np.array([record["weak"]["confidence"] for record in records])
        targets[confidence < threshold] = np.nan
    observed = np.isfinite(targets)
    positives = np.nansum(targets, axis=0)
    return torch.tensor(np.clip(1 - positives / observed.sum(0).clip(min=1), 0.1, 0.9), dtype=torch.float32)


def run(settings, resume=False, quick=False, manifest=None) -> dict:
    """Train the planned matrix without using the gold studies for model selection."""
    manifest = manifest if manifest is not None else prepare(settings)
    output = settings.output_root / "mri"
    records = manifest["records"]
    source_root = Path(__file__).resolve().parents[1]
    source_hashes = {
        name: file_sha256(source_root / name)
        for name in (
            "mri/data.py",
            "mri/labels.py",
            "mri/model.py",
            "mri/loss.py",
            "mri/pipeline.py",
            "common/training.py",
            "common/metrics.py",
            "common/cam.py",
        )
    }
    source_audit = read_json(settings.output_root / "audit/mri_data.json") if not quick else None
    if source_audit is not None:
        assert {row["study_id"] for row in source_audit["series"]} == {row["study_id"] for row in records}
    write_json(output / "label_agreement.json", label_agreement(records))
    splits = {split: [record for record in records if record["split"] == split] for split in ("train", "val", "gold")}
    if not all(splits.values()):
        raise ValueError("MRI train, validation and gold evaluation partitions must all contain studies.")
    matrix = list(product(settings.mri["architectures"], settings.mri["losses"], settings.mri["noise_strategies"]))
    if quick:
        matrix = [("scratch", "bce", "weighted")]
    results = []
    for architecture, loss_name, strategy in matrix:
        started = perf_counter()
        seed_everything(settings.training["seed"])
        if settings.training["device"] == "cuda":
            torch.cuda.reset_peak_memory_stats()
        datasets = {
            split: MRIDataset(rows, architecture, strategy, settings.mri["confidence_threshold"])
            for split, rows in splits.items()
        }
        # Drop only fully unsupervised studies under the high-confidence ablation.
        if strategy == "high_confidence":
            for split in ("train", "val"):
                datasets[split].records = [
                    record
                    for record in splits[split]
                    if any(
                        value is not None and confidence >= settings.mri["confidence_threshold"]
                        for value, confidence in zip(record["targets"], record["weak"]["confidence"], strict=True)
                    )
                ]
        alpha = training_alpha(datasets["train"].records, strategy, settings.mri["confidence_threshold"])
        model = StudyClassifier(
            architecture, pretrained=True, weights_path=settings.weights_root / "resnet50-11ad3fa6.pth"
        )
        run_dir = output / f"{architecture}_{loss_name}_{strategy}"
        criterion = partial(classification_loss, kind=loss_name, gamma=settings.mri["focal_gamma"], alpha=alpha)
        experiment_config = {
            **settings.training,
            "mri": settings.mri,
            "architecture": architecture,
            "loss": loss_name,
            "noise_strategy": strategy,
            "focal_alpha": alpha.tolist(),
            "training_source_sha256": source_hashes,
            "original_content_identity": source_audit["selected_original_content_identity"]
            if source_audit
            else "smoke",
            "pretrained_sha256": file_sha256(settings.weights_root / "resnet50-11ad3fa6.pth")
            if architecture == "resnet50"
            else None,
        }
        checkpoint = fit(
            model,
            make_loader(datasets["train"], settings.mri["batch_size"], settings.training, True),
            make_loader(datasets["val"], settings.mri["batch_size"], settings.training),
            criterion,
            experiment_config,
            run_dir,
            manifest["identity"],
            resume,
        )
        device = resolve_device(settings.training["device"])
        restore_model(model, checkpoint, device)
        probabilities, truth = [], []
        with torch.inference_mode():
            for image, target, _ in make_loader(datasets["gold"], settings.mri["batch_size"], settings.training):
                probabilities.append(model(image.to(device)).sigmoid().cpu().numpy())
                truth.append(target.numpy())
        metrics = multilabel_metrics(
            np.concatenate(truth), np.concatenate(probabilities), LABELS, settings.mri["threshold"]
        )
        write_json(run_dir / "metrics.json", metrics)
        write_json(
            run_dir / "execution.json",
            {
                "scientific_run": not quick,
                "selected_checkpoint_sha256": file_sha256(checkpoint),
                "data_identity": manifest["identity"],
                "gold_studies": len(splits["gold"]),
                "selection": "minimum weak-validation loss",
                "threshold": settings.mri["threshold"],
                "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
                "training_and_evaluation_seconds": perf_counter() - started,
            },
        )
        write_csv(
            run_dir / "predictions.csv",
            [
                {"StudyInstanceUID": record["study_id"], **dict(zip(LABELS, probability.tolist(), strict=True))}
                for record, probability in zip(splits["gold"], np.concatenate(probabilities), strict=True)
            ],
        )
        # Three fixed labels are requested, including a ligament, meniscus and fluid finding.
        cam_cases = []
        for label_index in (0, 2, 7):
            candidates = [i for i, record in enumerate(splits["gold"]) if record["targets"][label_index] == 1]
            if not candidates:
                continue
            sample_index = candidates[0]
            image, _, _ = datasets["gold"][sample_index]
            model.zero_grad(set_to_none=True)
            capture = GradCAM(model.cam_layer)
            try:
                heatmaps = capture.maps(model(image[None].to(device))[0, label_index], tuple(image.shape[-2:]))
                with np.load(splits["gold"][sample_index]["cache"], allow_pickle=False) as cached:
                    original = cached["image"][:, 1]
                strongest = int(heatmaps.flatten(1).mean(1).argmax())
                cam_plot(
                    original[strongest],
                    heatmaps[strongest].cpu().numpy(),
                    run_dir / f"cam_{label_index}.png",
                    LABELS[label_index],
                )
                maps = heatmaps.cpu().numpy()
                np.savez_compressed(run_dir / f"cam_{label_index}.npz", image=original, heatmaps=maps)
                cam_cases.append(
                    {
                        "label": LABELS[label_index],
                        "label_index": label_index,
                        "study_id": splits["gold"][sample_index]["study_id"],
                        "series_id": splits["gold"][sample_index]["series_id"],
                        "selected_stack": strongest,
                        "probability": float(np.concatenate(probabilities)[sample_index, label_index]),
                        "gold_target": 1,
                        "selection": "first gold-positive study; maximum mean normalized CAM stack",
                        "nonzero_fraction": float((maps[strongest] > 0).mean()),
                    }
                )
            finally:
                capture.close()
        write_json(run_dir / "cam_cases.json", cam_cases)
        results.append({"experiment": run_dir.name, "macro_auc": metrics["macro_auc"], "macro_f1": metrics["macro_f1"]})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_csv(output / "experiments.csv", results)
    prevalence = [sum(record["targets"][i] == 1 for record in splits["train"]) for i in range(12)]
    bar_plot(LABELS, prevalence, output / "prevalence.png", "Hallazgos positivos en entrenamiento", "Estudios")
    return {"task": "mri", "experiments": results, "scientific_run": not quick}
