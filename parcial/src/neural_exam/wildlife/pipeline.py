"""Offline YOLO fine-tuning, held-out camera evaluation and ecological monitoring."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from importlib import metadata
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from neural_exam.common.cam import GradCAM
from neural_exam.common.figures import bar_plot, cam_plot
from neural_exam.common.io import file_sha256, fingerprint, read_csv, read_json, write_csv, write_json
from neural_exam.common.training import resolve_device, seed_everything
from neural_exam.wildlife.data import prepare
from neural_exam.wildlife.figures import cam_contact_sheets, training_curves
from neural_exam.wildlife.metrics import coco_evaluate
from neural_exam.wildlife.monitoring import activity_report
from neural_exam.wildlife.review import label_review_status, prepare_label_review
from neural_exam.wildlife.review import prepare_occlusion_review as prepare_occlusion_review


def predict_records(model, records, settings, cache_root: Path | None = None) -> list[dict]:
    """Stream bounded batches and cache completed batches with checkpoint/data signatures."""
    rows = []
    batch = settings.wildlife.get("prediction_batch", 8)
    device = 0 if resolve_device(settings.training["device"]).type == "cuda" else "cpu"
    signature = fingerprint(
        {
            "data": [(row["image_id"], row.get("source_sha256")) for row in records],
            "imgsz": settings.wildlife["image_size"],
            "nms_iou": settings.wildlife["nms_iou"],
            "conf": 0.001,
            "max_det": 300,
            "producer_sha256": file_sha256(Path(__file__)),
            "schema": 2,
        }
    )
    for start in range(0, len(records), batch):
        chunk = records[start : start + batch]
        path = cache_root / signature / f"{start:08d}.json" if cache_root else None
        if path and path.exists():
            saved = read_json(path)
            if saved["images"] != [record["image_id"] for record in chunk]:
                raise ValueError("Prediction cache image order mismatch.")
            rows.extend(saved["predictions"])
            continue
        predictions = []
        results = model.predict(
            [record["image"] for record in chunk],
            conf=0.001,
            iou=settings.wildlife["nms_iou"],
            imgsz=settings.wildlife["image_size"],
            max_det=300,
            batch=batch,
            stream=True,
            device=device,
            verbose=False,
        )
        for record, result in zip(chunk, results, strict=True):
            boxes = result.boxes
            for coordinates, confidence, class_id in zip(
                boxes.xyxyn.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist(), strict=True
            ):
                x1, y1, x2, y2 = coordinates
                predictions.append(
                    {
                        "image_id": record["image_id"],
                        "class_id": int(class_id),
                        "confidence": confidence,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                    }
                )
        if path:
            write_json(path, {"images": [record["image_id"] for record in chunk], "predictions": predictions})
        rows.extend(predictions)
        if start % (batch * 100) == 0:
            print(f"Inference: {min(start + batch, len(records))}/{len(records)}", flush=True)
    return rows


def robustness_metrics(records, predictions, names, review_path=None, minimum=30, require_full=False) -> dict:
    """Evaluate relative area and reviewed occlusion with declared ignored regions."""
    sizes = {
        "small_lt_1pct": (0, float(np.nextafter(10000.0, 0))),
        "medium_1_to_5pct": (10000, 50000),
        "large_gt_5pct": (float(np.nextafter(50000.0, np.inf)), 1e10),
    }
    result = {
        "size": {name: coco_evaluate(records, predictions, names, area_range=bounds) for name, bounds in sizes.items()}
    }
    if review_path is None:
        return result
    objects = {box["annotation_id"]: (record, box) for record in records for box in record["boxes"]}
    reviews = read_csv(review_path) if review_path.exists() else []
    if len({row["annotation_id"] for row in reviews}) != len(reviews):
        raise ValueError("Duplicate annotation IDs in occlusion review.")
    levels = {}
    for row in reviews:
        if row["annotation_id"] not in objects:
            raise ValueError("Occlusion review refers to an object outside this evaluation partition.")
        record, box = objects[row["annotation_id"]]
        if row["image_id"] != record["image_id"] or row["species"] != names[box["class_id"]]:
            raise ValueError("Occlusion image/species mismatch.")
        if row["level"] not in {"", "low", "medium", "high", "ambiguous"}:
            raise ValueError("Occlusion levels must be low, medium, high, ambiguous or blank.")
        if row["level"]:
            if not row.get("visual_evidence", "").strip() or not row.get("reviewer", "").strip():
                raise ValueError("Reviewed occlusion requires visual evidence and a human reviewer.")
            levels[row["annotation_id"]] = row["level"]
    coverage = Counter(levels.values())
    result["occlusion"] = {
        level: {
            "reviewed_objects": coverage[level],
            "meets_minimum": coverage[level] >= minimum,
            **coco_evaluate(records, predictions, names, occlusions=levels, occlusion_group=level),
        }
        for level in ("low", "medium", "high")
    }
    result["occlusion_status"] = (
        "reviewed"
        if all(coverage[level] >= minimum for level in ("low", "medium", "high"))
        and (not require_full or len(levels) == len(objects))
        else "manual_review_pending"
    )
    result["occlusion_coverage"] = {
        "total_objects": len(objects),
        "reviewed_objects": len(levels),
        "ambiguous_objects": coverage["ambiguous"],
        "unreviewed_objects": len(objects) - len(levels),
        "require_full": require_full,
    }
    result["occlusion_method"] = (
        "visual review of trans_test; other levels, ambiguous and unreviewed objects are ignored crowd regions; "
        + ("full object cohort" if require_full else "sampled analysis")
    )
    return result


def detection_cam(model, record, settings, output, names, target_class=None) -> dict:
    """Explain a fixed class/proposal before NMS and retain the letterbox geometry."""
    from ultralytics.data.augment import LetterBox

    class_id = record["boxes"][0]["class_id"] if target_class is None else target_class
    detector = model.model.to(resolve_device(settings.training["device"])).float().eval()
    flags = [parameter.requires_grad for parameter in detector.parameters()]
    for parameter in detector.parameters():
        parameter.requires_grad_(True)
    with Image.open(record["image"]) as image:
        rgb = np.asarray(image.convert("RGB"))
    letterboxed = LetterBox(new_shape=(settings.wildlife["image_size"],) * 2, auto=False)(image=rgb)
    tensor = (
        torch.from_numpy(letterboxed.copy()).permute(2, 0, 1).float()[None].to(next(detector.parameters()).device) / 255
    )
    detector.zero_grad(set_to_none=True)
    head = detector.model[-1]
    original_dynamic = head.dynamic
    head.dynamic = True
    capture = GradCAM(detector.model[settings.wildlife["cam_layer"]])
    try:
        raw = detector(tensor)
        decoded = raw[0] if isinstance(raw, tuple) else raw
        if decoded.ndim != 3 or decoded.shape[1] != 4 + len(names):
            raise ValueError("Expected a differentiable YOLOv8 [batch,4+classes,proposals] output.")
        scores = decoded[0, 4 + class_id]
        # Select the proposal closest to the target object; empty-photo cases use the maximum score.
        if record["boxes"]:
            box = next((box for box in record["boxes"] if box["class_id"] == class_id), record["boxes"][0])
            ratio = min(letterboxed.shape[1] / rgb.shape[1], letterboxed.shape[0] / rgb.shape[0])
            center = torch.tensor(
                [
                    box["box"][0] * rgb.shape[1] * ratio + (letterboxed.shape[1] - rgb.shape[1] * ratio) / 2,
                    box["box"][1] * rgb.shape[0] * ratio + (letterboxed.shape[0] - rgb.shape[0] * ratio) / 2,
                ],
                device=tensor.device,
            )
            distances = ((decoded[0, :2].T - center) ** 2).sum(1)
            closest = distances.topk(min(32, len(distances)), largest=False).indices
            proposal = closest[scores[closest].argmax()]
        else:
            proposal = scores.argmax()
        scalar = scores[proposal]
        heatmap = capture.maps(scalar, tensor.shape[-2:])[0].cpu().numpy()
        if not np.isfinite(heatmap).all():
            raise RuntimeError("Non-finite Grad-CAM output.")
        cam_plot(letterboxed, heatmap, output, f"Grad-CAM: {names[class_id]}")
        return {
            "class": names[class_id],
            "proposal": int(proposal),
            "score": float(scalar.detach()),
            "heatmap_range": [float(heatmap.min()), float(heatmap.max())],
        }
    finally:
        capture.close()
        head.dynamic = original_dynamic
        for parameter, flag in zip(detector.parameters(), flags, strict=True):
            parameter.requires_grad_(flag)


def _capture_period(record: dict) -> str:
    try:
        hour = datetime.fromisoformat(record.get("timestamp") or "").hour
    except (ValueError, TypeError):
        return "unknown"
    return "day" if 8 <= hour < 17 else "night" if hour >= 20 or hour < 5 else "transition"


def _box_iou(box: list, prediction: list) -> float:
    cx, cy, width, height = box
    x, y, predicted_width, predicted_height = prediction
    intersection = max(0, min(cx + width / 2, x + predicted_width) - max(cx - width / 2, x)) * max(
        0, min(cy + height / 2, y + predicted_height) - max(cy - height / 2, y)
    )
    return intersection / (width * height + predicted_width * predicted_height - intersection)


def _select_cam_records(manifest, predictions, split, confidence) -> list[tuple]:
    by_image = {}
    for prediction in predictions:
        if prediction["confidence"] >= confidence:
            by_image.setdefault(prediction["image_id"], []).append(prediction)
    pools = {"false_positive": [], "missed_or_wrong_class": [], "detected_iou50": []}
    ordered = sorted(manifest["records"][split], key=lambda row: row["image_id"])
    support = manifest["train_class_boxes"]
    for row in ordered:
        found = by_image.get(row["image_id"], [])
        if not row["boxes"] and found:
            pools["false_positive"].append(
                (row, max(found, key=lambda prediction: prediction["confidence"])["class_id"])
            )
        for box in row["boxes"]:
            detected = any(
                prediction["class_id"] == box["class_id"] and _box_iou(box["box"], prediction["bbox"]) >= 0.5
                for prediction in found
            )
            pools["detected_iou50" if detected else "missed_or_wrong_class"].append((row, box["class_id"]))
    selected, used = [], set()
    for kind, pool in pools.items():
        for index, period in enumerate(("day", "night")):
            available = [item for item in pool if item[0]["image_id"] not in used]
            if available:
                row, class_id = min(
                    available,
                    key=lambda item: (
                        _capture_period(item[0]) != period,
                        support[manifest["names"][item[1]]] * (1 if index == 0 else -1),
                        item[0]["image_id"],
                        item[1],
                    ),
                )
                selected.append((row, kind, class_id))
                used.add(row["image_id"])
    for row in ordered:
        if len(selected) >= 6:
            break
        if row["boxes"] and row["image_id"] not in used:
            selected.append((row, "representative", row["boxes"][0]["class_id"]))
            used.add(row["image_id"])
    return selected[:6]


def _cam_cases(model, manifest, predictions, settings, output) -> list[dict]:
    cases = []
    for split in ("cis_test", "trans_test"):
        candidates = _select_cam_records(manifest, predictions[split], split, settings.wildlife["confidence"])
        for index, (record, kind, class_id) in enumerate(candidates[:6]):
            destination = output / "cam_cases" / f"{split}_{index}.png"
            details = detection_cam(model, record, settings, destination, manifest["names"], class_id)
            cases.append(
                {
                    "split": split,
                    "image_id": record["image_id"],
                    "camera_id": record["camera_id"],
                    "case": kind,
                    "capture_period": _capture_period(record),
                    "training_support": manifest["train_class_boxes"][manifest["names"][class_id]],
                    "figure": str(destination),
                    **details,
                }
            )
    write_json(output / "cam_cases.json", cases)
    cam_contact_sheets(cases, output)
    return cases


def _monitoring_demo(model, manifest, settings, output, cache_root) -> dict:
    frames = manifest["monitoring_frames"]
    cameras = sorted(
        {row["camera_id"] for row in frames}, key=lambda value: (0, int(value)) if value.isdigit() else (1, value)
    )
    selected = None
    for camera in cameras:
        group = [row for row in frames if row["camera_id"] == camera]
        sequences = Counter(row["sequence_id"] for row in group)
        complete = sorted(
            {
                row["sequence_id"]
                for row in group
                if sequences[row["sequence_id"]] == row.get("sequence_frames") and sequences[row["sequence_id"]] >= 2
            }
        )
        if complete and all(row["timestamp"] for row in group):
            selected = sorted(
                group, key=lambda row: (row["timestamp"], row["sequence_id"], row.get("frame_number") or 0)
            )
            demo_sequence = complete[0]
            break
    if selected is None:
        raise ValueError("No new camera has timestamps and a complete multi-frame sequence for the demo.")
    predictions = predict_records(model, selected, settings, cache_root / "monitoring")
    report = activity_report(selected, predictions, manifest["names"], settings.wildlife["confidence"])
    report.update(
        camera_id=camera,
        sequence_id=demo_sequence,
        cohort="all official trans_test frames of one predetermined camera; includes frames excluded from box evaluation",
        timezone="capture timezone unspecified; no conversion",
        complete_sequence_frames=[row["image_id"] for row in selected if row["sequence_id"] == demo_sequence],
    )
    by_image = {}
    for prediction in predictions:
        if prediction["confidence"] >= settings.wildlife["confidence"]:
            by_image.setdefault(prediction["image_id"], []).append(prediction)
    sequence_rows = [
        {
            "image_id": row["image_id"],
            "camera_id": camera,
            "sequence_id": demo_sequence,
            "timestamp": row["timestamp"],
            "frame_number": row["frame_number"],
            "detections": len(by_image.get(row["image_id"], [])),
            "species": ", ".join(
                sorted({manifest["names"][prediction["class_id"]] for prediction in by_image.get(row["image_id"], [])})
            ),
        }
        for row in selected
        if row["sequence_id"] == demo_sequence
    ]
    report["sequence_frames"] = sequence_rows
    write_json(output / "activity.json", report)
    write_json(output / "activity_predictions.json", predictions)
    write_csv(output / "activity_camera_days.csv", report["camera_days"])
    write_csv(
        output / "activity_species_days.csv",
        report["species_days"],
        ["camera_id", "date", "species", "detections", "photos_with_species"],
    )
    write_csv(output / "activity_sequence.csv", sequence_rows)
    return report


def run(settings, resume=False, quick=False, manifest=None) -> dict:
    """Train or recover a run; smoke never predicts on either held-out test partition."""
    from ultralytics import YOLO
    from ultralytics import settings as yolo_settings

    yolo_settings.update({"sync": False, "weights_dir": str(settings.weights_root)})
    seed_everything(settings.training["seed"])
    manifest = manifest if manifest is not None else prepare(settings, quick=quick)
    output = settings.output_root / "wildlife"
    if not quick:
        if settings.wildlife.get("require_training_backgrounds") and not manifest["split_summary"]["train"]["backgrounds"]:
            raise RuntimeError("The scientific configuration requires genuine training background photographs.")
        prepare_label_review(manifest, output, settings.training["seed"])
        status = label_review_status(output / "label_review.csv")
        if status["status"] != "reviewed":
            raise RuntimeError(
                "Human label review is pending. Inspect label_review.html and approve its sample before full training."
            )
    weights = settings.weights_root / settings.wildlife["model"]
    if not weights.is_file():
        raise FileNotFoundError("Explicitly download local wildlife weights first.")
    if settings.training["amp"] and resolve_device(settings.training["device"]).type == "cuda":
        receipt = settings.output_root / "receipts/amp.json"
        if quick:
            receipt = settings.output_root.parent / "receipts/amp.json"
        if not receipt.exists() or read_json(receipt)["status"] != "passed":
            raise RuntimeError("Run doctor --gpu-check --amp-check before training.")
    versions = {name: metadata.version(name) for name in ("torch", "torchvision", "ultralytics")}
    package = Path(__file__).resolve().parents[1]
    runtime_sources = [*Path(__file__).parent.glob("*.py"), *(package / "common").glob("*.py")]
    runtime_sources += [package / "config.py", package / "cli.py"]
    source_hashes = {path.relative_to(package).as_posix(): file_sha256(path) for path in runtime_sources}
    signature = fingerprint(
        {
            "configuration": settings.wildlife,
            "training": settings.training,
            "data": manifest["identity"],
            "quick": quick,
            "versions": versions,
            "pretrained_sha256": file_sha256(weights),
            "source_sha256": source_hashes,
        }
    )
    run_dir = output / "yolov8s"
    run_metadata = run_dir / "run.json"
    if run_metadata.exists() and read_json(run_metadata)["signature"] != signature:
        raise ValueError("YOLO configuration, environment or dataset changed; use a fresh output.")
    best, last = run_dir / "weights/best.pt", run_dir / "weights/last.pt"
    completed = run_dir / "training_complete.json"
    if (best.exists() or last.exists() or completed.exists()) and not run_metadata.is_file():
        raise ValueError("Existing YOLO checkpoints have no matching run metadata; use a fresh output.")
    if (last.exists() or completed.exists()) and not resume:
        raise FileExistsError("Existing YOLO run. Use --resume or a fresh output.")
    if not completed.exists():
        recovered = bool(
            last.exists()
            and best.exists()
            and torch.load(last, map_location="cpu", weights_only=False).get("epoch") == -1
        )
        if not recovered:
            write_json(
                run_metadata,
                {
                    "signature": signature,
                    "model": settings.wildlife["model"],
                    "seed": settings.training["seed"],
                    "versions": versions,
                    "source_sha256": source_hashes,
                    "scientific_run": not quick,
                    "selection": "maximum cis_val mAP50:95; test sets reserved",
                    "background": "explicit empty photographs",
                },
            )
            model = YOLO(str(last if resume and last.is_file() else weights))
            if resume and last.is_file():
                model.train(resume=True)
            else:
                model.train(
                    data=str(output / "dataset/data.yaml"),
                    epochs=settings.training["epochs"],
                    patience=settings.training["patience"],
                    imgsz=settings.wildlife["image_size"],
                    batch=settings.wildlife["batch_size"],
                    workers=settings.training["workers"],
                    device=0 if resolve_device(settings.training["device"]).type == "cuda" else "cpu",
                    amp=settings.training["amp"],
                    seed=settings.training["seed"],
                    deterministic=True,
                    optimizer="AdamW",
                    lr0=settings.training["learning_rate"],
                    weight_decay=settings.training["weight_decay"],
                    cache=False,
                    mosaic=1.0,
                    fliplr=0.5,
                    flipud=0.0,
                    close_mosaic=0 if quick else 10,
                    fraction=1.0,
                    project=str(output),
                    name="yolov8s",
                    exist_ok=True,
                    plots=True,
                )
        if not best.is_file():
            raise FileNotFoundError("YOLO did not produce best.pt.")
        write_json(
            completed, {"checkpoint": str(best), "sha256": file_sha256(best), "recovered_finished_backend": recovered}
        )
    checkpoint_hash = file_sha256(best)
    if checkpoint_hash != read_json(completed)["sha256"]:
        raise ValueError("The selected checkpoint changed after training completion.")
    model = YOLO(str(best))
    cache_root = output / "prediction_cache" / checkpoint_hash
    if quick:
        predictions = predict_records(model, manifest["records"]["val"], settings, cache_root)
        metrics = coco_evaluate(manifest["records"]["val"], predictions, manifest["names"])
        cam_record = next(row for row in manifest["records"]["val"] if row["boxes"])
        cam = detection_cam(model, cam_record, settings, output / "cam_diagnostic.png", manifest["names"])
        write_json(
            output / "cam_diagnostic.json",
            {"scientific_run": False, "split": "val", "image_id": cam_record["image_id"], **cam},
        )
        write_json(
            output / "smoke_validation.json", {"metrics": metrics, "scientific_run": False, "test_partitions_used": []}
        )
        return {"task": "wildlife", "scientific_run": False, "status": "smoke_passed", "test_partitions_used": []}
    training_curves(run_dir / "results.csv", output / "training_curves.png")
    scores, predictions = {}, {}
    for split in ("cis_test", "trans_test"):
        predictions[split] = predict_records(model, manifest["records"][split], settings, cache_root / split)
        scores[split] = coco_evaluate(manifest["records"][split], predictions[split], manifest["names"])
        scores[split]["protocol"] = {
            "checkpoint_sha256": checkpoint_hash,
            "conf": 0.001,
            "nms_iou": settings.wildlife["nms_iou"],
            "max_detections": 300,
            "categories_evaluated": sum(row["support"] > 0 for row in scores[split]["per_species"]),
        }
        write_json(output / split / "predictions.json", predictions[split])
        write_json(output / split / "metrics.json", scores[split])
        write_csv(output / split / "per_species.csv", scores[split]["per_species"])
    cis, trans = scores["cis_test"]["map50_95"], scores["trans_test"]["map50_95"]
    gap = cis - trans if cis is not None and trans is not None else None
    write_json(
        output / "generalization.json",
        {
            "cis_map50": scores["cis_test"]["map50"],
            "trans_map50": scores["trans_test"]["map50"],
            "cis_map50_95": cis,
            "trans_map50_95": trans,
            "gap": gap,
        },
    )
    full_occlusion = settings.wildlife.get("occlusion_review_mode") == "full"
    review_path = prepare_occlusion_review(manifest, output, settings.training["seed"], full=full_occlusion)
    robustness = robustness_metrics(
        manifest["records"]["trans_test"],
        predictions["trans_test"],
        manifest["names"],
        review_path,
        settings.wildlife["occlusion_min_examples"],
        require_full=full_occlusion,
    )
    robustness["size_by_split"] = {
        "cis_test": robustness_metrics(manifest["records"]["cis_test"], predictions["cis_test"], manifest["names"])[
            "size"
        ],
        "trans_test": robustness["size"],
    }
    write_json(output / "robustness.json", robustness)
    _monitoring_demo(model, manifest, settings, output, cache_root)
    cases = _cam_cases(model, manifest, predictions, settings, output)
    # Retain the figure expected by the original report while preserving all twelve cases.
    first_unseen = next(case for case in cases if case["split"] == "trans_test")
    (output / "cam_unseen.png").write_bytes(Path(first_unseen["figure"]).read_bytes())
    bar_plot(
        manifest["names"],
        list(manifest["train_class_boxes"].values()),
        output / "class_balance.png",
        "Distribución de animales en entrenamiento",
        "Cajas anotadas",
    )
    write_csv(
        output / "experiments.csv",
        [
            {
                "experiment": "yolov8s",
                "cis_map50": scores["cis_test"]["map50"],
                "trans_map50": scores["trans_test"]["map50"],
                "cis_map50_95": cis,
                "trans_map50_95": trans,
                "geographic_gap": gap,
            }
        ],
    )
    write_json(
        run_dir / "evaluation_complete.json",
        {
            "checkpoint_sha256": checkpoint_hash,
            "occlusion_status": robustness["occlusion_status"],
            "cam_cases": len(cases),
            "source_sha256": source_hashes,
        },
    )
    return {
        "task": "wildlife",
        "geographic_gap": gap,
        "scientific_run": True,
        "occlusion_status": robustness["occlusion_status"],
    }
