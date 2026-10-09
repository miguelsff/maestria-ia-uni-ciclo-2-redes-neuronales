"""Read-only scientific audit; write receipts only to the separate evidence mount."""

import argparse
import csv
import gc
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

DATA = Path("/data")
SOURCE = Path("/workspace/src/neural_exam")
EVIDENCE = Path("/evidence")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def same(actual, expected, context=""):
    if isinstance(expected, dict):
        for key, value in expected.items():
            same(actual[key], value, f"{context}/{key}")
    elif isinstance(expected, list):
        assert len(actual) == len(expected), context
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            same(left, right, f"{context}/{index}")
    elif isinstance(expected, float):
        assert actual is not None and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), (
            context, actual, expected
        )
    else:
        assert actual == expected, (context, actual, expected)


def source_checks(hashes):
    return {name: digest(SOURCE / name) == value for name, value in hashes.items()}


def save(name, result):
    result.update(audited_at_utc=datetime.now(timezone.utc).isoformat(), status="passed")
    (EVIDENCE / f"{name}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{name}: passed; receipt saved separately", flush=True)


def audit_mri():
    from neural_exam.mri.labels import LABELS, extract_labels

    root = DATA / "exam-output-mri/stratified-v2/mri"
    manifest = read_json(root / "manifest.json")
    original = read_csv(DATA / "rsna-knee-abnormality-detection/train.csv")
    assert list(original[0])[2:] == LABELS == manifest["labels"]
    original_by_id = {row["StudyInstanceUID"]: row for row in original}
    records = manifest["records"]
    assert len(records) == len({row["study_id"] for row in records}) == 600
    split_counts = Counter(row["split"] for row in records)
    assert split_counts == {"train": 434, "val": 108, "gold": 58}
    gold = {row["study_id"]: row for row in records if row["split"] == "gold"}
    for row in records:
        raw = original_by_id[row["study_id"]]
        same(extract_labels(raw["Report"]), row["weak"], row["study_id"])
        if row["split"] == "gold":
            same([float(raw[label]) for label in LABELS], row["targets"])
        else:
            same(row["weak"]["values"], row["targets"])
        cached = DATA / "exam-output-mri/stratified-v2" / row["cache"].removeprefix("/outputs/")
        with np.load(cached, allow_pickle=False) as arrays:
            values = arrays["image"]
            assert values.shape == (8, 3, 224, 224) and values.dtype == np.float32
            assert np.isfinite(values).all() and values.min() >= 0 and values.max() <= 1
    agreement = []
    for index, label in enumerate(LABELS):
        for cutoff in (0.0, 0.8):
            eligible = [r for r in gold.values() if r["weak"]["values"][index] is not None
                        and r["weak"]["confidence"][index] >= cutoff]
            ratio = sum(r["weak"]["values"][index] == r["targets"][index] for r in eligible) / len(eligible)
            agreement.append(dict(label=label, minimum_confidence=cutoff, observed=len(eligible),
                                  agreement=ratio, estimated_disagreement=1-ratio))
    same(read_json(root / "label_agreement.json")["per_label"], agreement)
    prevalence = read_csv(root / "prevalence.csv")
    for stored in prevalence:
        index = LABELS.index(stored["label"])
        cohort = [r for r in records if r["split"] == stored["split"]]
        assert int(stored["positive"]) == sum(r["targets"][index] == 1 for r in cohort)
        assert int(stored["observed"]) == sum(r["targets"][index] is not None for r in cohort)
    experiments = []
    for architecture, loss, strategy in product(("scratch", "resnet50"), ("bce", "focal"),
                                                ("weighted", "high_confidence")):
        name = f"{architecture}_{loss}_{strategy}"
        run = root / name
        execution, metadata = read_json(run / "execution.json"), read_json(run / "run.json")
        assert execution["scientific_run"] and execution["data_identity"] == manifest["identity"]
        assert execution["selected_checkpoint_sha256"] == digest(run / "best.pt")
        assert metadata["data_identity"] == manifest["identity"]
        if architecture == "resnet50":
            assert metadata["configuration"]["pretrained_sha256"] == digest(
                root.parent / "cache/weights/resnet50-11ad3fa6.pth")
        predictions = read_csv(run / "predictions.csv")
        assert len(predictions) == len({r["StudyInstanceUID"] for r in predictions}) == 58
        assert {r["StudyInstanceUID"] for r in predictions} == set(gold)
        stored = read_json(run / "metrics.json")
        metrics = []
        for index, label in enumerate(LABELS):
            truth = np.array([gold[r["StudyInstanceUID"]]["targets"][index] for r in predictions])
            scores = np.array([float(r[label]) for r in predictions])
            assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
            assert set(truth) == {0.0, 1.0}
            predicted = scores >= 0.5
            tp = int(((truth == 1) & predicted).sum())
            fp = int(((truth == 0) & predicted).sum())
            fn = int(((truth == 1) & ~predicted).sum())
            metrics.append(dict(label=label, support=58, positives=int(truth.sum()),
                                auc=float(roc_auc_score(truth, scores)),
                                precision=tp/(tp+fp) if tp+fp else 0.0,
                                recall=tp/(tp+fn) if tp+fn else 0.0,
                                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0))
        same(stored["per_label"], metrics, name)
        same(stored["macro_auc"], float(np.mean([r["auc"] for r in metrics])))
        same(stored["macro_f1"], float(np.mean([r["f1"] for r in metrics])))
        history, complete = read_csv(run / "history.csv"), read_json(run / "complete.json")
        assert len(history) == complete["epochs_executed"] <= 50
        assert all(math.isfinite(float(r[k])) for r in history for k in ("train_loss", "val_loss"))
        best = min(history, key=lambda r: float(r["val_loss"]))
        checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
        assert checkpoint["epoch"] == int(best["epoch"])
        same(checkpoint["best_loss"], float(best["val_loss"]))
        assert all(torch.isfinite(t).all() for t in checkpoint["model"].values())
        cam = []
        for index in (0, 2, 7):
            with np.load(run / f"cam_{index}.npz", allow_pickle=False) as arrays:
                heat = arrays["heatmaps"]
                assert heat.shape == arrays["image"].shape == (8, 224, 224)
                assert np.isfinite(heat).all() and heat.min() >= 0 and heat.max() <= 1
                cam.append(dict(label=LABELS[index], minimum=float(heat.min()), maximum=float(heat.max())))
        experiments.append(dict(experiment=name, epochs=len(history), best_epoch=checkpoint["epoch"],
                                macro_auc=stored["macro_auc"], macro_f1=stored["macro_f1"],
                                source_matches=source_checks(metadata["configuration"]["training_source_sha256"]),
                                cams=cam, historical_run_status=metadata.get("status")))
        del checkpoint
    save("mri-scientific", dict(data_identity=manifest["identity"], split_counts=split_counts,
         original_labels=LABELS, labels_reextracted=600, cached_studies_verified=600,
         expert_predictions_verified=464, agreement=agreement, prevalence=prevalence, experiments=experiments))


def audit_flood():
    import rasterio
    from neural_exam.flood.data import SPLITS, read_pair

    root = DATA / "exam-output-flood/flood"
    manifest = read_json(root / "manifest.json")
    by_split = manifest["records"]
    sets = {split: {r["id"] for r in records} for split, records in by_split.items()}
    assert sum(len(s) for s in sets.values()) == len(set().union(*sets.values()))
    for split, filename in SPLITS.items():
        path = DATA / "Sen1Floods11/v1.1/splits/flood_handlabeled" / filename
        assert digest(path) == manifest["split_sha256"][split]
        with path.open(encoding="utf-8") as stream:
            official = {"_".join(Path(r[0]).stem.split("_")[:2]) for r in csv.reader(stream)}
        assert sets[split] | set(manifest["empty_chips"][split]) == official
    assert all(r["country"] != "Bolivia" for s in ("train", "val", "weak") for r in by_split[s])
    assert all(r["country"] == "Bolivia" for r in by_split["bolivia"])
    assert manifest["band_order"] == ["VV", "VH"] and manifest["units"] == "dB"
    targets = {}
    for split in ("test", "bolivia"):
        for record in by_split[split]:
            local = {**record, **{k: str(DATA / record[k].removeprefix("/datasets/")) for k in ("image", "label")}}
            for key in ("image", "label"):
                assert digest(local[key]) == record["source_sha256"][key]
            _, target, profile = read_pair(local)
            assert int((target >= 0).sum()) == record["valid_pixels"]
            targets[record["id"]] = (target, profile)
    results = []
    for supervision, loss in product(("gold", "gold_weak"), ("bce", "dice", "combined")):
        name = f"{supervision}_{loss}"
        run = root / name
        execution, metadata = read_json(run / "execution.json"), read_json(run / "run.json")
        assert execution["scientific_run"] and metadata["data_identity"] == manifest["identity"]
        history, complete = read_csv(run / "history.csv"), read_json(run / "complete.json")
        assert len(history) == complete["epochs_executed"] <= 50
        best = min(history, key=lambda r: float(r["val_loss"]))
        checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
        assert checkpoint["epoch"] == int(best["epoch"])
        same(checkpoint["best_loss"], float(best["val_loss"]))
        assert all(torch.isfinite(t).all() for t in checkpoint["model"].values())
        for split in ("test", "bolivia"):
            stored = read_json(run / split / "metrics.json")
            chips = {r["chip_id"]: r for r in read_csv(run / split / "per_chip.csv")}
            assert set(chips) == sets[split]
            totals = dict.fromkeys(("tp", "fp", "fn", "tn"), 0)
            ious, reference = [], []
            for record in by_split[split]:
                target, profile = targets[record["id"]]
                with rasterio.open(run / split / f"{record['id']}_probability.tif") as raster:
                    assert raster.count == 1 and raster.dtypes == ("float32",) and raster.nodata == -1
                    assert raster.crs == profile["crs"] and raster.transform == profile["transform"]
                    probability = raster.read(1)
                valid = target >= 0
                assert np.isfinite(probability[valid]).all() and (probability[~valid] == -1).all()
                assert ((probability[valid] >= 0) & (probability[valid] <= 1)).all()
                prediction = probability >= stored["threshold"]
                counts = dict(tp=int(((target == 1) & prediction).sum()),
                              fp=int(((target == 0) & prediction).sum()),
                              fn=int(((target == 1) & ~prediction).sum()),
                              tn=int(((target == 0) & ~prediction).sum()))
                for key, count in counts.items():
                    assert count == int(chips[record["id"]][key])
                    totals[key] += count
                union = counts["tp"] + counts["fp"] + counts["fn"]
                if union:
                    ious.append(counts["tp"] / union)
                reference.append((counts["tp"] + 1e-7) / (union + 1e-7))
            same(stored["mean_chip_iou"], float(np.mean(ious)))
            same(stored["reference_mean_chip_iou"], float(np.mean(reference)))
            same(stored["defined_iou_chips"], len(ious))
            same(stored["empty_union_chips"], len(chips)-len(ious))
            same({k: stored["pooled_pixels"][k] for k in totals}, totals)
            tp, fp, fn = (totals[k] for k in ("tp", "fp", "fn"))
            for key, num, den in (("iou", tp, tp+fp+fn), ("precision", tp, tp+fp),
                                  ("recall", tp, tp+fn), ("f1", 2*tp, 2*tp+fp+fn),
                                  ("omission", fn, tp+fn), ("commission", fp, tp+fp)):
                same(stored["pooled_pixels"][key], num/den if den else None)
            results.append(dict(experiment=name, split=split, chips=len(chips), best_epoch=checkpoint["epoch"],
                                mean_chip_iou=stored["mean_chip_iou"], metrics=stored["pooled_pixels"],
                                checkpoint_sha256=digest(run / "best.pt")))
        del checkpoint
        print(f"Verified flood {name}", flush=True)
    save("flood-scientific", dict(data_identity=manifest["identity"],
         split_counts={s: len(r) for s,r in by_split.items()}, official_splits_verified=4,
         all_partitions_disjoint=True, bolivia_excluded_from_training=True,
         geotiffs_verified=sum(r["chips"] for r in results), results=results))


def audit_wildlife():
    from neural_exam.wildlife.metrics import coco_evaluate
    from neural_exam.wildlife.monitoring import activity_report

    root = DATA / "exam-output-wildlife-v6/wildlife"
    manifest = read_json(root / "manifest.json")
    freeze = read_json(root.parent / "receipts/source-freeze.json")
    checks = {name: digest(Path("/workspace") / name) == value for name,value in freeze["sha256"].items()}
    assert all(checks.values()), checks
    completed = read_json(root / "yolov8s/training_complete.json")
    checkpoint_hash = digest(root / "yolov8s/weights/best.pt")
    assert checkpoint_hash == completed["sha256"]
    run = read_json(root / "yolov8s/run.json")
    assert run["scientific_run"] and all(source_checks(run["source_sha256"]).values())
    history = read_csv(root / "yolov8s/results.csv")
    history = [{k.strip(): v for k,v in row.items()} for row in history]
    best = max(history, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    summary = dict(data_identity=manifest["identity"], checkpoint_sha256=checkpoint_hash,
                   frozen_sources_verified=len(checks), epochs=len(history),
                   selected_epoch=int(best["epoch"]), splits={}, size={}, monitoring={})
    robustness = read_json(root / "robustness.json")
    bounds = dict(small_lt_1pct=(0, float(np.nextafter(10000.0, 0))),
                  medium_1_to_5pct=(10000, 50000),
                  large_gt_5pct=(float(np.nextafter(50000.0, np.inf)), 1e10))
    for split in ("cis_test", "trans_test"):
        records = manifest["records"][split]
        predictions = read_json(root / split / "predictions.json")
        stored = read_json(root / split / "metrics.json")
        assert stored["protocol"]["checkpoint_sha256"] == checkpoint_hash
        recomputed = coco_evaluate(records, predictions, manifest["names"])
        same(stored, recomputed, split)
        summary["splits"][split] = dict(images=len(records), detections=len(predictions), **recomputed)
        summary["size"][split] = {}
        for group, area in bounds.items():
            recalculated = coco_evaluate(records, predictions, manifest["names"], area_range=area)
            same(robustness["size_by_split"][split][group], recalculated, f"{split}/{group}")
            summary["size"][split][group] = {k: recalculated[k] for k in ("map50", "map50_95", "eligible_objects")}
            gc.collect()
        del predictions
        gc.collect()
        print(f"Verified wildlife {split} and three size strata", flush=True)
    generalization = read_json(root / "generalization.json")
    same(generalization["gap"], summary["splits"]["cis_test"]["map50_95"]
         - summary["splits"]["trans_test"]["map50_95"])
    same(generalization["gap_map50"], summary["splits"]["cis_test"]["map50"]
         - summary["splits"]["trans_test"]["map50"])
    activity = read_json(root / "activity.json")
    frames = [r for r in manifest["monitoring_frames"] if r["camera_id"] == activity["camera_id"]]
    predictions = read_json(root / "activity_predictions.json")
    recalculated = activity_report(frames, predictions, manifest["names"], activity["confidence"])
    same(activity, recalculated)
    training_cameras = {r["camera_id"] for r in manifest["records"]["train"]}
    assert activity["camera_id"] not in training_cameras
    assert sum(r["photos"] for r in activity["camera_days"]) == len(frames)
    summary["monitoring"] = dict(camera=activity["camera_id"], frames=len(frames),
         camera_days=len(activity["camera_days"]), species_day_rows=len(activity["species_days"]),
         detections=sum(r["detections"] for r in activity["species_days"]),
         photos_with_fauna=sum(r["photos_with_fauna"] for r in activity["camera_days"]),
         distinct_species=len({r["species"] for r in activity["species_days"]}),
         unknown_dates=sum(r["date"] == "unknown" for r in activity["camera_days"]))
    summary["gap_map50_95"] = generalization["gap"]
    summary["gap_map50"] = generalization["gap_map50"]
    save("wildlife-scientific", summary)


def audit_occlusion():
    from neural_exam.wildlife.metrics import coco_evaluate

    root = DATA / "exam-output-wildlife-v6/wildlife"
    accepted_root = DATA / "exam-output-wildlife-v6-occlusion-sample-600"
    manifest = read_json(root / "manifest.json")
    predictions = read_json(root / "trans_test/predictions.json")
    rows = read_csv(accepted_root / "occlusion_review_accepted_582.csv")
    acceptance = read_json(accepted_root / "review_acceptance.json")
    assert digest(accepted_root / "occlusion_review_accepted_582.csv") == acceptance["accepted_csv_sha256"]
    mapping = {r["annotation_id"]: r["level"] for r in rows}
    counts = Counter(mapping.values())
    assert counts == {"low": 340, "medium": 69, "high": 48, "ambiguous": 125}
    signature_root = accepted_root / "accepted_evaluation/6b2983c33abccbc1"
    results = {}
    for level in ("low", "medium", "high"):
        metrics = coco_evaluate(manifest["records"]["trans_test"], predictions,
                                manifest["names"], occlusions=mapping, occlusion_group=level)
        assert metrics["eligible_objects"] == counts[level]
        existing = signature_root / f"{level}.json"
        complete = existing.exists()
        if complete:
            stored = read_json(existing)
            same(stored["metrics"], metrics, f"accepted/{level}")
            assert stored["uncertainty"]["replicates"] == 200
            assert stored["signature"].startswith(signature_root.name)
        results[level] = dict(metrics=metrics, saved_evaluation_complete=complete)
        print(f"Accepted occlusion {level}: point metrics independently recomputed", flush=True)
        gc.collect()
    save("occlusion-scientific", dict(accepted=counts, results=results,
         evaluation_includes_all_trans_frames=len(manifest["records"]["trans_test"]),
         fixed_checkpoint_sha256=digest(root / "yolov8s/weights/best.pt"),
         aggregate_completed=(accepted_root / "accepted_evaluation_latest.json").exists(),
         bootstrap_recomputed=False))


def audit_flood_cases():
    import rasterio
    from scipy.ndimage import binary_dilation, binary_erosion
    from neural_exam.flood.data import read_pair

    root = DATA / "exam-output-flood/flood"
    manifest = read_json(root / "manifest.json")
    expected = read_json(root.parent / "audit/qualitative-statistics.json")
    checked = []
    for case in expected:
        split, chip = case["split"], case["statistics"]["chip_id"]
        record = next(r for r in manifest["records"][split] if r["id"] == chip)
        local = {**record, **{k: str(DATA / record[k].removeprefix("/datasets/")) for k in ("image", "label")}}
        raw, target, _ = read_pair(local)
        with rasterio.open(root / "gold_bce" / split / f"{chip}_probability.tif") as raster:
            prediction = raster.read(1) >= 0.5
        valid = target >= 0
        masks = dict(fp=valid & (target == 0) & prediction,
                     fn=valid & (target == 1) & ~prediction,
                     tp=valid & (target == 1) & prediction,
                     tn=valid & (target == 0) & ~prediction)
        medians = {k: [float(np.median(band[mask])) for band in raw] for k,mask in masks.items()}
        same(case["statistics"]["median_db"], medians, chip)
        boundary = binary_dilation(target == 1, iterations=3) ^ binary_erosion(target == 1, iterations=3)
        errors = masks["fp"] | masks["fn"]
        near_boundary = int((boundary & errors).sum())
        same(case["statistics"]["errors_within_three_pixels_of_water_boundary"], near_boundary)
        same(case["statistics"]["boundary_fraction"], near_boundary/int(errors.sum()))
        for key, mask in masks.items():
            assert int(case["score"][key]) == int(mask.sum())
        checked.append(dict(chip=chip, median_db=medians, boundary_fraction=near_boundary/int(errors.sum())))
    save("flood-qualitative", dict(cases=checked, all_saved_statistics_match=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("mri", "flood", "wildlife", "occlusion", "flood-cases"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    {"mri": audit_mri, "flood": audit_flood, "wildlife": audit_wildlife,
     "occlusion": audit_occlusion, "flood-cases": audit_flood_cases}[args.task]()
