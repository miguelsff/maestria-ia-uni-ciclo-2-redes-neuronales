"""Isolate bootstrap groups while preserving the frozen evaluation algorithm."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import numpy as np

from neural_exam.common.io import file_sha256, fingerprint, read_csv, read_json, write_json
from neural_exam.common.locking import execution_lock
from neural_exam.config import PROJECT_ROOT, Settings, load_settings
from neural_exam.wildlife.uncertainty import bootstrap_metrics

SIZES = {
    "small_lt_1pct": (0, float(np.nextafter(10000.0, 0))),
    "medium_1_to_5pct": (10000, 50000),
    "large_gt_5pct": (float(np.nextafter(50000.0, np.inf)), 1e10),
}


def verify_inputs(settings: Settings) -> tuple[dict, dict, str]:
    """Require unchanged experimental sources, weights and prediction provenance."""
    root = settings.output_root
    freeze = read_json(root / "receipts/source-freeze.json")
    if any(file_sha256(PROJECT_ROOT / name) != digest for name, digest in freeze["sha256"].items()):
        raise ValueError("Frozen experimental sources changed; recovery cannot proceed.")
    if any(metadata.version(name) != version for name, version in freeze["versions"].items()):
        raise ValueError("Recovery runtime differs from the frozen runtime.")
    package = PROJECT_ROOT / "src/neural_exam"
    paths = [*(package / "wildlife").glob("*.py"), *(package / "common").glob("*.py")]
    paths += [package / "config.py", package / "cli.py"]
    sources = {path.relative_to(package).as_posix(): file_sha256(path) for path in paths}
    model_run = read_json(root / "wildlife/yolov8s/run.json")
    if not model_run["scientific_run"] or model_run["source_sha256"] != sources:
        raise ValueError("Recovery sources differ from the actual scientific run.")
    complete = read_json(root / "wildlife/yolov8s/training_complete.json")
    checkpoint = file_sha256(root / "wildlife/yolov8s/weights/best.pt")
    if checkpoint != complete["sha256"]:
        raise ValueError("The selected checkpoint changed.")
    manifest = read_json(root / "wildlife/manifest.json")
    if fingerprint({key: value for key, value in manifest.items() if key != "identity"}) != manifest["identity"]:
        raise ValueError("The preserved manifest is invalid.")
    for split in ("cis_test", "trans_test"):
        protocol = read_json(root / "wildlife" / split / "metrics.json")["protocol"]
        if (
            protocol["checkpoint_sha256"] != checkpoint
            or protocol["conf"] != 0.001
            or protocol["nms_iou"] != settings.wildlife["nms_iou"]
            or protocol["max_detections"] != 300
        ):
            raise ValueError("Saved predictions belong to another evaluation protocol.")
    signature = fingerprint(
        {
            "checkpoint": checkpoint,
            "manifest": manifest["identity"],
            "review_sha256": file_sha256(root / "wildlife/occlusion_review.csv"),
            "replicates": settings.wildlife["uncertainty_replicates"],
            "seed": settings.training["seed"],
            "source_sha256": sources,
        }
    )
    identity = {
        "uncertainty_signature": signature,
        "recovery_script_sha256": file_sha256(Path(__file__)),
        "prediction_sha256": {
            split: file_sha256(root / "wildlife" / split / "predictions.json") for split in ("cis_test", "trans_test")
        },
    }
    return manifest, identity, signature


def bootstrap_jobs(robustness: dict) -> list[str]:
    """Use exactly the groups scheduled by the frozen assessment function."""
    jobs = [f"{split}--{group}" for split in ("cis_test", "trans_test") for group in ("global", *SIZES)]
    if robustness["occlusion_status"] == "reviewed":
        jobs += [f"trans_test--occlusion_{level}" for level in ("low", "medium", "high")]
    return jobs


def evaluate_group(
    key: str, manifest: dict, predictions: list[dict], robustness: dict, reviews: list[dict], replicates: int, seed: int
) -> dict:
    """Delegate all sampling and metrics to the unchanged frozen implementation."""
    if key not in bootstrap_jobs(robustness):
        raise ValueError("Bootstrap group does not belong to the declared protocol.")
    split, group = key.split("--", 1)
    kwargs = {}
    if group in SIZES:
        kwargs["area_range"] = SIZES[group]
    elif group.startswith("occlusion_"):
        kwargs.update(
            occlusions={row["annotation_id"]: row["level"] for row in reviews},
            occlusion_group=group.removeprefix("occlusion_"),
        )
    return bootstrap_metrics(manifest["records"][split], predictions, manifest["names"], replicates, seed, **kwargs)


def assemble_results(robustness: dict, groups: dict) -> dict:
    """Preserve the original result schema, including unfinished human occlusion."""
    if set(groups) != set(bootstrap_jobs(robustness)):
        raise ValueError("Bootstrap coverage is incomplete.")
    result = {
        "global": {},
        "size": {},
        "occlusion": {},
        "limitations": "Conditional on sampled cameras and one training seed; sequence resampling does not measure new-region uncertainty.",
        "occlusion_status": robustness["occlusion_status"],
    }
    for split in ("cis_test", "trans_test"):
        result["global"][split] = groups[f"{split}--global"]
        result["size"][split] = {group: groups[f"{split}--{group}"] for group in SIZES}
    if robustness["occlusion_status"] == "reviewed":
        result["occlusion"] = {level: groups[f"trans_test--occlusion_{level}"] for level in ("low", "medium", "high")}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker")
    parser.add_argument("--finish", action="store_true")
    args = parser.parse_args()
    settings = load_settings()
    root = settings.output_root
    output = root / "wildlife"
    manifest, identity, signature = verify_inputs(settings)
    cache = output / "bootstrap_cache" / fingerprint(identity)
    robustness = read_json(output / "robustness.json")
    reviews = read_csv(output / "occlusion_review.csv")
    if args.worker:
        if args.worker not in bootstrap_jobs(robustness):
            raise ValueError("Bootstrap group does not belong to the declared protocol.")
        split = args.worker.split("--", 1)[0]
        predictions = read_json(output / split / "predictions.json")
        metrics = evaluate_group(
            args.worker,
            manifest,
            predictions,
            robustness,
            reviews,
            settings.wildlife["uncertainty_replicates"],
            settings.training["seed"],
        )
        write_json(cache / f"{args.worker}.json", {"identity": identity, "group": args.worker, "metrics": metrics})
        return
    with execution_lock(root):
        snapshot = root / "receipts/recovery-source/recover_wildlife_evaluation.py"
        if snapshot.exists() and file_sha256(snapshot) != identity["recovery_script_sha256"]:
            raise ValueError("Preserved recovery source differs; use another recovery snapshot.")
        if not snapshot.exists():
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(__file__, snapshot)
        groups = {}
        for key in bootstrap_jobs(robustness):
            path = cache / f"{key}.json"
            if not path.exists():
                print(f"Isolated bootstrap group: {key}", flush=True)
                subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", key], check=True)
            saved = read_json(path)
            if saved["identity"] != identity or saved["group"] != key:
                raise ValueError("Saved bootstrap group differs from current immutable inputs.")
            groups[key] = saved["metrics"]
        _, current_identity, current_signature = verify_inputs(settings)
        if current_identity != identity or current_signature != signature:
            raise ValueError("Evaluation inputs changed while isolated groups ran.")
        write_json(output / "uncertainty.json", {"signature": signature, **assemble_results(robustness, groups)})
        write_json(
            root / "receipts/isolated-bootstrap-recovery.json",
            {
                "identity": identity,
                "status": "bootstrap_complete",
                "groups": list(groups),
                "protocol_changed": False,
                "training_repeated": False,
                "method": "one fresh subprocess per group; original bootstrap function, seed and replicates",
            },
        )
        if args.finish:
            gpu = (
                subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.free", "--format=csv,noheader,nounits"],
                    text=True,
                )
                .strip()
                .splitlines()
            )
            if len(gpu) != 1:
                raise RuntimeError("Expected one available GPU for the remaining inference and CAM work.")
            utilization, free = [int(value.strip()) for value in gpu[0].split(",")]
            if utilization >= 20 or free < 8192:
                raise RuntimeError("Another GPU job is active; preserve completed bootstrap groups and retry later.")
            from neural_exam.reporting import prepare_report_assets
            from neural_exam.wildlife.pipeline import run

            run(settings, resume=True, manifest=manifest)
            prepare_report_assets(settings, task="wildlife")
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "jupyter",
                    "nbconvert",
                    "--execute",
                    "--to",
                    "notebook",
                    str(PROJECT_ROOT / "notebooks/03_wildlife.ipynb"),
                    "--output",
                    "03_wildlife",
                    "--output-dir",
                    str(root / "notebooks"),
                    "--ExecutePreprocessor.timeout=180",
                ],
                check=True,
            )
            print("Recovered evaluation, report assets and executed notebook complete.", flush=True)


if __name__ == "__main__":
    main()
