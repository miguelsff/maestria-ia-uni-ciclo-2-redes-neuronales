"""Evaluate the user-accepted cohort with frozen V6 predictions and sequence intervals."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from occlusion_accepted_review import accepted_inputs, describe_accepted_reviews
from wildlife_occlusion_sample import LEVELS, digest, read_json, verify_parent, write_json


def inputs(parent: Path, output: Path, scientific: bool = False) -> tuple[dict, list, dict, dict, str]:
    """Validate authorization, actual human grades, frozen files and environment identity."""
    protocol, rows, acceptance, status = accepted_inputs(output)
    verify_parent(parent, protocol["parent_sha256"])
    if scientific:
        versions = read_json(parent / "receipts/source-freeze.json")["versions"]
        if any(metadata.version(name) != version for name, version in versions.items()):
            raise ValueError("Evaluation runtime differs from frozen V6.")
    signature = hashlib.sha256(
        json.dumps(
            {
                "protocol": digest(output / "protocol.json"),
                "acceptance": digest(output / "review_acceptance.json"),
                "reviews": digest(output / "occlusion_review_sample_600.csv"),
                "evaluator": digest(Path(__file__)),
                "accepted_helper": digest(Path(__file__).with_name("occlusion_accepted_review.py")),
                "fixed_helper": digest(Path(__file__).with_name("wildlife_occlusion_sample.py")),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return protocol, rows, acceptance, status, signature


def compute(records: list, predictions: list, names: list, rows: list, level: str, replicates: int, seed: int) -> dict:
    """Keep every trans_test frame; exclude ungraded boxes through frozen crowd rules."""
    from neural_exam.wildlife.metrics import coco_evaluate
    from neural_exam.wildlife.uncertainty import bootstrap_metrics

    kwargs = {"occlusions": {row["annotation_id"]: row["level"] for row in rows}, "occlusion_group": level}
    return {
        "metrics": coco_evaluate(records, predictions, names, **kwargs),
        "uncertainty": bootstrap_metrics(records, predictions, names, replicates, seed, **kwargs),
    }


def worker(parent: Path, output: Path, level: str, expected_signature: str) -> None:
    """Run one isolated CPU worker and publish only if human inputs remain unchanged."""
    protocol, rows, _, status, signature = inputs(parent, output, scientific=True)
    if signature != expected_signature or status["missing_support"][level]:
        raise ValueError("Review identity changed or level has insufficient support.")
    manifest = read_json(parent / "wildlife/manifest.json")
    predictions = read_json(parent / "wildlife/trans_test/predictions.json")
    print(
        f"Starting {level}: {status['counts'][level]} accepted cases; {protocol['bootstrap_replicates']} sequence replicates",
        flush=True,
    )
    result = compute(
        manifest["records"]["trans_test"],
        predictions,
        manifest["names"],
        rows,
        level,
        protocol["bootstrap_replicates"],
        protocol["seed"],
    )
    if result["metrics"]["eligible_objects"] != status["counts"][level]:
        raise ValueError("COCO support differs from the accepted human cohort.")
    if inputs(parent, output)[4] != signature:
        raise ValueError("Human inputs changed during evaluation; no result published.")
    destination = output / "accepted_evaluation" / signature[:16]
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / f"{level}.json", {"signature": signature, "level": level, **result})
    print(f"Completed {level}", flush=True)


def evaluate(parent: Path, output: Path) -> dict:
    """Reuse matching completed levels and never retrain or rerun model inference."""
    protocol, _, acceptance, status, signature = inputs(parent, output, scientific=True)
    destination = output / "accepted_evaluation" / signature[:16]
    destination.mkdir(parents=True, exist_ok=True)
    descriptive = describe_accepted_reviews(output)
    write_json(destination / "descriptive_occlusion.json", descriptive)
    groups = {}
    for level in LEVELS[:3]:
        if status["missing_support"][level]:
            groups[level] = {"status": "insufficient_support", "accepted_objects": status["counts"].get(level, 0)}
            continue
        path = destination / f"{level}.json"
        if not path.exists() or read_json(path)["signature"] != signature:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__)),
                    "worker",
                    "--parent",
                    str(parent),
                    "--output",
                    str(output),
                    "--level",
                    level,
                    "--signature",
                    signature,
                ],
                check=True,
            )
        groups[level] = read_json(path)
    if inputs(parent, output)[4] != signature:
        raise ValueError("Human review changed during evaluation; run remains incomplete.")
    result = {
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "status": "evaluation_complete_with_documented_review_deviations",
        "signature": signature,
        "review_status": status,
        "review_acceptance": acceptance,
        "descriptive_occlusion": descriptive,
        "sample_detection": groups,
        "checkpoint_sha256": protocol["parent_sha256"]["wildlife/yolov8s/weights/best.pt"],
        "trained_or_ran_inference": False,
        "detection_scope": "Unweighted accepted-cohort AP using frozen COCO rules over all trans_test frames; "
        "unreviewed, excluded, ambiguous or other-level boxes are ignored crowd regions for each low/medium/high group.",
        "limitations": "Detection intervals are conditional 95% sequence-bootstrap intervals with 200 replicates "
        "and the fixed checkpoint. They do not measure annotation uncertainty, variation across training seeds "
        "or future-camera performance. Original complete-600 proportion intervals are not reused after exclusions. "
        "Missing human notes remain missing; the case-121 disagreement is preserved in archived sources.",
    }
    write_json(destination / "results.json", result)
    write_json(
        output / "accepted_evaluation_latest.json",
        {"signature": signature, "results": str(destination / "results.json"), "status": result["status"]},
    )
    return {"status": result["status"], "results": str(destination / "results.json")}


def main() -> None:
    """Check or execute only this explicitly accepted post-review cohort."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "run", "worker"))
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--level", choices=LEVELS[:3])
    parser.add_argument("--signature")
    args = parser.parse_args()
    if args.action == "check":
        result = inputs(args.parent, args.output)[3]
    elif args.action == "worker":
        if args.level is None or args.signature is None:
            parser.error("worker requires a level and signature")
        worker(args.parent, args.output, args.level, args.signature)
        return
    else:
        result = evaluate(args.parent, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
