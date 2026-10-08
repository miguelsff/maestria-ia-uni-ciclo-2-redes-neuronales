"""Evaluate genuine human sample labels using saved V6 predictions; never train."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from wildlife_occlusion_sample import (
    LEVELS,
    digest,
    read_json,
    read_protocol,
    read_reviews,
    review_status,
    verify_parent,
    weighted_proportions,
    write_json,
)


def inputs(parent: Path, output: Path, scientific: bool = False) -> tuple[dict, list, dict, str]:
    protocol = read_protocol(output)
    verify_parent(parent, protocol["parent_sha256"])
    rows = read_reviews(output)
    status = review_status(rows, protocol)
    if scientific:
        versions = read_json(parent / "receipts/source-freeze.json")["versions"]
        if any(metadata.version(name) != version for name, version in versions.items()):
            raise ValueError("Evaluation runtime differs from the frozen V6 runtime.")
        if status["complete"] != 600:
            raise ValueError("Human review is incomplete; no sample metrics were computed.")
    signature = hashlib.sha256(
        json.dumps(
            {
                "protocol": digest(output / "protocol.json"),
                "reviews": digest(output / "occlusion_review_sample_600.csv"),
                "evaluator": digest(Path(__file__)),
                "sample_helper": digest(Path(__file__).with_name("wildlife_occlusion_sample.py")),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return protocol, rows, status, signature


def worker(parent: Path, output: Path, level: str, expected_signature: str) -> None:
    protocol, rows, status, signature = inputs(parent, output, scientific=True)
    if signature != expected_signature or status["missing_support"][level]:
        raise ValueError("Review identity changed or level has insufficient human support.")
    from neural_exam.wildlife.metrics import coco_evaluate
    from neural_exam.wildlife.uncertainty import bootstrap_metrics

    manifest = read_json(parent / "wildlife/manifest.json")
    predictions = read_json(parent / "wildlife/trans_test/predictions.json")
    records = manifest["records"]["trans_test"]
    kwargs = {"occlusions": {row["annotation_id"]: row["level"] for row in rows}, "occlusion_group": level}
    result = {
        "signature": signature,
        "level": level,
        "metrics": coco_evaluate(records, predictions, manifest["names"], **kwargs),
        "uncertainty": bootstrap_metrics(
            records, predictions, manifest["names"], protocol["bootstrap_replicates"], protocol["seed"], **kwargs
        ),
    }
    # Refuse to publish a calculation if human evidence changed during the job.
    if inputs(parent, output)[3] != signature:
        raise ValueError("Human review changed during evaluation; discard the worker result.")
    destination = output / "evaluation" / signature[:16]
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / f"{level}.json", result)


def evaluate(parent: Path, output: Path) -> dict:
    protocol, rows, status, signature = inputs(parent, output, scientific=True)
    destination = output / "evaluation" / signature[:16]
    destination.mkdir(parents=True, exist_ok=True)
    proportions = weighted_proportions(rows, protocol)
    write_json(destination / "weighted_occlusion_proportions.json", proportions)
    groups = {}
    for level in LEVELS[:3]:
        if status["missing_support"][level]:
            groups[level] = {"status": "insufficient_support", "reviewed_objects": status["counts"].get(level, 0)}
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
    if inputs(parent, output)[3] != signature:
        raise ValueError("Human review changed during evaluation; preserve this run as incomplete.")
    result = {
        "at_utc": datetime.now(UTC).isoformat(),
        "status": status["status"],
        "signature": signature,
        "review_status": status,
        "weighted_occlusion_proportions": proportions,
        "sample_detection": groups,
        "checkpoint_sha256": protocol["parent_sha256"]["wildlife/yolov8s/weights/best.pt"],
        "trained_or_ran_inference": False,
        "detection_scope": "Unweighted sample AP using frozen COCO rules over all trans_test frames; "
        "unreviewed or other-level ground-truth boxes are ignored crowd regions.",
        "limitations": "Proportion intervals follow the sampling design. Detection intervals are conditional "
        "sequence-bootstrap intervals, not weighted population AP, annotation-design uncertainty "
        "or future-camera performance. Any targeted supplement is a separate cohort.",
    }
    write_json(destination / "results.json", result)
    write_json(
        output / "evaluation_latest.json",
        {"signature": signature, "results": str(destination / "results.json"), "status": status["status"]},
    )
    return {"status": status["status"], "results": str(destination / "results.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "run", "worker"))
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--level", choices=LEVELS[:3])
    parser.add_argument("--signature")
    args = parser.parse_args()
    if args.action == "check":
        result = inputs(args.parent, args.output)[2]
    elif args.action == "worker":
        if args.level is None or args.signature is None:
            parser.error("worker requires a level and review signature")
        worker(args.parent, args.output, args.level, args.signature)
        return
    else:
        result = evaluate(args.parent, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
