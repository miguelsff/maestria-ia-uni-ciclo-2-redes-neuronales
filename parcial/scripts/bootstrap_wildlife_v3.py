"""Stage immutable parent evidence and reusable local inputs in a fresh output."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main() -> None:
    """Copy metadata and weights, never source photographs or scientific smoke outputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent-version", choices=["v2", "v3", "v4", "v5"], default="v2")
    args = parser.parse_args()
    parent, output = args.parent.resolve(), args.output.resolve()
    if output == parent or output.is_relative_to(parent) or parent.is_relative_to(output):
        raise ValueError("The new output must be independent of the preserved parent.")
    receipt_path = output / "receipts/bootstrap.json"
    if output.exists() and not receipt_path.exists():
        existing = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
        if existing - {"receipts/bootstrap.log", "receipts/build.log"}:
            raise ValueError("Bootstrap requires an empty output directory except its own log.")
    parent_inputs = f"inputs/parent-{args.parent_version}"
    mappings = {
        "wildlife/manifest.json": f"{parent_inputs}/manifest.json",
        "wildlife/label_review.csv": f"{parent_inputs}/label_review.csv",
        "wildlife/label_review_sample.json": f"{parent_inputs}/label_review_sample.json",
        "wildlife/occlusion_review.csv": f"{parent_inputs}/occlusion_review.csv",
        "cache/wildlife-integrity.json": "cache/wildlife-integrity.json",
        "cache/weights/yolov8s.pt": "cache/weights/yolov8s.pt",
        "cache/weights/yolo26n.pt": "cache/weights/yolo26n.pt",
        "report-assets/neurips_2026.sty": "report-assets/neurips_2026.sty",
    }
    for optional in (
        "receipts/human-label-confirmation.json",
        "receipts/architecture-review.json",
        "inputs/user-reference/ave.png",
    ):
        if (parent / optional).is_file():
            mappings[optional] = optional
    hashes = {}
    for source, target in mappings.items():
        path = parent / source
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[source] = expected
        destination = output / target
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
                raise ValueError("Previously staged parent input differs; use another output.")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
    baseline = output / "baseline-manifest.json"
    if baseline.exists() and hashlib.sha256(baseline.read_bytes()).hexdigest() != hashes["wildlife/manifest.json"]:
        raise ValueError("The preserved baseline manifest changed.")
    if not baseline.exists():
        shutil.copyfile(output / parent_inputs / "manifest.json", baseline)
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt["parent_file_sha256"] != hashes:
            raise ValueError("Parent inputs differ from the bootstrap receipt.")
        print(json.dumps(receipt, indent=2))
        return
    receipt = {
        "parent": str(parent),
        "output": str(output),
        "parent_file_sha256": hashes,
        "scientific_training_started": False,
        "human_review_modified": False,
        "backgrounds": "reuse the verified parent cohort as a read-only input",
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
