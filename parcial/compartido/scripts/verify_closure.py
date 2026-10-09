"""Verify frozen sources and completed accepted results without scientific evaluation."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from closure_runtime import digest, read_json, source_path, verify_frozen

ACCEPTED_SIGNATURE = "6b2983c33abccbc18e293cc3a10ca30a710669c2ba8055a76bb462b3348a13a9"


def verify_portable(delivery: Path) -> dict:
    """Check the portable evidence and genuine recorded execution of its notebook."""
    integrity = read_json(delivery / "data/integrity.json")
    for name, expected in integrity["sha256"].items():
        path = (delivery / "data" / name).resolve()
        if not path.is_relative_to((delivery / "data").resolve()) or digest(path) != expected:
            raise ValueError(f"Portable evidence changed: {name}")
    result = read_json(delivery / "data/results.json")
    if integrity["signature"] != ACCEPTED_SIGNATURE or result["signature"] != ACCEPTED_SIGNATURE:
        raise ValueError("Portable evaluation signature changed.")
    notebook = read_json(delivery / "04_occlusion_582.ipynb")
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    if len(cells) != 4 or any(cell.get("execution_count") is None for cell in cells):
        raise ValueError("Portable notebook lacks complete recorded execution.")
    if any(output.get("output_type") == "error" for cell in cells for output in cell.get("outputs", [])):
        raise ValueError("Portable notebook contains an execution error.")
    return {"signature": ACCEPTED_SIGNATURE, "executed_code_cells": len(cells), "notebook_errors": 0}


def verify(exam: Path, parent: Path, occlusion: Path, delivery: Path, scientific_runtime: bool = False) -> dict:
    """Bind local source bytes, parent artifacts, accepted decisions and existing results."""
    frozen = verify_frozen(exam, parent / "receipts/source-freeze.json")
    sys.path.insert(0, str(source_path(exam, "scripts/evaluate_occlusion_accepted.py").parent.resolve()))
    evaluator = importlib.import_module("evaluate_occlusion_accepted")
    builder = importlib.import_module("build_occlusion_accepted_delivery")
    _, _, _, status, signature = evaluator.inputs(parent, occlusion, scientific=scientific_runtime)
    if signature != ACCEPTED_SIGNATURE:
        raise ValueError(f"Accepted evaluation identity changed: {signature}")
    results, path = builder.completed_result(parent, occlusion)
    portable = verify_portable(delivery)
    if digest(path) != digest(delivery / "data/results.json"):
        raise ValueError("Portable results differ from the completed original evaluation.")
    return {"status": "passed", "frozen_sources_verified": len(frozen), "signature": signature,
            "review_status": status, "results_sha256": digest(path), "portable": portable,
            "replicates_per_level": {level: group["uncertainty"]["replicates"]
                                     for level, group in results["sample_detection"].items()},
            "runtime_versions_checked": scientific_runtime, "bootstrap_recomputed": False,
            "trained_or_ran_inference": False, "full_fresh_scientific_reproduction": "not_verified"}


def main() -> None:
    """Run read-only checks; write a receipt only when explicitly requested."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exam-root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--occlusion", type=Path, required=True)
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--scientific-runtime", action="store_true")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    result = verify(args.exam_root, args.parent, args.occlusion, args.delivery, args.scientific_runtime)
    payload = json.dumps(result, indent=2) + "\n"
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
