"""Verify selected local files against the published GCS download manifest."""

import argparse
import base64
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def main() -> None:
    """Hash every selected file without modifying source data."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.data_root.resolve()
    manifest_path = root / "download_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures, verified_bytes = [], 0
    for index, item in enumerate(manifest["objects"], 1):
        path = (root / item["name"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Dataset manifest contains a path outside its root.")
        if not path.is_file():
            failures.append({"file": item["name"], "reason": "missing"})
            continue
        digest = hashlib.md5(usedforsecurity=False)
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
        matches = path.stat().st_size == int(item["size"])
        matches &= base64.b64encode(digest.digest()).decode() == item["md5Hash"]
        if not matches:
            failures.append({"file": item["name"], "reason": "size or MD5 mismatch"})
        verified_bytes += path.stat().st_size
        if index % 1000 == 0:
            print(f"Verified {index}/{len(manifest['objects'])} files", flush=True)
    result = {
        "completed_utc": datetime.now(UTC).isoformat(),
        "source": manifest["source"],
        "download_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "selected_files": len(manifest["objects"]),
        "verified_bytes": verified_bytes,
        "counts": manifest["counts"],
        "remote_weak_chips": manifest["remote_weak_chips"],
        "excluded_weak_chips": manifest["excluded_weak_chips"],
        "failures": failures,
        "raster_validation": "Performed separately by neural_exam.flood.data.prepare",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"verified_files": len(manifest["objects"]), "failures": len(failures)}), flush=True)
    if failures:
        raise SystemExit("Integrity audit failed; inspect the output without replacing source files.")


if __name__ == "__main__":
    main()
