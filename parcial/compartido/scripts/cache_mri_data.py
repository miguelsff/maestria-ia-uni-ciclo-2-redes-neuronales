"""Stream selected MRI series from Windows into an isolated Docker cache."""

import hashlib
import json
import os
import subprocess
import tarfile
from pathlib import Path


def main() -> None:
    """Copy exact bytes, inventory source directories and persist original hashes."""
    source = Path(os.environ.get("RSNA_HOST_DIR", "E:/Kaggle/rsna-knee-abnormality-detection"))
    output = Path(os.environ.get("MRI_OUTPUT_DIR", "E:/Kaggle/exam-output-mri/stratified-v2")) / "audit"
    selected = json.loads((output / "selected_series.json").read_text(encoding="utf-8"))
    inventory = {
        study.name: sorted(series.name for series in study.iterdir() if series.is_dir())
        for study in (source / "train_series").iterdir()
        if study.is_dir()
    }
    (output / "source_inventory.json").write_text(json.dumps(inventory), encoding="utf-8")
    files = [source / name for name in ("train.csv", "train_series.csv")]
    for row in selected:
        files.extend(sorted((source / "train_series" / row["study_id"] / row["series_id"]).glob("*.dcm")))
    process = subprocess.Popen(
        ["docker", "exec", "-i", "mri-cache-writer", "tar", "xf", "-", "-C", "/cache"], stdin=subprocess.PIPE
    )
    hashes = []
    try:
        with tarfile.open(fileobj=process.stdin, mode="w|") as archive:
            for index, path in enumerate(files, 1):
                relative = path.relative_to(source).as_posix()
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                        digest.update(block)
                hashes.append({"file": relative, "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
                archive.add(path, arcname=relative, recursive=False)
                if index % 1000 == 0:
                    print(f"Copied and hashed {index}/{len(files)} original files", flush=True)
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("Docker MRI cache transfer failed.")
    except BaseException:
        process.kill()
        raise
    (output / "cache_source_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    print(f"MRI cache copy completed: {len(files)} files; originals unchanged.", flush=True)


if __name__ == "__main__":
    main()
