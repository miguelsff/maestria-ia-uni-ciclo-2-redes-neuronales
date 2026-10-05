"""Copy selected SAR files into an independent Linux cache and verify MD5."""

import argparse
import base64
import hashlib
import json
import shutil
import time
from pathlib import Path


def _checksum(path):
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return base64.b64encode(digest.digest()).decode()


def main() -> None:
    """Keep originals read-only and write only beneath the dedicated cache."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    source, cache = args.source.resolve(), args.cache.resolve()
    manifest = json.loads((source / "download_manifest.json").read_text(encoding="utf-8"))
    copied, reused, total_bytes = 0, 0, 0
    started = time.monotonic()
    for index, item in enumerate(manifest["objects"], 1):
        original, target = (source / item["name"]).resolve(), (cache / item["name"]).resolve()
        if not original.is_relative_to(source) or not target.is_relative_to(cache):
            raise ValueError("Dataset path escapes its declared root.")
        if target.is_file() and target.stat().st_size == int(item["size"]) and _checksum(target) == item["md5Hash"]:
            reused += 1
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".copy")
            digest = hashlib.md5(usedforsecurity=False)
            with original.open("rb") as reader, temporary.open("wb") as writer:
                for block in iter(lambda: reader.read(4 * 1024 * 1024), b""):
                    writer.write(block)
                    digest.update(block)
            if (
                temporary.stat().st_size != int(item["size"])
                or base64.b64encode(digest.digest()).decode() != item["md5Hash"]
            ):
                raise ValueError(f"Checksum mismatch while caching {item['name']}")
            shutil.copystat(original, temporary)
            temporary.replace(target)
            copied += 1
        total_bytes += target.stat().st_size
        if index % 1000 == 0:
            print(f"Cached and verified {index}/{len(manifest['objects'])} files", flush=True)
    for name in ("download_manifest.json", "download_receipt.json"):
        shutil.copy2(source / name, cache / name)
    result = {
        "files": len(manifest["objects"]),
        "copied": copied,
        "reused": reused,
        "bytes": total_bytes,
        "elapsed_seconds": time.monotonic() - started,
        "verification": "Every cache file matches published GCS size and MD5",
        "originals": "read-only; never moved or modified",
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
