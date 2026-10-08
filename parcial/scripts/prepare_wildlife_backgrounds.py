"""Download a bounded, reproducible cohort of genuine empty training photographs."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from neural_exam.common.io import file_sha256, read_json, write_json
from neural_exam.wildlife.backgrounds import IMAGE_BASE, METADATA_URL, SPLITS_URL, select_backgrounds
from neural_exam.wildlife.data import PARTITIONS, _find_annotations


def download(url: str, destination: Path) -> dict:
    """Verify response length and advertised MD5 before atomically storing an asset."""
    receipt = destination.with_suffix(destination.suffix + ".receipt.json")
    if receipt.is_file() and destination.is_file():
        previous = read_json(receipt)
        if previous["url"] == url and file_sha256(destination) == previous["sha256"]:
            return previous
        raise ValueError(f"Existing asset differs from its receipt: {destination.name}")
    if destination.is_file():
        # Initially downloaded metadata may lack this script's receipt; verify it against HTTP.
        existing = destination.read_bytes()
    else:
        existing = None
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = response.read()
        expected_length = response.headers.get("Content-Length")
        provider = response.headers.get("x-goog-hash", "")
    if expected_length is not None and len(payload) != int(expected_length):
        raise ValueError("Incomplete background download.")
    md5 = next((part.strip()[4:] for part in provider.split(",") if part.strip().startswith("md5=")), None)
    if md5 and base64.b64encode(hashlib.md5(payload, usedforsecurity=False).digest()).decode() != md5:
        raise ValueError("Provider MD5 mismatch.")
    if existing is not None and existing != payload:
        raise ValueError("Existing background metadata differs from the provider; preserve it and use a new output.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if existing is None:
        temporary = destination.with_suffix(destination.suffix + ".part")
        temporary.write_bytes(payload)
        temporary.replace(destination)
    result = {"url": url, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(), "provider_md5": md5}
    write_json(receipt, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--cameras", type=int, default=8)
    parser.add_argument("--per-camera", type=int, default=125)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.annotations = _find_annotations(args.annotations)
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "caltech_camera_traps.json.zip"
    metadata_receipt = download(METADATA_URL, archive)
    split_receipt = download(SPLITS_URL, root / "CaltechCameraTrapsSplits_v0.json")
    with zipfile.ZipFile(archive) as stream:
        members = [name for name in stream.namelist() if name.endswith(".json")]
        if len(members) != 1:
            raise ValueError("Expected one full-collection metadata JSON.")
        metadata = json.loads(stream.read(members[0]))
    metadata_path = root / "full_metadata.json"
    if metadata_path.exists():
        if read_json(metadata_path) != metadata:
            raise ValueError("Existing full-collection metadata changed.")
    else:
        write_json(metadata_path, metadata)
    documents = {split: read_json(args.annotations / filename) for split, filename in PARTITIONS.items()}
    selection = {"cameras": args.cameras, "per_camera": args.per_camera, "seed": args.seed}
    chosen = select_backgrounds(metadata, read_json(root / "CaltechCameraTrapsSplits_v0.json"), documents, **selection)

    def fetch(row: dict) -> dict:
        image = root / "images" / row["file_name"]
        receipt = download(IMAGE_BASE + row["file_name"], image)
        with Image.open(io.BytesIO(image.read_bytes())) as decoded:
            decoded.load()
            if decoded.size != (int(row["width"]), int(row["height"])) or decoded.getexif().get(274, 1) != 1:
                raise ValueError("Background photograph differs from metadata dimensions or rotation.")
        return {**row, **receipt}

    if not 1 <= args.workers <= 8:
        raise ValueError("Use between one and eight bounded download workers.")
    images = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, row in enumerate(pool.map(fetch, chosen), 1):
            images.append(row)
            if index % 100 == 0 or index == len(chosen):
                print(f"Verified backgrounds: {index}/{len(chosen)}", flush=True)
    result = {
        "schema_version": 1,
        "selection": selection,
        "annotation_sha256": {
            split: file_sha256(args.annotations / filename) for split, filename in PARTITIONS.items()
        },
        "metadata_sha256": {
            "full_metadata.json": file_sha256(metadata_path),
            "CaltechCameraTrapsSplits_v0.json": file_sha256(root / "CaltechCameraTrapsSplits_v0.json"),
        },
        "metadata_download": metadata_receipt,
        "splits_download": split_receipt,
        "images": images,
        "human_review": "pending; download integrity is not visual label approval",
    }
    destination = root / "cohort.json"
    if destination.exists() and read_json(destination) != result:
        raise ValueError("An existing cohort is different; use a new output.")
    write_json(destination, result)
    print(json.dumps({"cohort": str(destination), "images": len(images), "human_review": "pending"}, indent=2))


if __name__ == "__main__":
    main()
