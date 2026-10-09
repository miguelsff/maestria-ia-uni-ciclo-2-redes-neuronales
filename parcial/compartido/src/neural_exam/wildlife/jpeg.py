"""Preserve source pixels while canonicalizing whitespace after JPEG end markers."""

from __future__ import annotations

import hashlib
import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from neural_exam.common.io import file_sha256, fingerprint


def canonical_jpeg(source: Path, destination: Path, source_sha256: str) -> dict | None:
    """Copy only a JPEG with trailing whitespace; never re-encode or modify its source."""
    with source.open("rb") as stream:
        header = stream.read(2)
        if header != b"\xff\xd8":
            return None
        stream.seek(-2, 2)
        ending = stream.read()
    if ending == b"\xff\xd9":
        return None
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != source_sha256:
        raise ValueError("JPEG source changed after its integrity audit.")
    cleaned = payload.rstrip(b" \t\r\n")
    if cleaned == payload or not cleaned.endswith(b"\xff\xd9"):
        raise ValueError("Unexpected JPEG trailer or missing end marker; source requires inspection.")
    with Image.open(io.BytesIO(payload)) as original, Image.open(io.BytesIO(cleaned)) as derived:
        original.load()
        derived.load()
        left, right = np.asarray(original), np.asarray(derived)
        if original.mode != derived.mode or not np.array_equal(left, right):
            raise ValueError("Canonical JPEG changed decoded pixels.")
        pixel_hash = hashlib.sha256(left.tobytes()).hexdigest()
    digest = hashlib.sha256(cleaned).hexdigest()
    if destination.is_symlink():
        raise ValueError("Canonical JPEG destination cannot be a symbolic link.")
    if destination.exists():
        if file_sha256(destination) != digest:
            raise ValueError("Existing canonical JPEG differs; preserve it and use a new output.")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(cleaned)
    return {
        "source": str(source),
        "image": str(destination),
        "source_sha256": source_sha256,
        "derived_sha256": digest,
        "decoded_pixels_sha256": pixel_hash,
        "removed_trailing_whitespace_bytes": len(payload) - len(cleaned),
        "decoded_pixels_identical": True,
    }


def prepare_jpeg_sources(records: dict, destination: Path) -> tuple[dict, dict]:
    """Audit eligible inputs and return only the paths that need canonical working copies."""
    sources = {}
    for rows in records.values():
        for row in rows:
            if row["source"] in sources and sources[row["source"]] != row["source_sha256"]:
                raise ValueError("One source photograph has conflicting integrity hashes.")
            sources[row["source"]] = row["source_sha256"]

    def prepare(item):
        source, digest = item
        return canonical_jpeg(Path(source), destination / f"{digest}.jpg", digest)

    copies = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, result in enumerate(pool.map(prepare, sorted(sources.items())), 1):
            if result:
                copies.append(result)
            if index % 5000 == 0 or index == len(sources):
                print(f"JPEG loader compatibility: {index}/{len(sources)}; lossless copies: {len(copies)}", flush=True)
    return (
        {item["source"]: Path(item["image"]) for item in copies},
        {
            "method": "remove whitespace after an existing JPEG end marker; no re-encoding",
            "eligible_sources_checked": len(sources),
            "canonical_copies": len(copies),
            "source_photographs_modified": False,
            "all_copied_pixels_verified_identical": True,
            "copy_registry_identity": fingerprint(copies),
            "copies": copies,
        },
    )
