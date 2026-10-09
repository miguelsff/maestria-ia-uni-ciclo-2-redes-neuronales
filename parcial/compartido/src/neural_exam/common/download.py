"""Explicit, resumable public downloads. No operation occurs on import."""

from __future__ import annotations

import base64
import hashlib
import json
import tarfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from neural_exam.common.io import file_sha256, write_json

CCT_BASE = "https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/"
STYLE_URL = "https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip"


def _inside(root: Path, relative: str) -> Path:
    """Reject archive members and remote keys that escape the intended directory."""
    root = root.resolve()
    target = (root / relative).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"Path escapes download destination: {relative}")
    return target


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return base64.b64encode(digest.digest()).decode()


def download_file(url: str, target: Path, size: int | None = None, md5: str | None = None) -> Path:
    """Resume a partial HTTP response and verify length/checksum before finalizing."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and (size is None or target.stat().st_size == size) and (md5 is None or _md5(target) == md5):
        return target
    partial = target.with_suffix(target.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if size is not None and offset == size and (md5 is None or _md5(partial) == md5):
        partial.replace(target)
        return target
    request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
    with urllib.request.urlopen(request, timeout=120) as response:
        if offset and response.status == 206:
            if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                raise ValueError("Server returned the wrong byte range; partial file retained.")
            mode = "ab"
        else:
            offset, mode = 0, "wb"
        expected = size
        if expected is None and response.headers.get("Content-Length"):
            expected = offset + int(response.headers["Content-Length"])
        advertised_hash = response.headers.get("x-goog-hash", "")
        if md5 is None:
            md5 = next(
                (item.strip()[4:] for item in advertised_hash.split(",") if item.strip().startswith("md5=")), None
            )
        with partial.open(mode) as stream:
            while block := response.read(4 * 1024 * 1024):
                stream.write(block)
    if expected is not None and partial.stat().st_size != expected:
        raise OSError("Incomplete download; partial file retained for resume.")
    if md5 and _md5(partial) != md5:
        raise ValueError("Downloaded checksum mismatch; remove the .part file before retrying.")
    partial.replace(target)
    return target


def safe_extract(archive: Path, root: Path) -> None:
    """Extract only regular files/directories after checking the entire member list."""
    root.mkdir(parents=True, exist_ok=True)
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as stream:
            members = stream.getmembers()
            for member in members:
                _inside(root, member.name)
                if not (member.isfile() or member.isdir()):
                    raise ValueError("Archive links and special files are not accepted.")
            stream.extractall(root, members=members, filter="data")
    else:
        with zipfile.ZipFile(archive) as stream:
            for member in stream.infolist():
                _inside(root, member.filename)
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Archive symlinks are not accepted.")
            stream.extractall(root)


def _gcs_objects(bucket: str, prefix: str) -> list[dict]:
    rows, token = [], None
    while True:
        query = {"prefix": prefix, "fields": "items(name,size,md5Hash),nextPageToken", "maxResults": 1000}
        if token:
            query["pageToken"] = token
        url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o?" + urllib.parse.urlencode(query)
        with urllib.request.urlopen(url, timeout=120) as stream:
            page = json.load(stream)
        rows.extend(page.get("items", []))
        token = page.get("nextPageToken")
        if not token:
            return rows


def download_dataset(settings, dataset: str) -> dict:
    """Download only a named dataset or asset after an explicit CLI invocation."""
    if dataset in {"cct-annotations", "wildlife-weights"}:
        if dataset == "cct-annotations":
            root = getattr(settings, "wildlife_annotations_root", None) or settings.wildlife_root
            targets = [(CCT_BASE + "eccv_18_annotations.tar.gz", root / "archives/eccv_18_annotations.tar.gz")]
        else:
            targets = [
                (
                    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt",
                    settings.weights_root / "yolov8s.pt",
                ),
                (
                    "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt",
                    settings.weights_root / "yolo26n.pt",
                ),
            ]
        receipts = []
        for url, target in targets:
            # HEAD prevents an existing, truncated file from being accepted without verification.
            with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as response:
                length = response.headers.get("Content-Length")
                advertised_hash = response.headers.get("x-goog-hash", "")
                checksum = next(
                    (item.strip()[4:] for item in advertised_hash.split(",") if item.strip().startswith("md5=")), None
                )
            download_file(url, target, int(length) if length else None, checksum)
            receipts.append(
                {
                    "url": url,
                    "file": str(target),
                    "bytes": target.stat().st_size,
                    "sha256": file_sha256(target),
                    "provider_md5": checksum,
                }
            )
        if dataset == "cct-annotations":
            safe_extract(targets[0][1], root)
        result = {"dataset": dataset, "files": receipts}
        write_json(settings.output_root / "receipts" / f"{dataset}.json", result)
        return result
    if dataset == "flood":
        prefixes = [
            "v1.1/splits/flood_handlabeled/",
            "v1.1/data/flood_events/HandLabeled/S1Hand/",
            "v1.1/data/flood_events/HandLabeled/LabelHand/",
            "v1.1/data/flood_events/WeaklyLabeled/S1Weak/",
            "v1.1/data/flood_events/WeaklyLabeled/S1OtsuLabelWeak/",
        ]
        objects = [
            obj
            for prefix in prefixes
            for obj in _gcs_objects("sen1floods11", prefix)
            if obj["name"].endswith((".tif", ".csv"))
        ]
        for index, obj in enumerate(objects):
            target = _inside(settings.flood_root, obj["name"])
            url = "https://storage.googleapis.com/sen1floods11/" + urllib.parse.quote(obj["name"], safe="/")
            download_file(url, target, int(obj["size"]), obj.get("md5Hash"))
            print(f"Sen1Floods11: {index + 1}/{len(objects)}", flush=True)
        result = {"dataset": dataset, "files": len(objects), "source": "gs://sen1floods11/v1.1"}
        write_json(settings.flood_root / "download_receipt.json", result)
        return result
    if dataset == "cct":
        archives = settings.wildlife_root / "archives"
        for filename in ("eccv_18_all_images_sm.tar.gz", "eccv_18_annotations.tar.gz"):
            archive = download_file(CCT_BASE + filename, archives / filename)
            safe_extract(archive, settings.wildlife_root)
        result = {"dataset": dataset, "source": CCT_BASE, "version": "official ECCV CCT20 benchmark"}
        write_json(settings.wildlife_root / "download_receipt.json", result)
        return result
    if dataset == "weights":
        weights = settings.weights_root
        download_file(
            "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt", weights / "yolov8s.pt"
        )
        # Ultralytics 8.4.21 uses this auxiliary model for its built-in AMP check.
        download_file(
            "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt", weights / "yolo26n.pt"
        )
        download_file("https://download.pytorch.org/models/resnet50-11ad3fa6.pth", weights / "resnet50-11ad3fa6.pth")
        return {"dataset": dataset, "path": str(weights)}
    if dataset == "style":
        assets = settings.output_root / "report-assets"
        archive = download_file(STYLE_URL, assets / "Formatting_Instructions_For_NeurIPS_2026.zip")
        safe_extract(archive, assets)
        result = {
            "dataset": dataset,
            "path": str(assets),
            "source": STYLE_URL,
            "sha256": file_sha256(archive),
            "bytes": archive.stat().st_size,
        }
        write_json(settings.output_root / "receipts/style.json", result)
        return result
    raise ValueError(f"Unsupported download: {dataset}")
