"""Prepare and serve a separate probability sample without changing frozen V6."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import secrets
import shutil
import sys
import threading
from collections import Counter, defaultdict
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
TEMPLATE = PROJECT / "reviews/occlusion_sample.html"
HUMAN_FIELDS = ("level", "visual_evidence", "reviewer")
FIELDS = (
    "annotation_id",
    "image_id",
    "species",
    "camera_id",
    "sequence_id",
    "size",
    "gallery",
    "stratum_population",
    "stratum_sample",
    "sample_weight",
    *HUMAN_FIELDS,
)
LEVELS = ("low", "medium", "high", "ambiguous")
PRESERVED = (
    "wildlife/manifest.json",
    "wildlife/occlusion_review.csv",
    "wildlife/label_review.csv",
    "wildlife/yolov8s/weights/best.pt",
    "wildlife/yolov8s/training_complete.json",
    "wildlife/trans_test/predictions.json",
    "wildlife/trans_test/metrics.json",
    "wildlife/robustness.json",
    "wildlife/uncertainty.json",
    "receipts/source-freeze.json",
    "notebooks/03_wildlife.ipynb",
    "reports/wildlife_draft.pdf",
)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_reviews(output: Path) -> list[dict]:
    with (output / "occlusion_review_sample_600.csv").open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_protocol(output: Path) -> dict:
    if digest(output / "protocol.json") != (output / "protocol.sha256").read_text().strip():
        raise ValueError("The fixed sample receipt changed.")
    protocol = read_json(output / "protocol.json")
    if digest(output / "sampling_analysis.json") != protocol["analysis_sha256"]:
        raise ValueError("The accepted sampling analysis changed.")
    return protocol


def write_reviews(output: Path, rows: list[dict]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    path = output / "occlusion_review_sample_600.csv"
    temporary = path.with_suffix(".csv.tmp")
    temporary.write_text(stream.getvalue(), encoding="utf-8-sig", newline="")
    os.replace(temporary, path)


def verify_parent(parent: Path, snapshot: dict | None = None) -> dict:
    """Verify actual frozen files, checkpoint and saved prediction protocol."""
    freeze = read_json(parent / "receipts/source-freeze.json")
    if any(digest(PROJECT / name) != expected for name, expected in freeze["sha256"].items()):
        raise ValueError("Frozen V6 experimental sources changed.")
    current = {name: digest(parent / name) for name in PRESERVED}
    if snapshot is not None and current != snapshot:
        raise ValueError("The preserved parent artifacts changed.")
    checkpoint = current["wildlife/yolov8s/weights/best.pt"]
    complete = read_json(parent / "wildlife/yolov8s/training_complete.json")
    protocol = read_json(parent / "wildlife/trans_test/metrics.json")["protocol"]
    if complete["sha256"] != checkpoint or protocol["checkpoint_sha256"] != checkpoint:
        raise ValueError("Saved predictions do not match the completed checkpoint.")
    if protocol["conf"] != 0.001 or protocol["nms_iou"] != 0.7 or protocol["max_detections"] != 300:
        raise ValueError("Saved prediction parameters differ from V6.")
    return current


def select_objects(manifest: dict, analysis: dict) -> list[dict]:
    """Draw without replacement within each of the declared 209 strata."""
    grouped = defaultdict(list)
    for record in manifest["records"]["trans_test"]:
        for box in record["boxes"]:
            area = box["box"][2] * box["box"][3] * 1e6
            size = "small" if area < 10000 else "medium" if area <= 50000 else "large"
            species = manifest["names"][box["class_id"]]
            grouped[(record["camera_id"], species, size)].append((record, box))
    quotas = analysis["proposed_allocation"]
    if len(quotas) != len(grouped) or sum(row["sample"] for row in quotas) != 600:
        raise ValueError("The accepted 600-case allocation is incomplete.")
    generator = random.Random(analysis["assumptions"]["planned_random_seed"])
    selected, visited = [], set()
    for quota in sorted(quotas, key=lambda row: (row["camera_id"], row["species"], row["size"])):
        key = (quota["camera_id"], quota["species"], quota["size"])
        if key in visited or key not in grouped:
            raise ValueError("Duplicate or unknown stratum.")
        visited.add(key)
        candidates = sorted(grouped[key], key=lambda pair: pair[1]["annotation_id"])
        count, number = quota["population"], quota["sample"]
        if count != len(candidates) or not min(2, count) <= number <= count:
            raise ValueError("Stratum population or sample quota changed.")
        for record, box in generator.sample(candidates, number):
            selected.append(
                {
                    "annotation_id": box["annotation_id"],
                    "image_id": record["image_id"],
                    "species": key[1],
                    "camera_id": key[0],
                    "sequence_id": record["sequence_id"],
                    "size": key[2],
                    "gallery": f"images/{record['image_id']}.jpg",
                    "stratum_population": str(count),
                    "stratum_sample": str(number),
                    "sample_weight": format(count / number, ".17g"),
                    "normalized_box": box["box"],
                    "source_filename": Path(record["source"]).name,
                    "source_sha256": record["source_sha256"],
                }
            )
    generator.shuffle(selected)
    if len({row["annotation_id"] for row in selected}) != 600:
        raise ValueError("Sample annotation IDs are not unique.")
    return selected


def validate_reviews(rows: list[dict], protocol: dict) -> None:
    reference = {row["annotation_id"]: row for row in protocol["selected"]}
    if len(rows) != 600 or len({row["annotation_id"] for row in rows}) != 600:
        raise ValueError("Review CSV must contain exactly the original 600 IDs.")
    for row in rows:
        original = reference.get(row["annotation_id"])
        if original is None or any(row.get(key) != original[key] for key in FIELDS if key not in HUMAN_FIELDS):
            raise ValueError("Sample identity, weights or image metadata changed.")
        if row.get("level") not in ("", *LEVELS):
            raise ValueError("Unknown occlusion level.")


def is_complete(row: dict) -> bool:
    return bool(row["level"] and row["visual_evidence"].strip() and row["reviewer"].strip())


def review_status(rows: list[dict], protocol: dict) -> dict:
    validate_reviews(rows, protocol)
    counts = Counter(row["level"] for row in rows if is_complete(row))
    completed = sum(counts.values())
    missing_support = {level: max(0, 30 - counts[level]) for level in LEVELS[:3]}
    status = (
        "human_review_pending"
        if completed < 600
        else ("supplement_required" if any(missing_support.values()) else "reviewed")
    )
    return {
        "status": status,
        "total": 600,
        "complete": completed,
        "pending": 600 - completed,
        "counts": dict(counts),
        "missing_support": missing_support,
    }


def weighted_proportions(rows: list[dict], protocol: dict) -> dict:
    """Estimate finite-population label proportions using the accepted design."""
    if review_status(rows, protocol)["complete"] != 600:
        raise ValueError("All 600 human levels, evidence and reviewers are required.")
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["camera_id"], row["species"], row["size"])].append(row)
    total = protocol["population_objects"]
    variance_bound = sum(
        (int(group[0]["stratum_population"]) / total) ** 2
        * (int(group[0]["stratum_population"]) - len(group))
        / (4 * len(group) * (int(group[0]["stratum_population"]) - 1))
        for group in grouped.values()
        if int(group[0]["stratum_population"]) > 1
    )
    margin_bound = 1.959963984540054 * math.sqrt(variance_bound)
    estimates = {}
    for level in LEVELS:
        estimate, variance = 0.0, 0.0
        for group in grouped.values():
            count, number = int(group[0]["stratum_population"]), len(group)
            proportion = sum(row["level"] == level for row in group) / number
            estimate += count / total * proportion
            if number < count:
                variance += (count / total) ** 2 * (1 - number / count) * proportion * (1 - proportion) / (number - 1)
        estimates[level] = {
            "proportion": estimate,
            "lower": max(0, estimate - margin_bound),
            "upper": min(1, estimate + margin_bound),
            "estimated_standard_error": math.sqrt(variance),
        }
    return {
        "method": "design-weighted proportions; marginal approximate normal 95% intervals using conservative binary variance bound",
        "population_objects": total,
        "sample_objects": 600,
        "levels": estimates,
        "planning_margin_bound": margin_bound,
        "limitation": "These intervals do not apply to detection mAP or future cameras.",
    }


def prepare(parent: Path, output: Path, images: Path, analysis_path: Path) -> dict:
    parent, output = parent.resolve(), output.resolve()
    if output == parent or output.is_relative_to(parent):
        raise ValueError("The sample must use a separate output from frozen V6.")
    analysis = read_json(analysis_path)
    snapshot = verify_parent(parent)
    for name, expected in analysis["source_sha256"].items():
        if digest(parent / "wildlife" / name) != expected:
            raise ValueError("The analyzed population changed.")
    manifest = read_json(parent / "wildlife/manifest.json")
    if manifest["identity"] != analysis["population"]["manifest_identity"]:
        raise ValueError("Analysis and manifest identity differ.")
    if (output / "protocol.json").exists():
        protocol = read_protocol(output)
        verify_parent(parent, protocol["parent_sha256"])
        if protocol["analysis_sha256"] != digest(analysis_path):
            raise ValueError("Existing sample uses a different accepted allocation.")
        return review_status(read_reviews(output), protocol)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refusing to overwrite a nonempty sample directory.")
    selected = select_objects(manifest, analysis)
    output.mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir()
    for row in selected:
        source, target = images / row["source_filename"], output / row["gallery"]
        if not target.exists():
            if digest(source) != row["source_sha256"]:
                raise ValueError("Selected original image differs from the manifest.")
            shutil.copyfile(source, target)
        if digest(target) != row["source_sha256"]:
            raise ValueError("Preview is not an exact copy of the original photo.")
    protocol = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol_name": "wildlife-v6-occlusion-stratified-600",
        "population_partition": "trans_test",
        "population_objects": analysis["population"]["objects"],
        "manifest_identity": manifest["identity"],
        "analysis_sha256": digest(analysis_path),
        "parent_sha256": snapshot,
        "frozen_experimental_sha256": read_json(parent / "receipts/source-freeze.json")["sha256"],
        "sample_size": 600,
        "seed": analysis["assumptions"]["planned_random_seed"],
        "generator": "random.Random.sample; independent SRS without replacement within strata",
        "generator_python_version": sys.version.split()[0],
        "strata": ["camera_id", "species", "size"],
        "minimum_detection_support_per_level": 30,
        "bootstrap_replicates": 200,
        "selection_used_predictions": False,
        "selected": selected,
        "limitations": analysis["limits"],
        "human_approval_created": False,
    }
    rows = [{key: row[key] if key not in HUMAN_FIELDS else "" for key in FIELDS} for row in selected]
    write_reviews(output, rows)
    write_json(output / "protocol.json", protocol)
    (output / "protocol.sha256").write_text(digest(output / "protocol.json") + "\n", encoding="utf-8")
    shutil.copyfile(analysis_path, output / "sampling_analysis.json")
    shutil.copyfile(TEMPLATE, output / "occlusion_review.html")
    write_json(output / "review_status.json", review_status(rows, protocol))
    verify_parent(parent, snapshot)
    return review_status(rows, protocol)


class ReviewServer(ThreadingHTTPServer):
    """Local-only persistence for genuine browser-entered human review fields."""

    def __init__(self, output: Path, port: int):
        self.output = output.resolve()
        self.protocol = read_protocol(output)
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.Lock()
        self.image_paths = {"/" + row["gallery"] for row in self.protocol["selected"]}
        super().__init__(("127.0.0.1", port), ReviewHandler)

    def state(self) -> dict:
        rows = read_reviews(self.output)
        status = review_status(rows, self.protocol)
        geometry = {row["annotation_id"]: row["normalized_box"] for row in self.protocol["selected"]}
        return {
            "revision": digest(self.output / "occlusion_review_sample_600.csv"),
            "status": status,
            "protocol_name": self.protocol["protocol_name"],
            "rows": [{**row, "normalized_box": geometry[row["annotation_id"]]} for row in rows],
        }


class ReviewHandler(BaseHTTPRequestHandler):
    def send_content(self, data: bytes, mime: str, status: int = 200, attachment: bool = False) -> None:
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if attachment:
            self.send_header("Content-Disposition", 'attachment; filename="occlusion_review_sample_600.csv"')
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, value: dict, status: int = 200) -> None:
        self.send_content(json.dumps(value, ensure_ascii=False).encode(), "application/json; charset=utf-8", status)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in ("/", "/occlusion_review.html"):
            document = (self.server.output / "occlusion_review.html").read_text(encoding="utf-8")
            self.send_content(
                document.replace("__REVIEW_TOKEN__", json.dumps(self.server.token)).encode(), "text/html; charset=utf-8"
            )
        elif path == "/api/state":
            with self.server.lock:
                self.send_json(self.server.state())
        elif path == "/api/download":
            with self.server.lock:
                self.server.state()
                self.send_content(
                    (self.server.output / "occlusion_review_sample_600.csv").read_bytes(),
                    "text/csv; charset=utf-8",
                    attachment=True,
                )
        elif path in self.server.image_paths:
            self.send_content((self.server.output / path.lstrip("/")).read_bytes(), "image/jpeg")
        else:
            self.send_json({"error": "Not found"}, 404)

    def do_POST(self) -> None:
        if self.path != "/api/reviews" or self.headers.get("X-Review-Token") != self.server.token:
            self.send_json({"error": "Unauthorized review request"}, 403)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 2_000_000:
                raise ValueError("Invalid request size.")
            payload = json.loads(self.rfile.read(length))
            with self.server.lock:
                state = self.server.state()
                if payload["revision"] != state["revision"]:
                    self.send_json({"error": "La revisión cambió en otra pestaña. Recargue antes de continuar."}, 409)
                    return
                rows = read_reviews(self.server.output)
                by_id = {row["annotation_id"]: row for row in rows}
                changes = payload["changes"]
                if not isinstance(changes, list) or not changes:
                    raise ValueError("No review changes supplied.")
                for change in changes:
                    if set(change) != {"annotation_id", *HUMAN_FIELDS} or change["annotation_id"] not in by_id:
                        raise ValueError("Only human fields of selected cases can be changed.")
                    if any(not isinstance(change[key], str) or len(change[key]) > 10000 for key in HUMAN_FIELDS):
                        raise ValueError("Invalid human review fields.")
                    if not change["reviewer"].strip():
                        raise ValueError("Ingresar el nombre del evaluador antes de guardar.")
                    by_id[change["annotation_id"]].update({key: change[key] for key in HUMAN_FIELDS})
                validate_reviews(rows, self.server.protocol)
                csv_path = self.server.output / "occlusion_review_sample_600.csv"
                shutil.copyfile(csv_path, self.server.output / "occlusion_review_sample_600.previous.csv")
                write_reviews(self.server.output, rows)
                result = self.server.state()
                write_json(self.server.output / "review_status.json", result["status"])
                with (self.server.output / "human_changes.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "at_utc": datetime.now(UTC).isoformat(),
                                "before": state["revision"],
                                "after": result["revision"],
                                "human_changes": changes,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                self.send_json({"revision": result["revision"], "status": result["status"]})
        except (ValueError, KeyError, TypeError) as error:
            self.send_json({"error": str(error)}, 400)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "serve", "status"))
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images-root", type=Path)
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--port", type=int, default=8902)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.images_root is None or args.analysis is None:
            parser.error("prepare requires --images-root and --analysis")
        result = prepare(args.parent, args.output, args.images_root, args.analysis)
    else:
        protocol = read_protocol(args.output)
        verify_parent(args.parent, protocol["parent_sha256"])
        result = review_status(read_reviews(args.output), protocol)
        if args.action == "serve":
            print(json.dumps({"url": f"http://127.0.0.1:{args.port}/occlusion_review.html", **result}), flush=True)
            ReviewServer(args.output, args.port).serve_forever()
            return
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
