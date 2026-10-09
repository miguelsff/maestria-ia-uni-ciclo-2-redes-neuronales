"""Consolidate returned human records without deciding conflicting occlusion levels."""

import csv
import hashlib
import io
import json
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

project = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project / "scripts"))
import wildlife_occlusion_sample as sample

source = Path("E:/Kaggle/exam-output-wildlife-v6-occlusion-sample-600")
output = project / "outputs/occlusion-consolidation-2026-10-08"
audit = sample.read_json(Path(__file__).parent / "audit.json")
assert not audit["errors"], "Resolve malformed inputs before consolidation"
assert sample.digest(source / "occlusion_review_sample_600.csv") == audit["live_csv_sha256"]
protocol = sample.read_protocol(source)
sample.verify_parent(Path("E:/Kaggle/exam-output-wildlife-v6"), protocol["parent_sha256"])
with zipfile.ZipFile(project / "outputs/occlusion-offline/revision_occlusion_600_offline.zip") as archive:
    baseline = list(csv.DictReader(io.StringIO(archive.read("revision_occlusion_600/occlusion_review_sample_600.csv").decode("utf-8-sig"))))
sample.validate_reviews(baseline, protocol)
base = {row["annotation_id"]: row for row in baseline}
numbers = {row["annotation_id"]: index + 1 for index, row in enumerate(protocol["selected"])}
candidates, snapshots = defaultdict(list), {}
for info in audit["sources"]:
    name = info["file"]
    raw = (source / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == info["sha256"], "Returned file changed during review"
    snapshots[name] = raw
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline="")))
    sample.validate_reviews(rows, protocol)
    for row in rows:
        if not any(row[field].strip() for field in sample.HUMAN_FIELDS):
            continue  # An untouched blank in another person's copy does not erase a review.
        if any(row[field] != base[row["annotation_id"]][field] for field in sample.HUMAN_FIELDS):
            candidates[row["annotation_id"]].append((name, row))

consolidated, conflicts, provenance = [], [], []
for original in baseline:
    identifier = original["annotation_id"]
    choices = candidates.get(identifier, [])
    levels = {row["level"] for _, row in choices if row["level"]}
    evidence = {row["visual_evidence"] for _, row in choices if row["visual_evidence"].strip()}
    result = dict(original)
    conflict = len(levels) > 1 or len(evidence) > 1
    origin = "distributed_initial_snapshot"
    if conflict:
        for name, row in choices:
            conflicts.append({"case": numbers[identifier], "annotation_id": identifier, "file": name, **{field: row[field] for field in sample.HUMAN_FIELDS}})
        # The provisional output leaves this case pending. Both genuine alternatives remain in conflicts.csv.
        origin = "unresolved_human_conflict"
    elif choices:
        name, selected = max(choices, key=lambda choice: (sample.is_complete(choice[1]), bool(choice[1]["level"]), bool(choice[1]["visual_evidence"].strip())))
        for field in sample.HUMAN_FIELDS:
            result[field] = selected[field]
        assert all(result[field] == selected[field] for field in sample.HUMAN_FIELDS)
        origin = name
    consolidated.append(result)
    provenance.append({"case": numbers[identifier], "annotation_id": identifier, "selected_source": origin,
        "alternatives": [{"file": name, **{field: row[field] for field in sample.HUMAN_FIELDS}} for name, row in choices]})

sample.validate_reviews(consolidated, protocol)
status = sample.review_status(consolidated, protocol)
conflicted_cases = {row["case"] for row in conflicts}
pending = []
assigned_ranges = [("Miguel", 1, 120), ("Omar", 121, 240), ("Enzo Neyra", 241, 360), ("Josue", 361, 480), ("Alexander Paco Cornejo", 481, 600)]
owner = {index: name for name, start, end in assigned_ranges for index in range(start, end + 1)}
for row in consolidated:
    if sample.is_complete(row):
        continue
    case = numbers[row["annotation_id"]]
    reason = "conflicting_human_reviews" if case in conflicted_cases else "missing_level" if not row["level"] else "missing_visual_evidence"
    pending.append({"case": case, "assigned_reviewer": owner[case], "annotation_id": row["annotation_id"]
        , "level": row["level"], "reviewer": row["reviewer"], "pending_reason": reason,
        "missing_fields": ", ".join(field for field in sample.HUMAN_FIELDS if not row[field].strip())})

output.mkdir(parents=True, exist_ok=True)
(output / "received_originals").mkdir(exist_ok=True)
for name, raw in snapshots.items():
    path = output / "received_originals" / name
    if path.exists() and path.read_bytes() != raw:
        raise ValueError("Refusing to replace a different archived human source")
    path.write_bytes(raw)

def write_csv(name, rows, fields):
    with (output / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

write_csv("occlusion_review_consolidated_DRAFT.csv", consolidated, sample.FIELDS)
write_csv("pending_cases.csv", pending, ["case", "assigned_reviewer", "annotation_id", "level", "reviewer", "pending_reason", "missing_fields"])
write_csv("conflicting_reviews.csv", conflicts, ["case", "annotation_id", "file", *sample.HUMAN_FIELDS])
sample.write_json(output / "provenance.json", {"rows": provenance})
summary = {
    "created_at_utc": datetime.now(UTC).isoformat(), "status": "human_completion_and_conflict_resolution_required",
    "source_files": audit["sources"], "protocol_sha256": audit["protocol_sha256"],
    "returned_cases_with_at_least_one_level": 582, "returned_cases_with_complete_evidence": 185,
    "unreviewed_case_numbers": [entry["case"] for entry in pending if entry["pending_reason"] == "missing_level"],
    "conflicting_case_numbers": sorted(conflicted_cases), "pending_reason_counts": dict(Counter(row["pending_reason"] for row in pending)),
    "draft_review_status": status, "human_evidence_written_or_inferred": False,
    "canonical_live_csv_written": False, "original_returned_csvs_written": False,
    "training_or_evaluation_started": False, "frozen_parent_and_44_sources_unchanged": True,
    "aggregation_rule": "Ignore untouched blanks and inherited copies; same-level overlap uses the existing complete authored record; differing level or evidence stays unresolved. Preserve all alternatives and source hashes.",
    "draft_sha256": sample.digest(output / "occlusion_review_consolidated_DRAFT.csv"),
}
sample.write_json(output / "consolidation_receipt.json", summary)
readback = sample.read_json(output / "provenance.json")
assert len(readback["rows"]) == 600
assert len(consolidated) == 600 and len(conflicted_cases) == 1 and len(pending) == 416
assert status["complete"] == 184
assert summary["pending_reason_counts"] == {"missing_visual_evidence": 397, "conflicting_human_reviews": 1, "missing_level": 18}
assert sample.digest(source / "occlusion_review_sample_600.csv") == audit["live_csv_sha256"]
for info in audit["sources"]:
    assert sample.digest(source / info["file"]) == info["sha256"]
with (output / "occlusion_review_consolidated_DRAFT.csv").open(encoding="utf-8-sig", newline="") as stream:
    assert list(csv.DictReader(stream)) == consolidated
print(json.dumps({key: value for key, value in summary.items() if key not in ["source_files", "aggregation_rule"]}, indent=2, ensure_ascii=False))
print(str(output))
