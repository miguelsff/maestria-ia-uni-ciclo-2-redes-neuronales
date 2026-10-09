"""Read-only audit of the five returned reviews against the fixed 600 cases."""

import csv
import io
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

project = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project / "scripts"))
import wildlife_occlusion_sample as sample

source = Path("E:/Kaggle/exam-output-wildlife-v6-occlusion-sample-600")
names = [
    "occlusion_review_Josue_2026-10-07T03-57-25-829Z.csv",
    "occlusion_review_Miguel_2026-10-08T01-44-38-653Z.csv",
    "occlusion_review_Omar_2026-10-06T23-56-13-695Z_3.csv",
    "occlusion_review_Alexander_Paco_Cornejo_2026-10-08T23-53-07-939Z.csv",
    "occlusion_review_Enzo_Neyra_2026-10-06T15-52-12-067Z.csv",
]
protocol = sample.read_protocol(source)
with zipfile.ZipFile(project / "outputs/occlusion-offline/revision_occlusion_600_offline.zip") as archive:
    baseline = list(csv.DictReader(io.StringIO(archive.read("revision_occlusion_600/occlusion_review_sample_600.csv").decode("utf-8-sig"))))
sample.validate_reviews(baseline, protocol)
reference = {row["annotation_id"]: row for row in baseline}
numbers = {row["annotation_id"]: index + 1 for index, row in enumerate(protocol["selected"])}
reviews, reports, errors = defaultdict(list), [], []
for name in names:
    data = (source / name).read_bytes()
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""))
    rows = list(reader)
    try:
        if reader.fieldnames != list(sample.FIELDS) or any(set(row) != set(sample.FIELDS) for row in rows):
            raise ValueError("CSV schema changed")
        sample.validate_reviews(rows, protocol)
        if any(not isinstance(row.get(field), str) for row in rows for field in sample.HUMAN_FIELDS):
            raise ValueError("Missing human field cells")
    except (ValueError, KeyError) as error:
        errors.append({"file": name, "error": str(error)})
        continue
    changes = [row for row in rows if any(row[field] != reference[row["annotation_id"]][field] for field in sample.HUMAN_FIELDS)]
    complete = [row for row in changes if sample.is_complete(row)]
    partial = [row for row in changes if not sample.is_complete(row)]
    cases = sorted(numbers[row["annotation_id"]] for row in complete)
    for row in changes:
        reviews[row["annotation_id"]].append((name, row))
    reports.append({
        "file": name, "sha256": sample.digest(source / name), "rows": len(rows),
        "changed_complete": len(complete), "changed_incomplete": len(partial),
        "complete_reviewers": dict(Counter(row["reviewer"] for row in complete)),
        "complete_levels": dict(Counter(row["level"] for row in complete)),
        "case_ranges": [[min(cases), max(cases)]] if cases and cases == list(range(min(cases), max(cases) + 1)) else cases,
        "incomplete_case_numbers": [numbers[row["annotation_id"]] for row in partial],
        "assigned_levels": sum(bool(row["level"]) for row in changes),
        "missing_evidence_with_level": sum(bool(row["level"]) and not row["visual_evidence"].strip() for row in changes),
        "missing_reviewer_with_level": sum(bool(row["level"]) and not row["reviewer"].strip() for row in changes),
        "changed_reviewer_names": dict(Counter(row["reviewer"] for row in changes)),
        "overlap_human_fields": [{"case": numbers[row["annotation_id"]], **{field: row[field] for field in sample.HUMAN_FIELDS}} for row in changes if numbers[row["annotation_id"]] in [1, 121, 122, 123, 124, 125]],
    })
conflicts = []
for identifier, values in reviews.items():
    distinct = {tuple(row[field] for field in sample.HUMAN_FIELDS) for _, row in values}
    if len(distinct) > 1:
        conflicts.append({"case": numbers[identifier], "files": [name for name, _ in values]})
complete_ids = {identifier for identifier, values in reviews.items() if any(sample.is_complete(row) for _, row in values)}
report = {
    "protocol_sha256": sample.digest(source / "protocol.json"), "sources": reports, "errors": errors,
    "distinct_complete_cases": len(complete_ids), "conflicts": conflicts,
    "missing_case_numbers": sorted(numbers[row["annotation_id"]] for row in baseline if row["annotation_id"] not in complete_ids),
    "live_csv_sha256": sample.digest(source / "occlusion_review_sample_600.csv"),
}
(Path(__file__).parent / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({"sources": [{key: value for key, value in row.items() if key not in ["incomplete_case_numbers", "case_ranges"]} for row in reports], "conflicts": conflicts, "distinct_complete_cases": len(complete_ids), "errors": errors}, indent=2, ensure_ascii=False))
