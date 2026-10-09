"""Record received review evidence without replacing the canonical human CSV."""

import json
from pathlib import Path

project = Path(__file__).resolve().parents[2]
output = project / "outputs/occlusion-consolidation-2026-10-08"
receipt = json.loads((output / "consolidation_receipt.json").read_text(encoding="utf-8"))
path = project / "WILDLIFE_STATUS.json"
original = path.read_bytes()
backup = output / "status_before_receipt.json"
if not backup.exists():
    backup.write_bytes(original)
status = json.loads(original.decode("utf-8-sig"))
status["human_occlusion_sample_review"]["evaluation_ready"] = False
status["human_occlusion_sample_review"]["received_reviews"] = {
    "recorded_at_utc": receipt["created_at_utc"],
    "status": receipt["status"],
    "files": len(receipt["source_files"]),
    "cases_with_level_in_received_files": receipt["returned_cases_with_at_least_one_level"],
    "complete_cases_in_received_files": receipt["returned_cases_with_complete_evidence"],
    "complete_cases_without_conflict": receipt["draft_review_status"]["complete"],
    "cases_missing_level": receipt["unreviewed_case_numbers"],
    "conflicting_cases": receipt["conflicting_case_numbers"],
    "cases_missing_written_evidence": receipt["pending_reason_counts"]["missing_visual_evidence"],
    "draft_csv": str(output / "occlusion_review_consolidated_DRAFT.csv"),
    "receipt": str(output / "consolidation_receipt.json"),
    "canonical_csv_unchanged": True,
    "human_or_teacher_approval_inferred": False,
}
temporary = path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
temporary.replace(path)
assert json.loads(path.read_text(encoding="utf-8"))["human_occlusion_sample_review"]["received_reviews"]["files"] == 5
print("Received reviews and exact outstanding human inputs recorded; canonical CSV unchanged.")
