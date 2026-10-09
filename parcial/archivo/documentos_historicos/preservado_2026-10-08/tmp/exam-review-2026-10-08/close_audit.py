"""Save the audit conclusion and verify that reviewed inputs remain preserved."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DESTINATION = ROOT / "audits/revision-2026-10-08"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run():
    before = read_json(DESTINATION / "preservation-before.json")
    preservation = {path: digest(path) == expected for path, expected in before.items()}
    assert all(preservation.values())
    scientific = {name: read_json(DESTINATION / f"{name}-scientific.json")
                  for name in ("mri", "flood", "wildlife", "occlusion")}
    assert all(receipt["status"] == "passed" for receipt in scientific.values())
    assert read_json(DESTINATION / "flood-qualitative.json")["status"] == "passed"
    assert read_json(DESTINATION / "structure.json")["status"] == "passed"
    accepted_root = Path("E:/Kaggle/exam-output-wildlife-v6-occlusion-sample-600")
    level_files = accepted_root / "accepted_evaluation/6b2983c33abccbc1"
    snapshot = {
        "completed_levels": [level for level in ("low", "medium", "high") if (level_files/f"{level}.json").exists()],
        "aggregate_complete": (accepted_root/"accepted_evaluation_latest.json").exists(),
        "delivery_job": read_json(accepted_root/"accepted_delivery_job.json"),
        "pdfs": [str(path) for path in accepted_root.glob("accepted_delivery/**/*.pdf")],
        "progress_log_tail": (accepted_root/"accepted_evaluation_run_20261008.log").read_text()[-850:],
    }
    report_checks = {
        "mri": {"pages": 9, "references": 7, "authors": 5,
                "visual_pages_checked": list(range(1, 10)), "mandatory_deliverables": 7,
                "mandatory_deliverables_fulfilled_scientifically": 7},
        "flood": {"pages": 9, "references": 7, "authors": 5,
                  "visual_pages_checked": list(range(1, 10)), "mandatory_deliverables": 5,
                  "mandatory_deliverables_fulfilled_scientifically": 5, "optional_optical_bonus": "falta"},
        "wildlife": {"pages": 10, "references": 6, "authors": 5,
                     "visual_pages_checked": list(range(1, 11)), "mandatory_deliverables": 6,
                     "mandatory_deliverables_fulfilled_scientifically": 5,
                     "occlusion_and_size_deliverable": "parcial",
                     "occlusion_current_report_integration": "falta"},
    }
    result = dict(
        completed_at_utc=datetime.now(timezone.utc).isoformat(),
        review_date_peru="2026-10-08", scope="Diagnostic audit only; no scientific fixes, retraining, publication or submissions",
        reviewed_input_hashes_unchanged=preservation,
        reports=report_checks,
        tests={"passed": 80, "skipped_cuda_only": 3, "warnings": 34,
               "runtime": "CPU; Python 3.12; local current source; isolated container",
               "limitations": "Synthetic control tests and saved-prediction recalculations do not establish fresh end-to-end scientific reproduction."},
        accepted_occlusion_snapshot=snapshot,
        common_unverified=["Teacher access to private repository", "Kaggle rule acceptance by all five members and external academic-use verification", "Course-platform submission receipt", "30 October 2026 class presentation"],
        shared_pending=["Publish the accepted-582 workflow and reproducibility inputs",
                        "Replace deleted branch links with verified main commit links",
                        "Document and clarify the generic own-components requirement versus the specific adjusted-YOLO requirement"],
        full_fresh_scientific_reproduction="no verificado",
        priority_pending=[
            {"priority": 0, "item": "Finish the already-running accepted-occlusion evaluation and validate its aggregate and intervals"},
            {"priority": 0, "item": "Integrate the accepted-582 results, sample limitations and occlusion conclusions into the wildlife report within 6-10 pages"},
            {"priority": 1, "item": "Publish and document the accepted workflow, source reviews and complete reproducible command; update stale branch references"},
            {"priority": 1, "item": "Record teacher-access and Kaggle-use evidence, clarify the own-components/YOLO interpretation, and retain platform/presentation receipts"},
        ],
    )
    (DESTINATION / "closure.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({"input_files_unchanged":len(preservation), "reports":report_checks,
                      "accepted_occlusion_snapshot":snapshot},indent=2,ensure_ascii=False))


if __name__ == "__main__":
    run()
