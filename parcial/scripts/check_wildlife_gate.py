"""Check current human evidence without training, preparing data, or loading YOLO."""

from neural_exam.common.io import write_json
from neural_exam.config import load_settings
from neural_exam.wildlife.review import architecture_review_status, label_review_status


def main() -> None:
    """Fail before scientific execution when genuine label review is incomplete."""
    settings = load_settings(device="cpu")
    labels = label_review_status(
        settings.output_root / "wildlife/label_review.csv",
        require_evidence=settings.wildlife.get("require_label_evidence", False),
    )
    architecture = architecture_review_status(settings.output_root / "receipts/architecture-review.json")
    status = {
        "status": "reviewed" if labels["status"] == architecture["status"] == "reviewed" else "human_review_pending",
        "labels": labels,
        "architecture": architecture,
        "model_loaded": False,
    }
    write_json(settings.output_root / "receipts/scientific-gate.json", status)
    print(status)
    if status["status"] != "reviewed":
        raise SystemExit("Human label or architecture evidence remains incomplete; no model was loaded.")


if __name__ == "__main__":
    main()
