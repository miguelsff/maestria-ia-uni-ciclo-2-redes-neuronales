"""Verify parent targets before carrying existing visual judgments to v3."""

from pathlib import Path

from neural_exam.common.io import read_json, write_json
from neural_exam.config import load_settings
from neural_exam.wildlife.corrections import migrate_reviews
from neural_exam.wildlife.review import prepare_label_review


def main() -> None:
    """Keep the changed photo unapproved and refresh HTML from the actual CSV."""
    settings = load_settings(device="cpu")
    output = settings.output_root / "wildlife"
    manifest = read_json(output / "manifest.json")
    parent = settings.wildlife.get("review_parent")
    if parent:
        print(migrate_reviews(manifest, output, Path(parent)))
    sample = read_json(output / "label_review_sample.json")
    sample["require_evidence"] = settings.wildlife.get("require_label_evidence", False)
    write_json(output / "label_review_sample.json", sample)
    prepare_label_review(manifest, output, settings.training["seed"])


if __name__ == "__main__":
    main()
