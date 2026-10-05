"""Persist the deterministic MRI series selection without opening DICOM files."""

from collections import defaultdict

from neural_exam.common.io import read_csv, write_json
from neural_exam.config import load_settings
from neural_exam.mri.data import choose_series, select_studies


def main() -> None:
    """Freeze the source-copy inventory from metadata and the established protocol."""
    settings = load_settings("configs/mri.toml")
    descriptors = defaultdict(list)
    for row in read_csv(settings.metadata_root / "train_series.csv"):
        descriptors[row["StudyInstanceUID"]].append(row)
    studies = select_studies(
        read_csv(settings.metadata_root / "train.csv"),
        settings.mri["max_studies"],
        settings.training["seed"],
        settings.mri["validation_fraction"],
    )
    write_json(
        settings.output_root / "audit/selected_series.json",
        [
            {
                "study_id": row["study_id"],
                "series_id": choose_series(descriptors[row["study_id"]])["SeriesInstanceUID"],
                "split": row["split"],
            }
            for row in studies
        ],
    )
    print(f"Selected {len(studies)} MRI studies.")


if __name__ == "__main__":
    main()
