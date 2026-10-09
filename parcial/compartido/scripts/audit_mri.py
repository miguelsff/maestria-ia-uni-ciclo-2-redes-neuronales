"""Audit MRI metadata and selected original DICOM files before training."""

import hashlib
from collections import Counter, defaultdict
from datetime import UTC, datetime

import pydicom

from neural_exam.common.io import file_sha256, fingerprint, read_csv, read_json, write_json
from neural_exam.config import load_settings
from neural_exam.mri.data import choose_series, find_dicom_root, is_gold, select_studies
from neural_exam.mri.labels import LABELS, extract_labels


def main() -> None:
    """Check correspondence and identity without using gold outcomes for selection."""
    settings = load_settings("configs/mri.toml")
    studies = read_csv(settings.metadata_root / "train.csv")
    descriptors = read_csv(settings.metadata_root / "train_series.csv")
    root = find_dicom_root(settings.mri_root)
    study_ids = {row["StudyInstanceUID"] for row in studies}
    inventory = read_json(settings.output_root / "audit/source_inventory.json")
    original_hashes = {row["file"]: row for row in read_json(settings.output_root / "audit/cache_source_hashes.json")}
    folder_ids = set(inventory)
    assert len(study_ids) == len(studies) == 4407
    assert study_ids == folder_ids, "Study metadata and source folders differ."
    assert sum(is_gold(row) for row in studies) == 58
    by_study = defaultdict(list)
    pairs = set()
    for row in descriptors:
        pair = row["StudyInstanceUID"], row["SeriesInstanceUID"]
        assert pair not in pairs, "Duplicate series descriptor."
        assert pair[0] in study_ids
        assert pair[1] in inventory[pair[0]], "Missing original series directory."
        pairs.add(pair)
        by_study[pair[0]].append(row)
    assert set(by_study) == study_ids
    selected = select_studies(
        studies, settings.mri["max_studies"], settings.training["seed"], settings.mri["validation_fraction"]
    )
    assert len(selected) == 600
    assert len({row["study_id"] for row in selected}) == len(selected)
    split_positive_counts = {
        split: [
            sum(row["targets"][index] == 1 for row in selected if row["split"] == split) for index in range(len(LABELS))
        ]
        for split in ("train", "val")
    }
    assert all(count > 0 for count in split_positive_counts["train"]), "Every target needs positive training evidence."
    assert all(count > 0 for count in split_positive_counts["val"]), (
        "Every target needs positive weak-validation evidence."
    )
    patient_splits, series_audit, content = defaultdict(set), [], []
    patient_equals_study = 0
    syntaxes = Counter()
    for index, record in enumerate(selected, 1):
        descriptor = choose_series(by_study[record["study_id"]])
        source = root / record["study_id"] / descriptor["SeriesInstanceUID"]
        files = sorted(source.glob("*.dcm"))
        assert len(files) >= 3
        file_hashes, instances, patients = [], set(), set()
        for path in files:
            header = pydicom.dcmread(path, stop_before_pixels=True)
            assert str(header.StudyInstanceUID) == record["study_id"]
            assert str(header.SeriesInstanceUID) == descriptor["SeriesInstanceUID"]
            instance = str(header.SOPInstanceUID)
            assert instance not in instances
            instances.add(instance)
            patients.add(str(header.get("PatientID", "")))
            syntaxes[str(header.file_meta.TransferSyntaxUID)] += 1
            digest = file_sha256(path)
            original = original_hashes[path.relative_to(root.parent).as_posix()]
            assert original["sha256"] == digest and original["bytes"] == path.stat().st_size
            file_hashes.append((path.name, path.stat().st_size, digest))
            content.append(
                {
                    "study_id": record["study_id"],
                    "series_id": descriptor["SeriesInstanceUID"],
                    "file": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": digest,
                }
            )
        assert len(patients) == 1
        patient = patients.pop()
        patient_equals_study += patient == record["study_id"]
        if patient:
            patient_splits[hashlib.sha256(patient.encode()).hexdigest()].add(record["split"])
        series_audit.append(
            {
                "study_id": record["study_id"],
                "split": record["split"],
                "series_id": descriptor["SeriesInstanceUID"],
                "slices": len(files),
                "content_sha256": fingerprint(file_hashes),
            }
        )
        if index % 25 == 0:
            print(f"Audited {index}/600 selected series", flush=True)
    overlaps = [key for key, values in patient_splits.items() if len(values) > 1]
    assert not overlaps, "Repeated available PatientID spans partitions."
    weak_rules = [extract_labels(row.get("Report", "")) for row in studies if not is_gold(row)]
    receipt = {
        "completed_utc": datetime.now(UTC).isoformat(),
        "metadata_sha256": {
            name: file_sha256(settings.metadata_root / name) for name in ("train.csv", "train_series.csv")
        },
        "source_metadata_matches": {
            name: file_sha256(settings.metadata_root / name) == file_sha256(root.parent / name)
            for name in ("train.csv", "train_series.csv")
        },
        "all_metadata_studies": len(studies),
        "all_metadata_series": len(descriptors),
        "selected_studies": len(selected),
        "selected_dicom_files": len(content),
        "selected_original_content_identity": fingerprint(content),
        "docker_cache_byte_matches": True,
        "split_counts": dict(Counter(row["split"] for row in selected)),
        "weak_split_positive_counts": split_positive_counts,
        "series": series_audit,
        "transfer_syntaxes": dict(syntaxes),
        "patient_id_equals_study_id": patient_equals_study,
        "patient_split_overlaps": overlaps,
        "patient_independence": "Anonymized IDs cannot establish independence of real patients across studies.",
        "weak_reports_supported": sum(row["supported"] for row in weak_rules),
        "weak_reports_by_script": dict(Counter(row["script"] for row in weak_rules)),
        "weak_supported_by_script": dict(Counter(row["script"] for row in weak_rules if row["supported"])),
        "labels": LABELS,
        "pixel_decoding": "Required separately by prepare for every selected slice.",
    }
    assert all(receipt["source_metadata_matches"].values())
    write_json(settings.output_root / "audit/mri_source_files.json", content)
    write_json(settings.output_root / "audit/mri_data.json", receipt)
    print({key: receipt[key] for key in ("split_counts", "selected_dicom_files", "patient_id_equals_study_id")})


if __name__ == "__main__":
    main()
