"""Label extraction and leakage checks using synthetic text, without clinical files."""

import numpy as np
import pydicom
import pytest
import torch
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from neural_exam.mri.data import MRIDataset, decode_stacks, select_studies
from neural_exam.mri.labels import LABELS, extract_labels
from neural_exam.mri.loss import classification_loss
from neural_exam.mri.model import StudyClassifier
from neural_exam.mri.pipeline import training_alpha


@pytest.mark.parametrize(
    ("report", "label", "expected"),
    [
        ("ACL tear, MCL intact.", "ACL", 1),
        ("ACL tear, MCL intact.", "MCL", 0),
        ("Sin derrame articular.", "Effusion", 0),
        ("Possible ACL tear.", "ACL", None),
        ("No se descarta fractura.", "Fracture", None),
        ("ACL is intact. ACL tear.", "ACL", None),
        ("骨折", "Fracture", 1),
        ("未见骨折", "Fracture", 0),
        ("Derrame y sinovitis.", "Synovitis", 1),
        ("No effusion or synovitis.", "Synovitis", 0),
        ("Possible ACL or MCL tear.", "MCL", None),
        ("ACL tear without effusion.", "ACL", 1),
        ("ACL tear without effusion.", "Effusion", 0),
    ],
)
def test_negation_uncertainty_and_conflicting_assertions(report, label, expected):
    result = extract_labels(report)
    assert result["values"][LABELS.index(label)] == expected


def test_unsupported_report_abstains():
    result = extract_labels("Unrecognized text without any target entity")
    assert not result["supported"]
    assert result["values"] == [None] * 12


def test_gold_studies_never_enter_training():
    studies = [{"StudyInstanceUID": "gold", "Report": "ACL tear", **dict.fromkeys(LABELS, "0")}]
    studies += [{"StudyInstanceUID": f"weak{i}", "Report": "ACL tear"} for i in range(8)]
    selected = select_studies(studies, 8, 42, 0.2)
    assert next(record["split"] for record in selected if record["study_id"] == "gold") == "gold"
    assert len({record["study_id"] for record in selected}) == len(selected)
    assert {record["split"] for record in selected} == {"train", "val", "gold"}
    assert select_studies(studies, 8, 42, 0.2) == selected


def test_weak_split_keeps_rare_label_positives_in_both_partitions():
    studies = [{"StudyInstanceUID": "gold", "Report": "ACL tear", **dict.fromkeys(LABELS, "0")}]
    studies += [
        {
            "StudyInstanceUID": f"weak{i}",
            "Report": "ACL tear. " + ("Lateral compartment osteoarthritis." if i < 4 else "No effusion."),
        }
        for i in range(40)
    ]
    selected = select_studies(studies, 41, 42, 0.2)
    label = LABELS.index("Lateral OA")
    assert sum(row["targets"][label] == 1 for row in selected if row["split"] == "train") >= 1
    assert sum(row["targets"][label] == 1 for row in selected if row["split"] == "val") >= 1


@pytest.mark.parametrize("kind", ["bce", "focal"])
def test_missing_labels_have_zero_gradient(kind):
    logits = torch.zeros((1, 2), requires_grad=True)
    target = torch.tensor([[1.0, float("nan")]])
    loss = classification_loss(logits, target, torch.ones_like(logits), kind=kind)
    loss.backward()
    assert logits.grad[0, 0] != 0
    assert logits.grad[0, 1] == 0


def test_original_cnn_study_output_shape():
    model = StudyClassifier("scratch").eval()
    with torch.inference_mode():
        assert model(torch.zeros(2, 2, 3, 32, 32)).shape == (2, 12)


def test_focal_balance_excludes_rejected_low_confidence_labels():
    records = [
        {"targets": [1.0] * 12, "weak": {"confidence": [0.9] * 12}},
        {"targets": [0.0] * 12, "weak": {"confidence": [0.2] * 12}},
    ]
    assert torch.allclose(training_alpha(records, "weighted", 0.8), torch.full((12,), 0.5))
    assert torch.allclose(training_alpha(records, "high_confidence", 0.8), torch.full((12,), 0.1))


def test_dicom_normalization_stays_float32_and_orders_geometry(tmp_path):
    for index in (2, 0, 1):
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = MRImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        dataset = FileDataset(str(tmp_path / f"{index}.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
        dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        dataset.ImagePositionPatient = [0, 0, index]
        dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        dataset.Rows = dataset.Columns = 8
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.BitsAllocated = dataset.BitsStored = 16
        dataset.HighBit = 15
        dataset.PixelRepresentation = 0
        dataset.PixelData = (np.arange(64, dtype=np.uint16).reshape(8, 8) + index * 100).tobytes()
        pydicom.dcmwrite(dataset.filename, dataset, enforce_file_format=True)
    stacks, details = decode_stacks(tmp_path, 1, 8)
    assert stacks.dtype == np.float32
    assert stacks.shape == (1, 3, 8, 8)
    assert np.isfinite(stacks).all() and 0 <= stacks.min() <= stacks.max() <= 1
    assert np.all(np.diff(stacks.mean(axis=(0, 2, 3))) > 0)
    assert details["geometric_sort"] and details["slice_count"] == 3


def test_integer_weak_labels_are_converted_to_floating_bce_targets(tmp_path):
    path = tmp_path / "cache.npz"
    np.savez_compressed(path, image=np.zeros((2, 3, 32, 32), dtype=np.float32))
    dataset = MRIDataset(
        [{"cache": str(path), "targets": [1, 0] * 6, "split": "train", "weak": {"confidence": [0.9] * 12}}]
    )
    image, target, weights = dataset[0]
    assert image.dtype == target.dtype == weights.dtype == torch.float32
    assert torch.isfinite(classification_loss(torch.zeros(12), target, weights))
