"""Exercise lossless JPEG compatibility and the actual installed YOLO verifier."""

import copy

import numpy as np
import pytest
from PIL import Image
from test_wildlife_data import fixture_settings

from neural_exam.common.io import file_sha256
from neural_exam.wildlife.data import prepare
from neural_exam.wildlife.jpeg import canonical_jpeg
from neural_exam.wildlife.loader import verify_manifest_images


def test_canonical_copy_preserves_source_bytes_metadata_and_decoded_pixels(tmp_path):
    source, target = tmp_path / "source.jpg", tmp_path / "copy.jpg"
    pixels = np.random.default_rng(42).integers(0, 256, (20, 20, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(source)
    valid = source.read_bytes()
    assert canonical_jpeg(source, target, file_sha256(source)) is None
    source.write_bytes(valid + b" \r\n\t")
    source.chmod(0o444)
    original, modified = source.read_bytes(), source.stat().st_mtime_ns
    result = canonical_jpeg(source, target, file_sha256(source))
    assert target.read_bytes() == valid
    assert source.read_bytes() == original and source.stat().st_mtime_ns == modified
    assert result["removed_trailing_whitespace_bytes"] == 4 and result["decoded_pixels_identical"]
    with Image.open(source) as before, Image.open(target) as after:
        assert np.array_equal(np.asarray(before), np.asarray(after))
    assert canonical_jpeg(source, target, file_sha256(source)) == result
    target.write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs"):
        canonical_jpeg(source, target, file_sha256(source))


@pytest.mark.parametrize("trailer", [b"not-whitespace", b"\x00"])
def test_unknown_trailer_is_rejected_without_modifying_source(tmp_path, trailer):
    source = tmp_path / "source.jpg"
    Image.new("RGB", (20, 20)).save(source)
    original = source.read_bytes() + trailer
    source.write_bytes(original)
    with pytest.raises(ValueError, match="Unexpected JPEG trailer"):
        canonical_jpeg(source, tmp_path / "target.jpg", file_sha256(source))
    assert source.read_bytes() == original


def test_missing_marker_and_source_digest_mismatch_are_rejected(tmp_path):
    source = tmp_path / "source.jpg"
    Image.new("RGB", (20, 20)).save(source)
    source.write_bytes(source.read_bytes()[:-2] + b" ")
    with pytest.raises(ValueError, match="missing end marker"):
        canonical_jpeg(source, tmp_path / "target.jpg", file_sha256(source))
    with pytest.raises(ValueError, match="source changed"):
        canonical_jpeg(source, tmp_path / "target.jpg", "incorrect")


def test_preparation_and_yolo_accept_all_unchanged_targets_including_backgrounds(tmp_path):
    settings = fixture_settings(tmp_path)
    for source in settings.wildlife_root.glob("*.jpg"):
        source.write_bytes(source.read_bytes() + b" ")
        source.chmod(0o444)
    originals = {source: source.read_bytes() for source in settings.wildlife_root.glob("*.jpg")}
    baseline = prepare(settings)
    settings.output_root = tmp_path / "corrected-output"
    settings.wildlife["canonical_jpeg_eoi"] = True
    corrected = prepare(settings)
    assert corrected["jpeg_materialization"]["canonical_copies"] == 10
    assert corrected["jpeg_materialization"]["all_copied_pixels_verified_identical"]
    for split, rows in corrected["records"].items():
        for before, after in zip(baseline["records"][split], rows, strict=True):
            assert {key: value for key, value in before.items() if key != "image"} == {
                key: value for key, value in after.items() if key != "image"
            }
    assert corrected["monitoring_frames"] == baseline["monitoring_frames"]
    assert prepare(settings)["identity"] == corrected["identity"]
    assert all(source.read_bytes() == original for source, original in originals.items())
    result = verify_manifest_images(corrected)
    assert result["status"] == "passed" and result["problem_count"] == 0
    assert all(row["accepted_images"] == 2 and row["backgrounds"] == 1 for row in result["splits"].values())
    # Real loader acceptance must also reject a target lost after preparation.
    altered = copy.deepcopy(corrected)
    altered["records"]["train"][0]["boxes"].append(altered["records"]["train"][0]["boxes"][0])
    assert verify_manifest_images(altered)["status"] == "failed"
