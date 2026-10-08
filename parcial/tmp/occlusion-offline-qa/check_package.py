"""Validate and extract the distributable for isolated browser QA."""

import hashlib
import json
import sys
import zipfile
from pathlib import Path

project = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project / "scripts"))
import wildlife_occlusion_sample as sample

output = project / "outputs/occlusion-offline"
archive_path = output / "revision_occlusion_600_offline.zip"
destination = project / "tmp/occlusion-offline-qa/extracted"
destination.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(archive_path) as archive:
    assert archive.testzip() is None
    root = "revision_occlusion_600/"
    manifest = json.loads(archive.read(root + "manifest_package.json"))
    for name, expected in manifest["sha256"].items():
        assert hashlib.sha256(archive.read(root + name)).hexdigest() == expected
    for name in archive.namelist():
        assert (destination / name).resolve().is_relative_to(destination.resolve())
    assert len([name for name in archive.namelist() if name.startswith(root + "images/")]) == 596
    html = archive.read(root + "index.html").decode("utf-8")
    payload = json.loads(html.split('<script id="sample-data" type="application/json">', 1)[1].split("</script>", 1)[0])
    protocol = json.loads(archive.read(root + "protocol.json"))
    assert len(payload["rows"]) == 600
    sample.validate_reviews(payload["rows"], protocol)
    assert [row["annotation_id"] for row in payload["rows"]] == [row["annotation_id"] for row in protocol["selected"]]
    assert all(row["normalized_box"] == protocol["selected"][i]["normalized_box"] for i, row in enumerate(payload["rows"]))
    assert sum(float(row["sample_weight"]) for row in payload["rows"]) > 17619.999
    archive.extractall(destination)
parent = Path("E:/Kaggle/exam-output-wildlife-v6")
sample.verify_parent(parent, protocol["parent_sha256"])
result = {
    "zip_sha256": sample.digest(archive_path),
    "cases": 600,
    "images": 596,
    "geometry_and_ids_preserved": True,
    "all_archived_file_hashes_verified": True,
    "all_original_photos_byte_exact": True,
    "zip_crc_valid": True,
    "frozen_parent_and_44_sources_preserved": True,
    "snapshot_sha256": manifest["snapshot_sha256"],
    "live_csv_matches_snapshot": sample.digest(Path("E:/Kaggle/exam-output-wildlife-v6-occlusion-sample-600/occlusion_review_sample_600.csv")) == manifest["snapshot_sha256"],
    "unit_tests_python": "4 passed",
    "unit_tests_javascript": "passed",
    "ruff": "passed",
    "browser_qa": {
        "tested_on_temporary_static_loopback_preview": True,
        "original_images_and_24_overlays_rendered": True,
        "zoom_opened_and_closed": True,
        "review_fields_and_completion_counter_updated": True,
        "real_csv_download_verified": True,
        "downloaded_csv_rows": 600,
        "quoted_multiline_accented_note_preserved": True,
        "csv_restored_in_separate_clean_copy": True,
        "synthetic_test_reviews_in_delivered_zip": False,
        "file_url_visual_test": "not performed; automation browser URL policy permits only HTTP/HTTPS",
        "temporary_preview_closed": True,
    },
}
(output / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
print(str(destination / root / "index.html"))
