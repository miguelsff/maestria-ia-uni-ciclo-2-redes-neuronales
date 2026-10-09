"""Verify portable packaging on temporary data; never create live human reviews."""

import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
SPEC = importlib.util.spec_from_file_location("package_occlusion_offline", PROJECT / "scripts/package_occlusion_offline.py")
PACKAGER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACKAGER)
import wildlife_occlusion_sample as sample  # noqa: E402


class OfflinePackageTests(unittest.TestCase):
    """The distributable preserves case identity, originals and genuine source data."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "live"
        (self.source / "images").mkdir(parents=True)
        image = self.source / "images/fixture.jpg"
        image.write_bytes(b"Synthetic fixture image, never a real review photograph")
        analysis = self.source / "sampling_analysis.json"
        analysis.write_text("{}", encoding="utf-8")
        selected = []
        for index in range(600):
            row = {field: "fixture" for field in sample.FIELDS if field not in sample.HUMAN_FIELDS}
            row.update(
                annotation_id=str(index),
                gallery="images/fixture.jpg",
                normalized_box=[0.5, 0.5, 0.2, 0.2],
                source_sha256=sample.digest(image),
            )
            selected.append(row)
        self.protocol = {"analysis_sha256": sample.digest(analysis), "selected": selected}
        self.rows = [{field: "" if field in sample.HUMAN_FIELDS else row[field] for field in sample.FIELDS} for row in selected]
        self.rows[0].update(level="low", visual_evidence='Synthetic note, "quoted"\n</script><img src=x> __REVIEW_CORE__', reviewer="Fixture only")
        self.write_fixture()
        self.output = self.root / "portable/review.zip"

    def write_fixture(self):
        sample.write_json(self.source / "protocol.json", self.protocol)
        (self.source / "protocol.sha256").write_text(sample.digest(self.source / "protocol.json"), encoding="utf-8")
        sample.write_reviews(self.source, self.rows)

    def test_snapshot_is_portable_and_source_is_unchanged(self):
        before = {path.relative_to(self.source).as_posix(): sample.digest(path) for path in self.source.rglob("*") if path.is_file()}
        result = PACKAGER.package(self.source, self.output)
        self.assertEqual(result["cases"], 600)
        self.assertEqual(result["images"], 1)
        with zipfile.ZipFile(self.output) as archive:
            self.assertIsNone(archive.testzip())
            root = PACKAGER.ROOT + "/"
            manifest = json.loads(archive.read(root + "manifest_package.json"))
            for name, expected in manifest["sha256"].items():
                self.assertEqual(hashlib.sha256(archive.read(root + name)).hexdigest(), expected)
            self.assertEqual(archive.read(root + "occlusion_review_sample_600.csv"), (self.source / "occlusion_review_sample_600.csv").read_bytes())
            html = archive.read(root + "index.html").decode()
            payload_text = html.split('<script id="sample-data" type="application/json">', 1)[1].split("</script>", 1)[0]
            payload = json.loads(payload_text)
            self.assertEqual(payload["rows"][0]["visual_evidence"], self.rows[0]["visual_evidence"])
            self.assertNotIn("</script><img", html)
            self.assertNotIn("fetch(", html)
            self.assertIn("connect-src 'none'", html)
            self.assertEqual(len(payload["rows"]), 600)
            self.assertEqual(manifest["complete_reviews_at_snapshot"], 1)
            reader = sample.csv.DictReader(io.StringIO(archive.read(root + "occlusion_review_sample_600.csv").decode("utf-8-sig")))
            self.assertEqual(list(reader), self.rows)
        after = {path.relative_to(self.source).as_posix(): sample.digest(path) for path in self.source.rglob("*") if path.is_file()}
        self.assertEqual(before, after)

    def test_rejects_changed_image_without_publishing_zip(self):
        (self.source / "images/fixture.jpg").write_bytes(b"Changed fixture")
        with self.assertRaisesRegex(ValueError, "original image changed"):
            PACKAGER.package(self.source, self.output)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.output.with_suffix(".zip.tmp").exists())

    def test_rejects_case_changes_and_unsafe_paths(self):
        self.rows[1]["annotation_id"] = self.rows[0]["annotation_id"]
        sample.write_reviews(self.source, self.rows)
        with self.assertRaisesRegex(ValueError, "original 600 IDs"):
            PACKAGER.package(self.source, self.output)
        self.rows[1]["annotation_id"] = "1"
        for row in self.protocol["selected"]:
            row["gallery"] = "../outside.jpg"
        for row in self.rows:
            row["gallery"] = "../outside.jpg"
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            PACKAGER.package(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_output_cannot_modify_live_review(self):
        with self.assertRaisesRegex(ValueError, "outside the live"):
            PACKAGER.package(self.source, self.source / "review.zip")


if __name__ == "__main__":
    unittest.main()
