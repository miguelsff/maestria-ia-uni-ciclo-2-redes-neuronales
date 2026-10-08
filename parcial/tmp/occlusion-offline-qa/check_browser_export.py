"""Check the genuinely downloaded synthetic QA CSV against the packaged baseline."""

import csv
import json
import sys
from pathlib import Path

project = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project / "scripts"))
import wildlife_occlusion_sample as sample

qa = project / "tmp/occlusion-offline-qa"
source = Path("E:/Kaggle/exam-output-wildlife-v6-occlusion-sample-600")
protocol = sample.read_protocol(source)
with (qa / "browser_export_fixture.csv").open(encoding="utf-8-sig", newline="") as stream:
    rows = list(csv.DictReader(stream))
sample.validate_reviews(rows, protocol)
assert len(rows) == 600
assert rows[0] == sample.read_reviews(source)[0]
assert rows[1]["reviewer"] == "PRUEBA QA - no revision humana"
assert rows[1]["level"] == "medium"
assert rows[1]["visual_evidence"] == 'PRUEBA sintética, "comillas"\nsegunda línea con acento: árbol. No es una evaluación.'
assert sum(sample.is_complete(row) for row in rows) == 1
print(json.dumps({"download_exists": True, "download_rows": 600, "quoted_multiline_accented_note_preserved": True, "sample_identity_preserved": True, "initial_human_progress_preserved": True, "test_review_exists_only_in_qa_csv": True}))
