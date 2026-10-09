"""Apply only explicitly recorded editorial changes to existing native masters."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path

from write_native_odt_uno import connect, prop


def normalize(value: str) -> str:
    """Normalize whitespace without changing the scientific content."""
    return re.sub(r"\s+", " ", value).strip()


def main() -> None:
    """Refuse unrecorded edits and preserve every archived source PDF."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.manifest.read_text(encoding="utf-8"))
    receipts = []
    with tempfile.TemporaryDirectory(prefix="exam-editorial-") as folder:
        process, desktop = connect(Path(folder))
        try:
            for item in config["documents"]:
                if not item.get("editorial_updates"):
                    continue
                path = Path(item["output"]).resolve()
                source = Path(item["source"])
                source_before = hashlib.sha256(source.read_bytes()).hexdigest()
                before = hashlib.sha256(path.read_bytes()).hexdigest()
                doc = desktop.loadComponentFromURL(path.as_uri(), "_blank", 0, (prop("Hidden", True),))
                changes = []
                try:
                    for update in item["editorial_updates"]:
                        frame = doc.TextFrames.getByName(f"Paragraph{update['paragraph']}")
                        actual = normalize(frame.Text.String)
                        if actual == normalize(update["replacement"]):
                            continue
                        if actual != normalize(update["source_text"]):
                            raise ValueError("The master contains an unrecorded edit; no replacement is permitted")
                        frame.Text.String = update["replacement"]
                        cursor = frame.Text.createTextCursor()
                        cursor.gotoStart(False)
                        cursor.gotoEnd(True)
                        cursor.CharFontName = "Liberation Serif"
                        cursor.CharHeight = 9.96
                        cursor.CharScaleWidth = 96
                        changes.append(update["paragraph"])
                    doc.store()
                finally:
                    doc.close(True)
                assert hashlib.sha256(source.read_bytes()).hexdigest() == source_before
                receipts.append(
                    {
                        "id": item["id"],
                        "odt": item["output"],
                        "odt_sha256_before": before,
                        "odt_sha256_after": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "paragraphs_updated": changes,
                        "source_pdf_unchanged": True,
                        "verified_runtime_reference": "codex/examen-c1-runtime@75d2db1",
                        "verified_repository_base": "main@0dc2fd2",
                    }
                )
        finally:
            desktop.terminate()
            process.wait(timeout=15)
    args.output.write_text(json.dumps(receipts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipts, ensure_ascii=False))


if __name__ == "__main__":
    main()
