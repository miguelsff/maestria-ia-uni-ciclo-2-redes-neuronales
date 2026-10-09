"""Edit copies of real ODTs, export them and verify masters stay unchanged."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from write_native_odt_uno import connect, prop


def digest(path: Path) -> str:
    """Return the complete SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_copy(desktop: object, item: dict, folder: Path) -> dict:
    """Modify paragraph, table and formula objects in a temporary copy only."""
    master = Path(item["output"]).resolve()
    before = digest(master)
    copy = (folder / master.name).resolve()
    shutil.copyfile(master, copy)
    doc = desktop.loadComponentFromURL(copy.as_uri(), "_blank", 0, (prop("Hidden", True),))
    frames = doc.TextFrames
    frame = frames.getByName("Paragraph1")
    frame.Text.String = "QAEDIT"
    result = {
        "id": item["id"],
        "master": str(master),
        "paragraph_edited": True,
        "writer_text_frames": frames.Count,
        "writer_tables": doc.TextTables.Count,
        "writer_equations": doc.EmbeddedObjects.Count,
    }
    if doc.TextTables.Count:
        table = doc.TextTables.getByIndex(0)
        table.getCellByPosition(0, 0).String = "QACELL"
        result["table_cell_edited"] = True
    else:
        result["table_cell_edited"] = "No aplica: documento sin tablas"
    if doc.EmbeddedObjects.Count:
        equation = doc.EmbeddedObjects.getByIndex(0)
        equation.Component.Formula = "QATEST = a + b"
        result["equation_edited"] = True
    else:
        result["equation_edited"] = "No aplica: no contiene ecuaciones desplegadas"
    doc.store()
    pdf = copy.with_suffix(".pdf")
    doc.storeToURL(pdf.as_uri(), (prop("FilterName", "writer_pdf_Export"),))
    doc.close(True)
    with zipfile.ZipFile(copy) as archive:
        content = archive.read("content.xml").decode("utf-8")
        assert "QAEDIT" in content
        if result["table_cell_edited"] is True:
            assert "QACELL" in content
        if result["equation_edited"] is True:
            assert any(
                "QATEST" in archive.read(name).decode("utf-8")
                for name in archive.namelist()
                if name.startswith("Object") and name.endswith("content.xml")
            )
    after = digest(master)
    assert before == after, "The scientific master changed during copy editing"
    visible = subprocess.check_output(["pdftotext", str(pdf), "-"], text=True)
    markers = {"paragraph": "QAEDIT" in visible}
    if result["table_cell_edited"] is True:
        markers["table_cell"] = "QACELL" in visible
    if result["equation_edited"] is True:
        markers["equation"] = "QATEST" in visible
    assert all(markers.values()), "An edited native object is missing from the exported PDF"
    result.update(
        master_sha256_before=before,
        master_sha256_after=after,
        master_unchanged=True,
        copy_odt=str(copy),
        copy_pdf=str(pdf),
        pdf_created=pdf.exists() and pdf.stat().st_size > 0,
        edited_markers_visible_in_export=markers,
    )
    return result


def main() -> None:
    """Run native editing checks inside an isolated renderer session."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--copies", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.copies.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="exam-editability-") as folder:
        process, desktop = connect(Path(folder))
        try:
            results = [verify_copy(desktop, item, args.copies) for item in config["documents"]]
            args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(results, ensure_ascii=False))
        finally:
            desktop.terminate()
            process.wait(timeout=15)


if __name__ == "__main__":
    main()
