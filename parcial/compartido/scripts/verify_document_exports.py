"""Compare clean-clone document exports with approved PDFs without changing masters."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import fitz
from PIL import Image, ImageChops

NS = {
    "draw": "urn:oasis:names:tc:opendocument:xmlns:drawing:1.0",
    "table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
    "math": "http://www.w3.org/1998/Math/MathML",
}
AUTHORS = (
    "Chilón Cruzado Josué",
    "Concepción Tiza Miguel Ángel",
    "Neyra Dianderas Enzo Israel",
    "Tello Guerra Omar Salvador",
    "Alexander Vladimir Paco Cornejo",
)


def digest(path: Path) -> str:
    """Hash the complete file, including metadata."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict | list:
    """Read a UTF-8 JSON receipt."""
    return json.loads(path.read_text(encoding="utf-8"))


def normalized(text: str) -> str:
    """Ignore accents and whitespace for author checks only."""
    decomposed = unicodedata.normalize("NFKD", text)
    return " ".join("".join(c for c in decomposed if not unicodedata.combining(c)).casefold().split())


def native_counts(path: Path) -> dict:
    """Count native paragraph frames, Writer tables and embedded MathML objects."""
    with zipfile.ZipFile(path) as archive:
        content = ET.fromstring(archive.read("content.xml"))
        paragraphs = sum(
            frame.get(f"{{{NS['draw']}}}name", "").startswith("Paragraph")
            for frame in content.findall(".//draw:frame", NS)
        )
        tables = len(content.findall(".//table:table", NS))
        equations = 0
        for name in archive.namelist():
            if name.startswith("Object") and name.endswith("content.xml"):
                formula = ET.fromstring(archive.read(name))
                if formula.tag == f"{{{NS['math']}}}math":
                    assert not formula.findall(".//math:merror", NS), name
                    equations += 1
    return {"paragraphs": paragraphs, "tables": tables, "equations": equations}


def compare_pages(approved: Path, exported: Path, folder: Path) -> list[dict]:
    """Compare extracted text, dimensions and exact Poppler RGB pixels per page."""
    folder.mkdir(parents=True, exist_ok=True)
    for label, pdf in (("approved", approved), ("exported", exported)):
        subprocess.run(["pdftoppm", "-png", "-r", "120", str(pdf), str(folder / label)], check=True)
    with fitz.open(approved) as original, fitz.open(exported) as regenerated:
        assert len(original) == len(regenerated), approved
        records = []
        for index, (left, right) in enumerate(zip(original, regenerated, strict=True), 1):
            # Poppler pads page numbers according to the page count.
            suffix = str(index).zfill(len(str(len(original))))
            with Image.open(folder / f"approved-{suffix}.png") as a, Image.open(folder / f"exported-{suffix}.png") as b:
                a, b = a.convert("RGB"), b.convert("RGB")
                assert a.size == b.size, (approved, index)
                difference = ImageChops.difference(a, b)
                bbox = difference.getbbox()
                if bbox:
                    difference.save(folder / f"difference-{suffix}.png")
                records.append(
                    {
                        "page": index,
                        "dimensions_equal": left.rect == right.rect,
                        "text_equal": left.get_text() == right.get_text(),
                        "pixels_equal": bbox is None,
                        "difference_bbox": bbox,
                        "pixel_size": a.size,
                        "approved_rgb_sha256": hashlib.sha256(a.tobytes()).hexdigest(),
                        "exported_rgb_sha256": hashlib.sha256(b.tobytes()).hexdigest(),
                    }
                )
    return records


def main() -> None:
    """Validate baseline exports, edited exports and preserved repository documents."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--approved-receipt", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--edited", required=True, type=Path)
    parser.add_argument("--editability", required=True, type=Path)
    parser.add_argument("--pixels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    inventory = load(args.inventory)
    manifest = load(args.manifest)["documents"]
    approved_records = load(args.approved_receipt)["documents"]
    edits = {item["id"]: item for item in load(args.editability)}
    baseline_receipt = load(args.baseline / "exportacion_documentos.json")
    edited_receipt = load(args.edited / "exportacion_documentos.json")
    assert len(manifest) == len(edits) == len(approved_records) == 8
    assert len(baseline_receipt["documents"]) == len(edited_receipt["documents"]) == 8
    assert baseline_receipt["renderer_image_id"] == edited_receipt["renderer_image_id"]
    assert baseline_receipt["renderer_image_id"] == load(args.approved_receipt)["renderer_image_id"]
    preserved = []
    for item in inventory["documents"]:
        path = Path(item["path"])
        assert digest(path) == item["sha256"], path
        record = dict(item, unchanged_after_exports=True)
        if path.suffix == ".pdf":
            with fitz.open(path) as pdf:
                record["pages"] = len(pdf)
        preserved.append(record)
    assert sum(Path(item["path"]).suffix == ".pdf" for item in preserved) == 18
    assert sum(Path(item["path"]).suffix == ".odt" for item in preserved) == 8
    documents = []
    for item in manifest:
        master = Path(item["output"])
        approved = master.with_suffix(".pdf")
        exported = args.baseline / approved.relative_to("parcial")
        historical = [r for r in approved_records if Path(r["odt"]).name == master.name]
        assert len(historical) == 1, master
        assert digest(master) == historical[0]["odt_sha256_after"], master
        assert digest(approved) == historical[0]["pdf_sha256"], approved
        assert Path(item["source"]).is_file(), item["source"]
        counts = native_counts(master)
        edit = edits[item["id"]]
        assert counts["tables"] == edit["writer_tables"]
        assert counts["equations"] == edit["writer_equations"]
        edited_pdf = args.edited / approved.name
        with fitz.open(edited_pdf) as pdf:
            edited_text = "".join(page.get_text() for page in pdf)
        markers = {"paragraph": "QAEDIT" in edited_text}
        if edit["table_cell_edited"] is True:
            markers["table_cell"] = "QACELL" in edited_text
        if edit["equation_edited"] is True:
            markers["equation"] = "QATEST" in edited_text
        assert all(markers.values()), edited_pdf
        assert edit["master_unchanged"] and all(edit["edited_markers_visible_in_export"].values())
        with fitz.open(approved) as pdf, fitz.open(exported) as new_pdf:
            assert len(pdf) == item["pages"], approved
            authors = None
            references = None
            if item["id"] in ("mri", "inundaciones", "fauna"):
                authors = {name: normalized(name) in normalized(pdf[0].get_text()) for name in AUTHORS}
                references = "referencias" in normalized(pdf[-1].get_text())
                assert all(authors.values()) and references, approved
            metadata = {"approved": pdf.metadata, "exported": new_pdf.metadata}
        pages = compare_pages(approved, exported, args.pixels / item["id"])
        documents.append(
            {
                "id": item["id"],
                "master": str(master),
                "approved_pdf": str(approved),
                "master_sha256": digest(master),
                "approved_pdf_sha256": digest(approved),
                "exported_pdf_sha256": digest(exported),
                "binary_pdf_equal": digest(approved) == digest(exported),
                "native_objects": counts,
                "authors_on_first_page": authors,
                "references_on_last_page": references,
                "edited_markers_visible_in_published_export": markers,
                "pdf_metadata": metadata,
                "pages": pages,
            }
        )
    totals = {key: sum(d["native_objects"][key] for d in documents) for key in ("paragraphs", "tables", "equations")}
    pages = [page for document in documents for page in document["pages"]]
    passed = len(pages) == 67 and all(p["dimensions_equal"] and p["text_equal"] and p["pixels_equal"] for p in pages)
    result = {
        "phase": "F",
        "base_sha": inventory["base_sha"],
        "renderer_image_id": baseline_receipt["renderer_image_id"],
        "rendering": "Poppler pdftoppm PNG, 120 dpi, exact RGB equality",
        "script_sha256": digest(Path(__file__)),
        "status": "passed" if passed else "failed",
        "totals": dict(totals, final_pages=len(pages), preserved_odt=8, preserved_pdf=18),
        "preserved_documents": preserved,
        "documents": documents,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    assert totals == {"paragraphs": 574, "tables": 57, "equations": 15}, totals
    assert passed, "Review page differences recorded in the receipt and temporary pixel directory"
    assert all(record["odt_preserved"] for record in baseline_receipt["documents"] + edited_receipt["documents"])
    print(json.dumps({"status": result["status"], "totals": result["totals"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
