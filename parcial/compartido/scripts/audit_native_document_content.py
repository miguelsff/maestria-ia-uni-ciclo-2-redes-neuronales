"""Compare every native paragraph and table cell against complete PDF sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

import pymupdf as fitz
from initialize_exam_documents import extract_document

NS = {
    "draw": "urn:oasis:names:tc:opendocument:xmlns:drawing:1.0",
    "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
    "table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
    "svg": "urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0",
}


def cleaned(value: str) -> str:
    """Resolve verified T1 glyphs and normalize whitespace only."""
    value = value.replace("\x1c", "fi").replace("\x15", "-")
    return re.sub(r"\s+", " ", value).strip()


def flattened(value: str) -> str:
    """Ignore whitespace and case when checking visible editorial context."""
    return "".join(cleaned(value).casefold().split())


def native_text(element: ET.Element) -> str:
    """Read ODF text including explicit spaces and line breaks."""
    result = element.text or ""
    for child in element:
        if child.tag == f"{{{NS['text']}}}line-break":
            result += "\n"
        elif child.tag == f"{{{NS['text']}}}s":
            result += " " * int(child.get(f"{{{NS['text']}}}c", "1"))
        elif child.tag == f"{{{NS['text']}}}tab":
            result += "\t"
        else:
            result += native_text(child)
        result += child.tail or ""
    return result


def source_paragraph(item: dict) -> str:
    """Reconstruct the full expected native paragraph from source spans."""
    lines = []
    for line in item["lines"]:
        text = ""
        previous = None
        for span in line["spans"]:
            if (
                previous
                and span["bbox"][0] - previous["bbox"][2] > 1.2
                and not previous["text"].endswith(" ")
                and not span["text"].startswith(" ")
            ):
                text += " "
            text += span["text"]
            previous = span
        lines.append(text)
    return cleaned("\n".join(lines))


def audit(item: dict, assets: Path) -> dict:
    """Check native text completeness, table identity and output page count."""
    expected = extract_document(item, assets)
    master = Path(item["output"])
    with zipfile.ZipFile(master) as archive:
        root = ET.fromstring(archive.read("content.xml"))
        math_objects = [
            ET.fromstring(archive.read(name))
            for name in archive.namelist()
            if name.startswith("Object ") and name.endswith("/content.xml")
        ]
    math_ns = "{http://www.w3.org/1998/Math/MathML}"
    formulas = [
        cleaned(node.text or "")
        for obj in math_objects
        for node in obj.iter(math_ns + "annotation")
        if node.get("encoding", "").startswith("StarMath")
    ]
    expected_formulas = [cleaned(eq["starmath"]) for eq in item.get("equations", [])]
    formula_errors = Counter(expected_formulas) != Counter(formulas) or any(
        list(obj.iter(math_ns + "merror")) for obj in math_objects
    )
    frames = {f.get(f"{{{NS['draw']}}}name"): f for f in root.findall(".//draw:frame", NS)}
    paragraphs = [p for page in expected["page_objects"] for p in page["paragraphs"]]
    tables = [t for page in expected["page_objects"] for t in page["tables"]]
    mismatches = []
    updates = {update["paragraph"]: update for update in item.get("editorial_updates", [])}
    expected_paragraphs = []
    for index, paragraph in enumerate(paragraphs, 1):
        frame = frames[f"Paragraph{index}"]
        actual = cleaned(native_text(frame.find("draw:text-box/text:p", NS)))
        wanted = source_paragraph(paragraph)
        if index in updates:
            update = updates[index]
            if wanted != cleaned(update["source_text"]):
                raise ValueError("The explicitly approved editorial source changed")
            wanted = cleaned(update["replacement"])
        expected_paragraphs.append(wanted)
        if actual != wanted:
            mismatches.append({"paragraph": index, "expected": wanted, "actual": actual})
    cell_errors = []
    for index, table in enumerate(tables, 1):
        actual_table = root.find(f".//table:table[@table:name='Table{index}']", NS)
        rows = []
        for row in actual_table.findall("table:table-row", NS):
            values = []
            for cell in row.findall("table:table-cell", NS):
                values.append(cleaned(native_text(cell)))
                values.extend([""] * (int(cell.get(f"{{{NS['table']}}}number-columns-spanned", "1")) - 1))
            rows.append(values)
        wanted = [[cleaned(value) for value in row] for row in table["rows"]]
        if rows != wanted:
            cell_errors.append({"table": index, "expected": wanted, "actual": rows})
    target = fitz.open(master.with_suffix(".pdf"))
    editorial_context = []
    for index, update in updates.items():
        page_number = int(frames[f"Paragraph{index}"].get(f"{{{NS['text']}}}anchor-page-number"))
        editorial_context.append(
            {
                "paragraph": index,
                "page": page_number,
                "entire_replacement_visible_on_its_page": flattened(update["replacement"])
                in flattened(target[page_number - 1].get_text()),
            }
        )
    expected_text = (
        "\n".join(expected_paragraphs)
        + "\n"
        + "\n".join(value for table in tables for row in table["rows"] for value in row)
    )
    visible_text = "\n".join(page.get_text() for page in target)

    def tokens(value: str) -> Counter:
        return Counter(
            token for token in re.findall(r"[^\W_]+(?:_[^\W_]+)*", cleaned(value).casefold()) if len(token) >= 3
        )

    missing = tokens(expected_text) - tokens(visible_text)
    return {
        "id": item["id"],
        "source": item["source"],
        "odt": str(master),
        "source_sha256": hashlib.sha256(Path(item["source"]).read_bytes()).hexdigest(),
        "odt_sha256": hashlib.sha256(master.read_bytes()).hexdigest(),
        "pages_source": expected["pages"],
        "pages_exported": len(target),
        "paragraphs_compared": len(paragraphs),
        "paragraph_mismatches": mismatches,
        "tables_compared": len(tables),
        "table_mismatches": cell_errors,
        "source_to_odt_complete": not mismatches and not cell_errors,
        "recorded_editorial_updates": len(updates),
        "editorial_context_checks": editorial_context,
        "editorial_context_visible": all(
            check["entire_replacement_visible_on_its_page"] for check in editorial_context
        ),
        "native_mathml_objects": len(math_objects),
        "formula_annotations": formulas,
        "formulas_match_manifest_without_parser_errors": not formula_errors,
        "visible_pdf_missing_tokens": dict(missing),
        "odt_to_pdf_token_coverage_complete": not missing,
    }


def main() -> None:
    """Audit the complete initialization manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.manifest.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="document-content-", dir="tmp") as folder:
        results = [audit(item, Path(folder)) for item in config["documents"]]
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            [
                {k: v for k, v in result.items() if k not in ["paragraph_mismatches", "table_mismatches"]}
                for result in results
            ],
            ensure_ascii=False,
        )
    )
    if any(
        not result["source_to_odt_complete"]
        or not result["odt_to_pdf_token_coverage_complete"]
        or not result["formulas_match_manifest_without_parser_errors"]
        or not result["editorial_context_visible"]
        or result["pages_source"] != result["pages_exported"]
        for result in results
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
