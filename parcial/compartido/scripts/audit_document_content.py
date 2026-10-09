"""Compare source PDF, native ODT XML and exported PDF without writing documents."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

import pymupdf
from pypdf import PdfReader

TEXT_NS = "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}"
CONTROL_GLYPHS = {"\x1c": "fi", "\x15": "-"}


def normalize(text: str) -> str:
    """Normalize validated Computer Modern glyph codes and Unicode ligatures."""
    for glyph, replacement in CONTROL_GLYPHS.items():
        text = text.replace(glyph, replacement)
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    return text


def tokens(text: str) -> Counter:
    """Ignore reading order while retaining word and numeric multiplicities."""
    return Counter(re.findall(r"[^\W_]+(?:[.,][0-9]+)?", normalize(text), re.UNICODE))


def xml_text(element: ET.Element) -> str:
    """Honor explicit ODF spaces, tabs and line breaks between rich-text spans."""
    result = element.text or ""
    for child in element:
        if child.tag == TEXT_NS + "s":
            result += " " * int(child.get(TEXT_NS + "c", "1"))
        elif child.tag == TEXT_NS + "tab":
            result += "\t"
        elif child.tag == TEXT_NS + "line-break":
            result += "\n"
        else:
            result += xml_text(child)
        result += child.tail or ""
    return result


def odt_text(path: Path) -> tuple[str, dict]:
    """Inspect native XML, including Writer frames/tables and embedded math text."""
    with zipfile.ZipFile(path) as archive:
        tree = ET.fromstring(archive.read("content.xml"))
        paragraphs = [xml_text(e) for e in tree.iter() if e.tag in {TEXT_NS + "p", TEXT_NS + "h"}]
        embedded = []
        for name in archive.namelist():
            if name.endswith("/content.xml"):
                embedded.append(" ".join(ET.fromstring(archive.read(name)).itertext()))
    return "\n".join(paragraphs + embedded), {"paragraphs": len(paragraphs), "embedded_xml_objects": len(embedded)}


def pdf_text(path: Path) -> tuple[str, list[dict], str]:
    """Keep source lines, control glyph locations and an independent extraction."""
    doc = pymupdf.open(path)
    lines = []
    for page_number, page in enumerate(doc, 1):
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                spans = line["spans"]
                text = "".join(s["text"] for s in spans)
                lines.append({"page": page_number, "bbox": list(line["bbox"]), "text": text,
                              "controls": sorted({f"U+{ord(c):04X}" for c in text if ord(c) < 32}),
                              "fonts": sorted({s["font"] for s in spans})})
    second = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return "\n".join(line["text"] for line in lines), lines, second


def comparison(source: str, target: str) -> dict:
    """Report coverage as a diagnostic, never as automatic editorial approval."""
    original, received = tokens(source), tokens(target)
    missing = original - received
    return {"source_tokens": sum(original.values()), "missing_token_instances": sum(missing.values()),
            "missing_tokens": dict(missing), "order_independent_token_coverage":
            1 - sum(missing.values()) / max(1, sum(original.values())),
            "whitespace_insensitive_character_multiset_loss": dict(Counter(c for c in normalize(source) if c.isalnum())
                                                                 - Counter(c for c in normalize(target) if c.isalnum()))}


def audit(source: Path, odt: Path, exported: Path) -> dict:
    """Separate source extraction, XML omissions and PDF-only rendering candidates."""
    inputs = (("source", source), ("odt", odt), ("pdf", exported))
    before = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in inputs}
    source_text, lines, source_second = pdf_text(source)
    native, structure = odt_text(odt)
    output, _, output_second = pdf_text(exported)
    native_flat = "".join(normalize(native).split())
    output_flat = "".join(normalize(output).split())
    absent_lines = []
    for line in lines:
        flat = "".join(normalize(line["text"]).split())
        if flat.endswith("-"):
            flat = flat[:-1]  # A source line may end halfway through a hyphenated word.
        if len(flat) < 25:
            continue
        if flat not in native_flat or flat not in output_flat:
            absent_lines.append({**line, "present_in_odt_xml": flat in native_flat,
                                 "present_in_exported_pdf": flat in output_flat,
                                 "classification": "xml_or_order_candidate" if flat not in native_flat else "render_or_pdf_extraction_candidate"})
    hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in inputs}
    if hashes != before:
        raise ValueError("A document changed during inspection; use a stable read-only snapshot.")
    return {"hashes": hashes, "structure": structure, "source_to_odt": comparison(source_text, native),
            "source_to_pdf": comparison(source_text, output), "odt_to_pdf": comparison(native, output),
            "source_extractor_difference": comparison(source_text, source_second),
            "exported_extractor_difference": comparison(output, output_second),
            "source_control_lines": [line for line in lines if line["controls"]], "line_candidates": absent_lines,
            "limitations": "Token/line differences require inspection: formula objects, cell order, line wrapping and "
                           "font encodings can change extraction. Character multisets are an additional loss indicator, "
                           "not proof of semantic preservation. Known control maps apply to the inspected Computer Modern sources."}


def main() -> None:
    """Read a diagnostic manifest and write receipts only in its own output folder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    results = {}
    for item in manifest["documents"]:
        results[item["id"]] = audit(Path(item["source"]), Path(item["output"]), Path(item["output"]).with_suffix(".pdf"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: {"xml_missing": value["source_to_odt"]["missing_token_instances"],
                           "pdf_missing": value["source_to_pdf"]["missing_token_instances"]}
                      for key, value in results.items()}))


if __name__ == "__main__":
    main()
