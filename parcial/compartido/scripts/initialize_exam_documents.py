"""Initialize complete editable ODT masters once, separately from export.

Run inside the managed renderer with the repository mounted at /work. Existing
ODT masters are protected; the regular exporter never calls this initializer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import tempfile
from pathlib import Path

import pymupdf as fitz
from extract_exam_document import font_family, table_data, table_regions


def extract_document(source: dict, assets: Path) -> dict:
    """Preserve the full PDF as native paragraphs, cells and formula objects."""
    pdf = fitz.open(source["source"])
    document = {**source, "pages": len(pdf), "page_size": list(pdf[0].rect)[2:], "page_objects": []}
    for page_number, page in enumerate(pdf, 1):
        ruled_tables = page.find_tables().tables
        regions = []
        for table in ruled_tables:
            rect = fitz.Rect(table.bbox)
            if rect.width > 90 and rect.height > 15 and not any(r.intersects(rect) for r in regions):
                regions.append(rect)
        regions += [rect for rect in table_regions(page) if not any(rect.intersects(other) for other in regions)]
        equations = [e for e in source.get("equations", []) if e["page"] == page_number]
        formula_regions = [fitz.Rect(e["bbox"]) for e in equations]
        objects = {"paragraphs": [], "tables": [], "equations": equations, "images": []}
        for rect in regions:
            data, widths, heights = table_data(page, rect)
            merged_rows = []
            for table in ruled_tables:
                if abs(table.bbox[0] - rect.x0) < 1 and abs(table.bbox[1] - rect.y0) < 1:
                    cells = table.rows[0].cells
                    if len(cells) == len(widths) and all(cell for cell in cells):
                        widths = [cell[2] - cell[0] for cell in cells]
                    for row_index, row in enumerate(data):
                        values = [value for value in row if value]
                        if len(values) == 1 and len(values[0]) > 20:
                            data[row_index] = [values[0]] + [""] * (len(row) - 1)
                            merged_rows.append(row_index)
            if data:
                objects["tables"].append(
                    {
                        "bbox": list(rect),
                        "rows": data,
                        "widths": widths,
                        "heights": heights,
                        "font_size": 8.2,
                        "merged_rows": merged_rows,
                    }
                )
        for block in page.get_text("dict")["blocks"]:
            if block["type"] == 1:
                if fitz.Rect(block["bbox"]).get_area() / page.rect.get_area() > 0.82:
                    raise ValueError("A full-page image is not an editable document")
                path = assets / f"{source['id']}-p{page_number}-figure-{len(objects['images'])}.{block['ext']}"
                path.write_bytes(block["image"])
                objects["images"].append({"bbox": list(block["bbox"]), "path": str(path.resolve())})
                continue
            raw = []
            for line in block["lines"]:
                if any(
                    r.contains(
                        fitz.Rect(line["bbox"]).tl + (fitz.Rect(line["bbox"]).br - fitz.Rect(line["bbox"]).tl) / 2
                    )
                    for r in regions
                ):
                    continue
                spans = [
                    span
                    for span in line["spans"]
                    if not any(
                        r.contains(
                            fitz.Point((span["bbox"][0] + span["bbox"][2]) / 2, (span["bbox"][1] + span["bbox"][3]) / 2)
                        )
                        for r in formula_regions
                    )
                ]
                if spans:
                    bounds = fitz.Rect(spans[0]["bbox"])
                    for span in spans[1:]:
                        bounds |= fitz.Rect(span["bbox"])
                    raw.append({**line, "spans": spans, "bbox": list(bounds)})
            if not raw:
                continue
            lines = []
            for line in raw:
                if lines and abs(line["spans"][0]["origin"][1] - lines[-1]["spans"][0]["origin"][1]) < 5:
                    lines[-1]["spans"].extend(line["spans"])
                    lines[-1]["bbox"] = list(fitz.Rect(lines[-1]["bbox"]) | fitz.Rect(line["bbox"]))
                else:
                    lines.append({**line, "spans": list(line["spans"])})
            for line in lines:
                for span in line["spans"]:
                    span["family"] = font_family(span["font"])
                    if "CMEX" in span["font"]:
                        span["text"] = span["text"].replace("P", "∑")
            bbox = fitz.Rect(lines[0]["bbox"])
            for line in lines[1:]:
                bbox |= fitz.Rect(line["bbox"])
            size = max(span["size"] for line in lines for span in line["spans"])
            first = max(lines[0]["spans"], key=lambda span: span["size"])
            top = first["origin"][1] - size * 0.89
            line_height = (
                statistics.median(
                    [
                        b["spans"][0]["origin"][1] - a["spans"][0]["origin"][1]
                        for a, b in zip(lines, lines[1:], strict=False)
                    ]
                )
                if len(lines) > 1
                else size * 1.08
            )
            width = max(bbox.width + 8, 396 if bbox.x0 < 120 and bbox.x1 > 400 else bbox.width + 8)
            center = page_number == 1 and bbox.y0 < 300 and (bbox.x0 > 140 or size > 14)
            objects["paragraphs"].append(
                {
                    "bbox": [
                        bbox.x0,
                        top,
                        min(page.rect.width - 12, bbox.x0 + width),
                        top + max(bbox.height + 12, len(lines) * line_height + 12),
                    ],
                    "font_size": size,
                    "line_height": max(size, line_height),
                    "lines": lines,
                    "center": center,
                    "first_line_indent": max(0, lines[0]["bbox"][0] - bbox.x0),
                }
            )
        if source.get("merge_overlapping_paragraphs"):
            merged = []
            for paragraph in objects["paragraphs"]:
                if (
                    merged
                    and abs(merged[-1]["bbox"][0] - paragraph["bbox"][0]) < 1
                    and any(
                        abs(a["spans"][0]["origin"][1] - b["spans"][0]["origin"][1]) < 2
                        for a in merged[-1]["lines"]
                        for b in paragraph["lines"]
                    )
                ):
                    previous = merged[-1]
                    for line in paragraph["lines"]:
                        match = next(
                            (
                                a
                                for a in previous["lines"]
                                if abs(a["spans"][0]["origin"][1] - line["spans"][0]["origin"][1]) < 2
                            ),
                            None,
                        )
                        if match is None:
                            previous["lines"].append(line)
                        else:
                            match["spans"].extend(line["spans"])
                            match["spans"].sort(key=lambda s: s["bbox"][0])
                    previous["lines"].sort(key=lambda a: a["spans"][0]["origin"][1])
                    previous["bbox"][3] = max(previous["bbox"][3], paragraph["bbox"][3])
                else:
                    merged.append(paragraph)
            objects["paragraphs"] = merged
        document["page_objects"].append(objects)
    return document


def main() -> None:
    """Initialize an explicit manifest without overwriting editable masters."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    config = json.loads(args.manifest.read_text(encoding="utf-8"))
    for item in config["documents"]:
        if Path(item["output"]).exists():
            raise FileExistsError(f"Existing master is protected: {item['output']}")
    before = {
        item["source"]: hashlib.sha256(Path(item["source"]).read_bytes()).hexdigest() for item in config["documents"]
    }
    temp_root = Path("tmp")
    temp_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="odt-initialization-", dir=temp_root) as folder:
        assets = Path(folder)
        payload = {
            "documents": [extract_document(item, assets) for item in config["documents"]],
            "receipt": config["receipt"],
        }
        payload_path = assets / "payload.json"
        payload_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        backend = Path(__file__).with_name("write_native_odt_uno.py")
        subprocess.run(["/usr/bin/python3", str(backend), str(payload_path)], check=True)
    after = {
        item["source"]: hashlib.sha256(Path(item["source"]).read_bytes()).hexdigest() for item in config["documents"]
    }
    if before != after:
        raise RuntimeError("An immutable PDF source changed during initialization")


if __name__ == "__main__":
    main()
