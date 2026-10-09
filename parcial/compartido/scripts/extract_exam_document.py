"""Extract PDF paragraphs and native table cells without rasterizing pages."""

from __future__ import annotations

import statistics

import pymupdf as fitz


def font_family(font: str) -> str:
    """Map source PostScript families to registered renderer fonts."""
    if "Nimbus" in font or "Times" in font:
        return "Liberation Serif"
    if "Courier" in font or "Mono" in font or "CMTT" in font:
        return "Liberation Mono"
    if "CMEX" in font or "CMSY" in font:
        return "DejaVu Serif"
    if "CM" in font:
        return "Latin Modern Roman"
    if "Helvetica" in font or "Arial" in font:
        return "Liberation Sans"
    return "Liberation Serif"


def horizontal_rules(page: fitz.Page) -> list[tuple[float, float, float]]:
    """Find long horizontal table rules, excluding small fraction bars."""
    result = []
    for drawing in page.get_drawings():
        for item in drawing["items"]:
            if item[0] == "l":
                a, b = item[1:3]
                if abs(a.y - b.y) < 0.2 and abs(a.x - b.x) > 90:
                    result.append((min(a.x, b.x), max(a.x, b.x), a.y))
    return result


def table_regions(page: fitz.Page) -> list[fitz.Rect]:
    """Identify booktabs tables by their three matching horizontal rules."""
    groups: dict[tuple[int, int], list[float]] = {}
    for x0, x1, y in horizontal_rules(page):
        groups.setdefault((round(x0), round(x1)), []).append(y)
    regions = []
    for (x0, x1), ys in groups.items():
        ys = sorted(set(ys))
        while len(ys) >= 3:
            top, middle, bottom = ys[:3]
            if middle - top < 45 and bottom - middle > 7:
                regions.append(fitz.Rect(x0, top, x1, bottom))
                ys = ys[3:]
            else:
                ys = ys[1:]
    return regions


def table_data(page: fitz.Page, rect: fitz.Rect) -> tuple[list[list[str]], list[float], list[float]]:
    """Reconstruct native cells using line positions and body column centers."""
    lines = []
    for block in page.get_text("dict", clip=rect)["blocks"]:
        for line in block.get("lines", []):
            text = "".join(s["text"] for s in line["spans"]).strip()
            if text:
                lines.append((line["bbox"], text, line["spans"][0]["origin"][1]))
    rows: list[list[tuple]] = []
    for item in sorted(lines, key=lambda v: (v[2], v[0][0])):
        if not rows or abs(item[2] - statistics.mean(x[2] for x in rows[-1])) > 3:
            rows.append([])
        rows[-1].append(item)
    if not rows:
        return [], [], []
    # The row containing the most separate cells is the table's column template.
    template = max(rows, key=len)
    centers = sorted((item[0][0] + item[0][2]) / 2 for item in template)
    bounds = [rect.x0] + [(a + b) / 2 for a, b in zip(centers, centers[1:], strict=False)] + [rect.x1]
    data = []
    for row in rows:
        cells = [""] * len(centers)
        for bbox, text, _ in row:
            center = (bbox[0] + bbox[2]) / 2
            col = min(range(len(centers)), key=lambda i: abs(centers[i] - center))
            cells[col] = (cells[col] + " " + text).strip()
        data.append(cells)
    row_height = rect.height / len(rows)
    return data, [b - a for a, b in zip(bounds, bounds[1:], strict=False)], [row_height] * len(rows)
