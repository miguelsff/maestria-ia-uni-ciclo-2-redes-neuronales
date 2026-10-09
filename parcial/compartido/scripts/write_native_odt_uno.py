"""Initialize Writer-native frames, tables and formula objects from a payload."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import uno
from com.sun.star.beans import PropertyValue
from com.sun.star.style import LineSpacing
from com.sun.star.text.TextContentAnchorType import AT_PAGE

PT_TO_MM100 = 2540 / 72


def prop(name: str, value: object) -> PropertyValue:
    """Create a UNO named argument."""
    item = PropertyValue()
    item.Name, item.Value = name, value
    return item


def mm(value: float) -> int:
    """Convert PDF points to Writer's hundredths of millimeters."""
    return round(value * PT_TO_MM100)


def connect(profile: Path) -> tuple:
    """Launch an isolated headless Writer instance owned by this process."""
    executable = "soffice"
    proc = subprocess.Popen(
        [
            executable,
            f"-env:UserInstallation={profile.as_uri()}",
            "--headless",
            "--accept=socket,host=localhost,port=2094;urp;StarOffice.ComponentContext",
            "--norestore",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    local = uno.getComponentContext()
    resolver = local.ServiceManager.createInstanceWithContext("com.sun.star.bridge.UnoUrlResolver", local)
    for _ in range(100):
        try:
            ctx = resolver.resolve("uno:socket,host=localhost,port=2094;urp;StarOffice.ComponentContext")
            desktop = ctx.ServiceManager.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)
            return proc, desktop
        except Exception:
            time.sleep(0.1)
    proc.terminate()
    raise RuntimeError("The isolated Writer instance did not start")


def frame(doc: object, page: int, bbox: list, name: str) -> object:
    """Create a Writer TextFrame rather than a Draw text shape."""
    result = doc.createInstance("com.sun.star.text.TextFrame")
    result.Name = name
    result.AnchorType, result.AnchorPageNo = AT_PAGE, page
    result.HoriOrient = result.VertOrient = 0
    result.HoriOrientRelation = result.VertOrientRelation = 7
    result.HoriOrientPosition, result.VertOrientPosition = mm(bbox[0]), mm(bbox[1])
    result.Width, result.Height = mm(bbox[2] - bbox[0]), mm(bbox[3] - bbox[1])
    result.SizeType = 1
    result.Surround = uno.Enum("com.sun.star.text.WrapTextMode", "THROUGH")
    result.LeftMargin = result.RightMargin = result.TopMargin = result.BottomMargin = 0
    result.LeftBorderDistance = result.RightBorderDistance = result.TopBorderDistance = result.BottomBorderDistance = 0
    border = uno.createUnoStruct("com.sun.star.table.BorderLine2")
    for side in ["LeftBorder", "RightBorder", "TopBorder", "BottomBorder"]:
        result.setPropertyValue(side, border)
    result.BackColor = -1
    result.BackTransparent = True
    result.FillStyle = uno.Enum("com.sun.star.drawing.FillStyle", "NONE")
    doc.Text.insertTextContent(doc.Text.End, result, False)
    return result


def insert_paragraph(doc: object, page: int, item: dict, idx: int) -> None:
    """Write a rich, editable paragraph with original line breaks."""
    box = frame(doc, page, item["bbox"], f"Paragraph{idx}")
    cursor = box.Text.createTextCursor()
    cursor.CharFontName = "Liberation Serif"
    cursor.CharHeight = item["font_size"]
    spacing = LineSpacing()
    spacing.Mode, spacing.Height = 3, mm(item["line_height"])
    cursor.ParaLineSpacing = spacing
    cursor.ParaTopMargin = cursor.ParaBottomMargin = 0
    cursor.ParaFirstLineIndent = mm(item.get("first_line_indent", 0))
    if item.get("center"):
        cursor.ParaAdjust = 3
    for line_index, line in enumerate(item["lines"]):
        if line_index:
            box.Text.insertControlCharacter(cursor, 1, False)
        previous = None
        baseline = max(line["spans"], key=lambda span: span["size"])["origin"][1]
        for span in line["spans"]:
            if (
                previous
                and span["bbox"][0] - previous["bbox"][2] > 1.2
                and not previous["text"].endswith(" ")
                and not span["text"].startswith(" ")
            ):
                gap = span["bbox"][0] - previous["bbox"][2]
                spaces = max(1, round(gap / (span["size"] * 0.24))) if gap > 12 else 1
                box.Text.insertString(cursor, " " * spaces, False)
            cursor.CharFontName = span["family"]
            cursor.CharHeight = span["size"]
            cursor.CharWeight = 150.0 if span["flags"] & 16 else 100.0
            cursor.CharPosture = uno.Enum("com.sun.star.awt.FontSlant", "ITALIC" if span["flags"] & 2 else "NONE")
            cursor.CharColor = span["color"]
            cursor.CharScaleWidth = 96
            cursor.CharEscapementHeight = 100
            cursor.CharEscapement = round((baseline - span["origin"][1]) / span["size"] * 100)
            text = span["text"].replace("\x1c", "fi").replace("\x15", "-")
            if any(ord(char) < 32 for char in text):
                raise ValueError("An unresolved PDF control glyph would discard document text")
            box.Text.insertString(cursor, text, False)
            previous = span


def insert_table(doc: object, page: int, item: dict, idx: int) -> None:
    """Write a real TextTable into a Writer frame."""
    box = frame(doc, page, item["bbox"], f"TableFrame{idx}")
    table = doc.createInstance("com.sun.star.text.TextTable")
    rows = item["rows"]
    table.initialize(len(rows), len(rows[0]))
    table.Name = f"Table{idx}"
    box.Text.insertTextContent(box.Text.End, table, False)
    table.LeftMargin = table.RightMargin = table.TopMargin = table.BottomMargin = 0
    table.HoriOrient = 0
    table.Width = box.Width
    separators = table.TableColumnSeparators
    total = sum(item["widths"])
    accumulated = 0
    for col, separator in enumerate(separators):
        accumulated += item["widths"][col]
        separator.Position = round(accumulated / total * 10000)
        separator.IsVisible = True
    table.TableColumnSeparators = separators
    first_values = [row[0] for row in rows]
    target = None
    if any("medium_1_to_5pct" in value for value in first_values):
        target = 4500
    elif any("ResNet/" in value or "gold_weak" in value for value in first_values):
        target = min(separators[0].Position + 750, 4000)
    if target is not None:
        first = separators[0].Position
        for index, separator in enumerate(separators):
            separator.Position = (
                round(target + (separator.Position - first) * (10000 - target) / (10000 - first)) if index else target
            )
        table.TableColumnSeparators = separators
    if "Clases con soporte" in rows[0] and len(separators) == 3:
        separators = table.TableColumnSeparators
        separators[1].Position += 500
        separators[2].Position += 250
        table.TableColumnSeparators = separators
    border = uno.createUnoStruct("com.sun.star.table.TableBorder2")
    for side in ["Top", "Bottom", "Left", "Right", "Horizontal", "Vertical"]:
        setattr(border, "Is" + side + "LineValid", True)
    table.TableBorder2 = border
    for row_index, values in enumerate(rows):
        row = table.Rows.getByIndex(row_index)
        row.IsAutoHeight = False
        row.Height = mm(item["heights"][row_index])
        for col_index, value in enumerate(values):
            cell = table.getCellByPosition(col_index, row_index)
            cell.String = value
            cell.LeftBorderDistance = cell.RightBorderDistance = cell.TopBorderDistance = cell.BottomBorderDistance = 0
            cursor = cell.createTextCursor()
            cursor.gotoStart(False)
            cursor.gotoEnd(True)
            cursor.CharFontName = "Liberation Serif"
            cursor.CharHeight = item["font_size"]
            cursor.CharWeight = 100.0
            cursor.CharScaleWidth = 96
            cursor.ParaTopMargin = cursor.ParaBottomMargin = 0
            cursor.ParaAdjust = 0 if col_index == 0 else 3
            spacing = LineSpacing()
            spacing.Mode, spacing.Height = 3, mm(item["font_size"] * 1.05)
            cursor.ParaLineSpacing = spacing
    for row_index in item.get("merged_rows", []):
        start = f"A{row_index + 1}"
        end = f"{chr(65 + len(rows[0]) - 1)}{row_index + 1}"
        cursor = table.createCursorByCellName(start)
        cursor.gotoCellByName(end, True)
        cursor.mergeRange()
        cell = table.getCellByName(start)
        cell.String = rows[row_index][0]
        cursor = cell.createTextCursor()
        cursor.ParaAdjust = 3
        cursor.CharFontName = "Liberation Serif"
        cursor.CharHeight = item["font_size"]


def write_document(desktop: object, item: dict) -> dict:
    """Create and save a single protected initial ODT master."""
    output = Path(item["output"]).resolve()
    if output.exists():
        raise FileExistsError(f"Existing master is protected: {output}")
    doc = desktop.loadComponentFromURL("private:factory/swriter", "_blank", 0, (prop("Hidden", True),))
    page_style = doc.StyleFamilies.getByName("PageStyles").getByName("Standard")
    page_style.Width, page_style.Height = mm(item["page_size"][0]), mm(item["page_size"][1])
    page_style.TopMargin = page_style.BottomMargin = page_style.LeftMargin = page_style.RightMargin = 0
    for _ in range(item["pages"] - 1):
        cursor = doc.Text.createTextCursor()
        cursor.gotoEnd(False)
        doc.Text.insertControlCharacter(cursor, 0, False)
        cursor.BreakType = uno.Enum("com.sun.star.style.BreakType", "PAGE_BEFORE")
    counts = {"paragraphs": 0, "tables": 0, "mathml_objects": 0, "figures": 0}
    for page_index, page in enumerate(item["page_objects"], 1):
        for paragraph in page["paragraphs"]:
            counts["paragraphs"] += 1
            for shift in item.get("layout_adjustments", {}).get("paragraph_shifts", []):
                if page_index == shift["page"] and shift["y_min"] <= paragraph["bbox"][1] < shift["y_max"]:
                    paragraph = {**paragraph, "bbox": list(paragraph["bbox"])}
                    paragraph["bbox"][1] += shift["dy"]
                    paragraph["bbox"][3] += shift["dy"]
            insert_paragraph(doc, page_index, paragraph, counts["paragraphs"])
        for table in page["tables"]:
            counts["tables"] += 1
            for shift in item.get("layout_adjustments", {}).get("paragraph_shifts", []):
                if page_index == shift["page"] and shift["y_min"] <= table["bbox"][1] < shift["y_max"]:
                    table = {**table, "bbox": list(table["bbox"])}
                    table["bbox"][1] += shift["dy"]
                    table["bbox"][3] += shift["dy"]
            insert_table(doc, page_index, table, counts["tables"])
        for equation in page["equations"]:
            obj = doc.createInstance("com.sun.star.text.TextEmbeddedObject")
            obj.CLSID = "078B7ABA-54FC-457F-8551-6147e776a997"
            obj.AnchorType, obj.AnchorPageNo = AT_PAGE, page_index
            obj.HoriOrient = obj.VertOrient = 0
            obj.HoriOrientRelation = obj.VertOrientRelation = 7
            obj.Surround = uno.Enum("com.sun.star.text.WrapTextMode", "THROUGH")
            doc.Text.insertTextContent(doc.Text.End, obj, False)
            obj.Component.BaseFontHeight = equation.get("font_size", 9)
            obj.Component.IsTextMode = equation.get("inline", False)
            obj.Component.Formula = equation["starmath"]
            bbox = equation["bbox"]
            obj.HoriOrientPosition = mm(bbox[0])
            obj.VertOrientPosition = mm(bbox[1] + equation.get("vertical_shift", 0))
            # Math computes its own size. Resizing an embedded formula scales
            # its glyphs and can make a fraction collide with adjacent text.
            natural = obj.Component.getVisualAreaSize(1)
            obj.Width, obj.Height = natural.Width, natural.Height
            counts["mathml_objects"] += 1
        for image in page["images"]:
            obj = doc.createInstance("com.sun.star.text.TextGraphicObject")
            obj.AnchorType, obj.AnchorPageNo = AT_PAGE, page_index
            obj.HoriOrient = obj.VertOrient = 0
            obj.HoriOrientRelation = obj.VertOrientRelation = 7
            obj.Surround = uno.Enum("com.sun.star.text.WrapTextMode", "THROUGH")
            obj.GraphicURL = Path(image["path"]).resolve().as_uri()
            doc.Text.insertTextContent(doc.Text.End, obj, False)
            bbox = image["bbox"]
            obj.HoriOrientPosition, obj.VertOrientPosition = mm(bbox[0]), mm(bbox[1])
            obj.Width, obj.Height = mm(bbox[2] - bbox[0]), mm(bbox[3] - bbox[1])
            counts["figures"] += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    doc.storeAsURL(output.as_uri(), (prop("FilterName", "writer8"),))
    actual = {
        "writer_text_frames": doc.TextFrames.Count,
        "writer_tables": doc.TextTables.Count,
        "writer_embedded_objects": doc.EmbeddedObjects.Count,
    }
    doc.close(True)
    # Reopen after Math has finished its layout. Fresh embedded objects can
    # expose their initial placeholder size until the document is stored.
    doc = desktop.loadComponentFromURL(output.as_uri(), "_blank", 0, (prop("Hidden", True),))
    for index in range(doc.EmbeddedObjects.Count):
        obj = doc.EmbeddedObjects.getByIndex(index)
        natural = obj.Component.getVisualAreaSize(1)
        obj.Width, obj.Height = natural.Width, natural.Height
    doc.store()
    doc.close(True)
    return {"odt": str(output), "counts": counts, **actual}


def main() -> None:
    """Create documents inside one isolated Writer session."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("payload", type=Path)
    args = parser.parse_args()
    config = json.loads(args.payload.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="exam-odt-") as folder:
        process, desktop = connect(Path(folder))
        try:
            results = [write_document(desktop, item) for item in config["documents"]]
            Path(config["receipt"]).write_text(
                json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps(results, ensure_ascii=False))
        finally:
            desktop.terminate()
            process.wait(timeout=15)


if __name__ == "__main__":
    main()
