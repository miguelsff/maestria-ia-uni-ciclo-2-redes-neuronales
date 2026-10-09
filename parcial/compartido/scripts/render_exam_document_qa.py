"""Render all exported PDF pages with Poppler and build review contact sheets."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    """Keep full page images temporary and archive small review sheets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--pages", type=Path, required=True)
    parser.add_argument("--sheets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.pages.mkdir(parents=True, exist_ok=True)
    args.sheets.mkdir(parents=True, exist_ok=True)
    documents = json.loads(args.manifest.read_text(encoding="utf-8"))["documents"]
    receipts = []
    for document in documents:
        pdf = Path(document["output"]).with_suffix(".pdf")
        prefix = args.pages / document["id"]
        subprocess.run(["pdftoppm", "-png", "-r", "120", str(pdf), str(prefix)], check=True)
        pages = sorted(args.pages.glob(document["id"] + "-*.png"), key=lambda p: int(p.stem.rsplit("-", 1)[1]))
        if len(pages) != document["pages"]:
            raise ValueError("The rendered page count differs from the complete source")
        sheets = []
        for offset in range(0, len(pages), 4):
            canvas = Image.new("RGB", (1560, 2100), "#ececec")
            draw = ImageDraw.Draw(canvas)
            selected = pages[offset : offset + 4]
            for position, page in enumerate(selected):
                image = Image.open(page).convert("RGB")
                image.thumbnail((750, 990), Image.Resampling.LANCZOS)
                x, y = 15 + (position % 2) * 780, 40 + (position // 2) * 1040
                canvas.paste(image, (x, y))
                draw.text((x, y - 25), f"{document['id']} / {offset + position + 1}", fill="#111111")
            target = args.sheets / f"{document['id']}-{offset // 4 + 1:02}.png"
            canvas.save(target, optimize=True)
            sheets.append(
                {
                    "path": target.as_posix(),
                    "pages": list(range(offset + 1, offset + len(selected) + 1)),
                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                }
            )
        receipts.append(
            {
                "id": document["id"],
                "pdf": pdf.as_posix(),
                "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
                "rendered_pages": len(pages),
                "renderer": "Poppler pdftoppm 120 dpi",
                "sheets": sheets,
            }
        )
    args.output.write_text(json.dumps(receipts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"documents": len(receipts), "rendered_pages": sum(r["rendered_pages"] for r in receipts)}))


if __name__ == "__main__":
    main()
