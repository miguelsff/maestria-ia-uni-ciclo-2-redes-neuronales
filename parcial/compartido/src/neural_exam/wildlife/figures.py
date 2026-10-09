"""Recorded YOLO learning curves and traceable qualitative contact sheets."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from neural_exam.common.figures import COLORS, save_figure


def training_curves(source: Path, destination: Path) -> None:
    """Show component losses separately from the cis-validation selection metric."""
    history = pd.read_csv(source)
    history.columns = history.columns.str.strip()
    figure, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    for axis, component in zip(axes[:3], ("box", "cls", "dfl"), strict=True):
        for split, label, color, style in (("train", "Train", COLORS[0], "-"), ("val", "cis_val", COLORS[1], "--")):
            axis.plot(history["epoch"], history[f"{split}/{component}_loss"], label=label, color=color, linestyle=style)
        axis.set(xlabel="Época", ylabel="Pérdida", title=component)
        axis.legend()
    for column, label, color, style in (
        ("metrics/mAP50(B)", "mAP50", COLORS[0], "--"),
        ("metrics/mAP50-95(B)", "mAP50:95", COLORS[2], "-"),
    ):
        axes[3].plot(history["epoch"], history[column], label=label, color=color, linestyle=style)
    axes[3].set(xlabel="Época", ylabel="mAP", ylim=(0, 1), title="Selección en cis_val")
    axes[3].legend()
    save_figure(figure, destination)


def cam_contact_sheets(cases: list[dict], output: Path) -> None:
    """Keep all twelve cases and identifiers visible in two report figures."""
    font = ImageFont.load_default(size=22)
    for split in ("cis_test", "trans_test"):
        group = [row for row in cases if row["split"] == split]
        if not group:
            continue
        tile_width, tile_height = 800, 450
        sheet = Image.new("RGB", (tile_width * 2, tile_height * ((len(group) + 1) // 2)), "white")
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(group):
            x, y = (index % 2) * tile_width, (index // 2) * tile_height
            with Image.open(row["figure"]) as image:
                image = image.convert("RGB")
                image.thumbnail((tile_width - 12, tile_height - 64))
                sheet.paste(image, (x + (tile_width - image.width) // 2, y))
            draw.text((x + 8, y + tile_height - 60), f"{row['class']} | {row['case']}", font=font, fill="black")
            draw.text(
                (x + 8, y + tile_height - 30),
                f"{row['image_id'][:12]} | {row['capture_period']} | n={row['training_support']}",
                font=font,
                fill="black",
            )
        sheet.save(output / f"cam_{split}_grid.png")
