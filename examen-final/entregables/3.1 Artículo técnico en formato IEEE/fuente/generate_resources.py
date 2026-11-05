"""Rebuild article resources from saved PeruFoodNet results, without model execution.

Dependencies: numpy==2.5.2, pandas==3.1.0, matplotlib==3.11.2.
The results directory supplies table CSVs, winner.json, matrix_index.json,
the selected history, test/matriz_confusion.csv (including class names),
and test/ejemplos/error.jpg. No dataset, weights, Git or MLflow is required.
"""

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FAMILIES = {"resnet18": "ResNet-18", "yolo26m": "YOLO26m-cls"}
PDF_METADATA = {"CreationDate": None, "ModDate": None}


def number(value: float) -> str:
    """Format four decimals for Spanish LaTeX."""
    return f"{value:.4f}".replace(".", "{,}")


def write_table(path: Path, columns: str, header: str, rows: list[str]) -> None:
    """Write a complete tabular fragment with booktabs rules."""
    lines = [rf"\begin{{tabular}}{{{columns}}}\toprule", header + r"\\\midrule"]
    lines.extend(row + r" \\" for row in rows)
    lines.append(r"\bottomrule\end{tabular}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_tables(summary: pd.DataFrame, fixed: pd.DataFrame, winner: dict, output: Path) -> None:
    """Format saved means and fixed-rate comparisons; do not select configurations."""
    rows = []
    for family, label in FAMILIES.items():
        for regime in ("capa final", "red completa"):
            for budget in (10, 40):
                row = summary[
                    (summary.modelo == family)
                    & (summary["régimen"] == regime)
                    & (summary.fotos_por_clase == budget)
                ].iloc[0]
                mean = number(row.media_f1)
                if (family, regime, budget) == (
                    winner["modelo"], "red completa" if winner["régimen"] == "full" else "capa final",
                    winner["fotos_por_clase"],
                ):
                    mean = r"\mathbf{" + mean + "}"
                adjustment = "Capa" if regime == "capa final" else "Completa"
                rate = round(row.tasa / 1e-4)
                rows.append(
                    f"{label} & {adjustment} & {budget} & {rate} & "
                    rf"${mean} \pm {number(row['desviación_muestral'])}$"
                )
    write_table(
        output / "validation_rows.tex", "llrrc",
        r"Modelo & Ajuste & Fotos/clase & $\eta$ & F1 macro", rows,
    )
    rows = []
    for family, label in FAMILIES.items():
        for budget in (10, 40):
            for rate in (0.0001, 0.0003):
                group = fixed[
                    (fixed.modelo == family) & (fixed.fotos_por_clase == budget) & (fixed.tasa == rate)
                ]
                head = group.loc[group["régimen"] == "capa final", "f1_validación"].item()
                full = group.loc[group["régimen"] == "red completa", "f1_validación"].item()
                rows.append(
                    f"{label} & {budget} & {round(rate / 1e-4)} & "
                    f"${number(head)}$ & ${number(full)}$ & ${number(full - head)}$"
                )
    write_table(
        output / "ablation_rows.tex", "lrrccc",
        r"Modelo & Fotos & $\eta$ & Capa & Completa & $\Delta$", rows,
    )


def generate_curves(history: pd.DataFrame, best_epoch: int, figures: Path) -> None:
    """Plot the saved history and mark its already recorded best epoch."""
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.18), layout="constrained")
    for ax, suffix, title in zip(axes[:2], ("loss", "accuracy"), ("Pérdida", "Exactitud")):
        for split, label, color, marker in (
            ("train", "Entrenamiento", "#0072B2", "o"),
            ("val", "Validación", "#D55E00", "s"),
        ):
            ax.plot(
                history.epoch, history[f"{split}/{suffix}"], label=label,
                color=color, marker=marker, markersize=2.6, linewidth=1,
            )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Época")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Entropía cruzada")
    axes[0].legend(fontsize=7, frameon=False)
    axes[1].set(ylim=(0, 1.04), ylabel="Fracción correcta")
    axes[2].plot(history.epoch, history["val/f1_macro"], color="#D55E00", marker="s", markersize=3)
    axes[2].axvline(best_epoch, color="0.3", linestyle="--", linewidth=0.8)
    best_f1 = history.loc[history.epoch == best_epoch, "val/f1_macro"].item()
    axes[2].annotate(
        f"Selección: {best_epoch}", (best_epoch, round(best_f1, 3)),
        xytext=(best_epoch + 1, 0.73), fontsize=8,
        arrowprops={"arrowstyle": "->", "linewidth": 0.7},
    )
    axes[2].set(title="F1 macro de validación", xlabel="Época", ylabel="F1 macro", ylim=(0.55, 0.95))
    axes[2].grid(alpha=0.2)
    last_epoch = int(history.epoch.iloc[-1])
    for ax in axes:
        ax.set_xticks(sorted({1, *range(4, last_epoch + 1, 4), last_epoch}))
    fig.savefig(figures / "curves.pdf", metadata=PDF_METADATA)
    plt.close(fig)


def generate_confusion(matrix: pd.DataFrame, figures: Path) -> None:
    """Render saved counts using the class order contained in the CSV."""
    counts = matrix.to_numpy()
    display = counts.astype(float)
    display[display == 0] = np.nan
    count = len(matrix)
    fig, ax = plt.subplots(figsize=(7.16, 6.5), layout="constrained")
    image = ax.imshow(display, cmap="Blues", vmin=0, vmax=15)
    ax.set_xticks(range(count), range(1, count + 1), fontsize=8, rotation=90)
    ax.set_yticks(range(count), [f"{i + 1:02d} {name}" for i, name in enumerate(matrix.index)], fontsize=7.8)
    ax.set_xlabel("Clase propuesta (mismo índice que el eje vertical)", fontsize=9)
    ax.set_ylabel("Clase real", fontsize=9)
    for i, j in zip(*np.nonzero(counts)):
        ax.text(
            j, i, str(counts[i, j]), ha="center", va="center", fontsize=6.3,
            color="white" if counts[i, j] >= 9 else "black",
        )
    bar = fig.colorbar(image, ax=ax, fraction=0.028, pad=0.02, ticks=[0, 5, 10, 15])
    bar.set_label("Fotografías", fontsize=9)
    fig.savefig(figures / "confusion.pdf", metadata=PDF_METADATA)
    plt.close(fig)


def main() -> None:
    """Generate only the five article resources at the explicitly requested destination."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True, help="Saved results/ directory from the code package.")
    parser.add_argument("--output", type=Path, required=True, help="Article source directory or a separate output folder.")
    args = parser.parse_args()
    results, output = args.results.resolve(), args.output.resolve()
    if results == output or results in output.parents:
        parser.error("--output must be outside the input results directory")
    summary = pd.read_csv(results / "resumen_validacion.csv")
    fixed = pd.read_csv(results / "comparacion_misma_tasa.csv")
    winner = json.loads((results / "winner.json").read_text(encoding="utf-8-sig"))["winner"]
    index = json.loads((results / "matrix_index.json").read_text(encoding="utf-8-sig"))
    record = next(row for row in index if row["run_id"] == winner["run_id_seed42"])
    history = pd.read_csv(results / "histories" / f"{winner['run_id_seed42']}.csv")
    matrix = pd.read_csv(results / "test/matriz_confusion.csv", index_col=0)
    photo = results / "test/ejemplos/error.jpg"
    if not photo.is_file():
        parser.error(f"Missing saved example: {photo}")
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42,
    })
    figures = output / "figuras"
    figures.mkdir(parents=True, exist_ok=True)
    generate_tables(summary, fixed, winner, output)
    generate_curves(history, int(record["best_epoch"]), figures)
    generate_confusion(matrix, figures)
    shutil.copyfile(photo, figures / "error.jpg")
    print(f"Generated two LaTeX tables and three figures in {output}")


if __name__ == "__main__":
    main()
