"""Build flood-only report assets from the six persisted scientific runs."""

import argparse
import shutil
from itertools import product

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from neural_exam.common.figures import COLORS, save_figure
from neural_exam.common.io import read_csv, read_json, write_csv, write_json
from neural_exam.config import load_settings
from neural_exam.reporting import latex_escape


def _table(rows, columns, caption):
    lines = [
        r"\begin{table}[H]\centering\small",
        r"\caption{" + caption + "}",
        r"\begin{tabular}{" + "l" + "r" * (len(columns) - 1) + "}",
        r"\toprule",
        " & ".join(latex_escape(title) for _, title in columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key)
            cells.append("ND" if value is None else f"{value:.4f}" if isinstance(value, float) else latex_escape(value))
        lines.append(" & ".join(cells) + r" \\")
    return "\n".join([*lines, r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])


def prepare_report_assets(settings) -> dict:
    """Require a complete matrix and preserve both chip IoU conventions."""
    if settings.output_root.name == "smoke":
        raise ValueError("Scientific flood reports cannot use smoke outputs.")
    source = settings.output_root / "flood"
    manifest = read_json(source / "manifest.json")
    destination = settings.output_root / "reports/flood"
    destination.mkdir(parents=True, exist_ok=True)
    rows, detailed, reference, resources = [], [], [], []
    histories = {}
    matrix = list(product(settings.flood["supervision"], settings.flood["losses"]))
    if len(matrix) != 6:
        raise ValueError("The final flood report requires the six planned configurations.")
    for supervision, loss in matrix:
        name = f"{supervision}_{loss}"
        run = source / name
        completed = read_json(run / "complete.json")
        execution = read_json(run / "execution.json")
        if read_json(run / "run.json")["data_identity"] != manifest["identity"]:
            raise ValueError(f"Dataset identity differs from the current manifest: {name}")
        if not execution["scientific_run"]:
            raise ValueError(f"Not a scientific run: {name}")
        scores = {split: read_json(run / split / "metrics.json") for split in ("test", "bolivia")}
        inside, outside = (scores[split]["mean_chip_iou"] for split in ("test", "bolivia"))
        rows.append(
            {
                "experiment": name,
                "test_iou": inside,
                "bolivia_iou": outside,
                "geographic_gap": inside - outside if inside is not None and outside is not None else None,
            }
        )
        for split, score in scores.items():
            detailed.append({"experiment": name, "split": split, **score["pooled_pixels"]})
            reference.append(
                {
                    "experiment": name,
                    "split": split,
                    "defined": score["defined_iou_chips"],
                    "empty": score["empty_union_chips"],
                    "mean": score["mean_chip_iou"],
                    "official": score["reference_mean_chip_iou"],
                }
            )
        histories[name] = read_csv(run / "history.csv")
        best = min(histories[name], key=lambda row: float(row["val_loss"]))
        resources.append(
            {
                "experiment": name,
                "epochs": completed["epochs_executed"],
                "best_epoch": int(best["epoch"]),
                "steps": execution["steps_per_epoch"],
                "minutes": execution["training_wall_seconds"] / 60,
                "vram_gib": execution["peak_cuda_allocated_bytes"] / 1024**3,
            }
        )
    write_csv(destination / "metrics_all.csv", detailed)
    write_csv(destination / "iou_conventions.csv", reference)
    write_csv(destination / "resources.csv", resources)
    by_name = {row["experiment"]: row for row in rows}
    comparisons = []
    for loss in settings.flood["losses"]:
        gold, mixed = by_name[f"gold_{loss}"], by_name[f"gold_weak_{loss}"]
        comparisons.append(
            {
                "loss": loss,
                "id_delta": mixed["test_iou"] - gold["test_iou"],
                "bolivia_delta": mixed["bolivia_iou"] - gold["bolivia_iou"],
            }
        )
    minimum_id, maximum_id = min(row["test_iou"] for row in rows), max(row["test_iou"] for row in rows)
    minimum_bolivia, maximum_bolivia = min(row["bolivia_iou"] for row in rows), max(row["bolivia_iou"] for row in rows)
    (destination / "abstract_results.tex").write_text(
        f"Las seis corridas obtienen IoU por chip entre {minimum_id:.4f} y {maximum_id:.4f} en ID "
        f"y entre {minimum_bolivia:.4f} y {maximum_bolivia:.4f} en Bolivia. "
        "El efecto de agregar supervisión débil se compara dentro de cada pérdida; "
        "la diferencia de pasos y una única semilla limitan su interpretación causal.",
        encoding="utf-8",
    )
    baseline_id = read_json(source / "gold_bce/test/metrics.json")
    baseline_bolivia = read_json(source / "gold_bce/bolivia/metrics.json")
    analysis = (
        f"La IoU principal varía de {minimum_id:.4f} a {maximum_id:.4f} en ID y de "
        f"{minimum_bolivia:.4f} a {maximum_bolivia:.4f} en Bolivia. "
        "Las diferencias manual+débil menos manual, manteniendo la pérdida, son: "
        + "; ".join(
            f"{latex_escape(row['loss'])}, ID {row['id_delta']:+.4f} y Bolivia {row['bolivia_delta']:+.4f}"
            for row in comparisons
        )
        + ". No se atribuyen únicamente a calidad de etiquetas, porque también aumenta el presupuesto de pasos.\n\n"
        f"En el baseline manual+BCE, IoU por chip y global por píxel son "
        f"{baseline_id['mean_chip_iou']:.4f}/{baseline_id['pooled_pixels']['iou']:.4f} en ID y "
        f"{baseline_bolivia['mean_chip_iou']:.4f}/{baseline_bolivia['pooled_pixels']['iou']:.4f} en Bolivia. "
        "Esta diferencia confirma que la agregación cambia la pregunta evaluada: los chips pequeños de agua "
        "pesan igual en el promedio y menos en el cociente global. "
        f"Hay {baseline_id['empty_union_chips']} uniones vacías en ID y "
        f"{baseline_bolivia['empty_union_chips']} en Bolivia; ambas convenciones están en la tabla de referencia."
    )
    analysis += "\n\nEn las matrices del baseline, normalizadas por clase real, "
    for split, score in (("ID", baseline_id), ("Bolivia", baseline_bolivia)):
        counts = score["pooled_pixels"]
        false_positive_rate = counts["fp"] / (counts["fp"] + counts["tn"])
        analysis += (
            f"{split} omite {100 * counts['omission']:.1f}\\% del agua de referencia "
            f"y marca como agua {100 * false_positive_rate:.1f}\\% de los píxeles no acuáticos. "
        )
    analysis += (
        "El predominio de verdaderos negativos refleja el desbalance; una tasa baja de falsos positivos "
        "sobre no agua puede coexistir con omisiones relevantes de agua. "
        "Esta normalización por clase real no equivale a la comisión, cuyo denominador son las predicciones de agua."
    )
    (destination / "results_analysis.tex").write_text(analysis, encoding="utf-8")
    improved_id = sum(row["id_delta"] > 0 for row in comparisons)
    improved_bolivia = sum(row["bolivia_delta"] > 0 for row in comparisons)
    (destination / "conclusions.tex").write_text(
        f"La supervisión débil mejora la IoU principal en {improved_id} de tres pérdidas en ID "
        f"y en {improved_bolivia} de tres en Bolivia. "
        "La matriz permite observar sensibilidad a la pérdida y a la fuente de etiquetas, "
        "sin demostrar una mejora universal ni aislar el efecto de las etiquetas del número de actualizaciones. "
        "Los casos cualitativos muestran ambigüedad radiométrica, omisión de agua de respuesta alta y "
        "errores de detalle espacial. La evaluación externa debe conservar la agregación y el signo real de la brecha.\n\n",
        encoding="utf-8",
    )
    tables = _table(
        rows,
        [
            ("experiment", "Experimento"),
            ("test_iou", "IoU ID"),
            ("bolivia_iou", "IoU Bolivia"),
            ("geographic_gap", "Brecha"),
        ],
        "IoU promedio sobre chips con unión no vacía. Brecha: ID menos Bolivia.",
    )
    for split, title in [("test", "prueba en distribución"), ("bolivia", "Bolivia")]:
        tables += _table(
            [row for row in detailed if row["split"] == split],
            [
                ("experiment", "Experimento"),
                ("iou", "IoU global"),
                ("f1", "F1"),
                ("precision", "Prec."),
                ("recall", "Recall"),
                ("omission", "Omisión"),
                ("commission", "Comisión"),
            ],
            f"Métricas agrupadas por píxel en {title}.",
        )
    tables += _table(
        reference,
        [
            ("experiment", "Experimento"),
            ("split", "Partición"),
            ("defined", "Definidos"),
            ("empty", "Unión vacía"),
            ("official", "IoU referencia"),
        ],
        "Compatibilidad con la referencia oficial: epsilon de $10^{-7}$ y unión vacía igual a uno.",
    )
    (destination / "tables.tex").write_text(tables, encoding="utf-8")
    resource_table = _table(
        resources,
        [
            ("experiment", "Experimento"),
            ("epochs", "Épocas"),
            ("best_epoch", "Mejor época"),
            ("steps", "Pasos/época"),
            ("minutes", "Minutos"),
            ("vram_gib", "VRAM GiB"),
        ],
        "Presupuesto de entrenamiento y memoria CUDA asignada máxima; el límite del contenedor es 12 GiB de RAM.",
    )
    (destination / "resources.tex").write_text(resource_table, encoding="utf-8")
    inventory = [
        {
            "split": split,
            "chips": len(records),
            "valid_pixels": sum(record["valid_pixels"] for record in records),
            "water_fraction": sum(record["water_pixels"] for record in records)
            / sum(record["valid_pixels"] for record in records),
        }
        for split, records in manifest["records"].items()
    ]
    (destination / "inventory.tex").write_text(
        _table(
            inventory,
            [
                ("split", "Partición"),
                ("chips", "Chips"),
                ("valid_pixels", "Píxeles válidos"),
                ("water_fraction", "Fracción agua"),
            ],
            "Inventario efectivo después de excluir nodata.",
        ),
        encoding="utf-8",
    )
    figure, axes = plt.subplots(1, 3, figsize=(7.5, 3.5))
    supervision_titles = {"gold": "Manual", "gold_weak": "Manual + débil"}
    loss_titles = {"bce": "BCE", "dice": "Dice", "combined": "BCE + Dice"}
    for axis, loss in zip(axes, settings.flood["losses"], strict=True):
        for index, supervision in enumerate(settings.flood["supervision"]):
            history = histories[f"{supervision}_{loss}"]
            axis.plot(
                [int(row["epoch"]) for row in history],
                [float(row["val_loss"]) for row in history],
                color=COLORS[index],
                linestyle="-" if index == 0 else "--",
                label=supervision_titles[supervision],
            )
        axis.set_title(loss_titles[loss], fontsize=12)
        axis.set_xlabel("Época", fontsize=12)
        axis.set_ylabel("Pérdida", fontsize=12)
        axis.tick_params(labelsize=11)
        axis.legend(fontsize=11)
    save_figure(figure, destination / "validation_curves.png")
    figure, axis = plt.subplots(figsize=(7.5, 3.5))
    plot_supervision_titles = {"gold": "Manual", "gold_weak": "Manual +\ndébil"}
    positions = np.arange(len(rows))
    for offset, key, label, color in [
        (-0.18, "test_iou", "ID", COLORS[0]),
        (0.18, "bolivia_iou", "Bolivia", COLORS[1]),
    ]:
        axis.bar(
            positions + offset,
            [np.nan if row[key] is None else row[key] for row in rows],
            width=0.36,
            label=label,
            color=color,
        )
    axis.set(
        xticks=positions,
        xticklabels=[f"{plot_supervision_titles[supervision]}\n{loss_titles[loss]}" for supervision, loss in matrix],
        ylim=(0, 1),
        ylabel="IoU promedio por chip",
    )
    axis.tick_params(labelsize=11)
    axis.set_ylabel("IoU promedio por chip", fontsize=12)
    axis.legend(fontsize=11)
    save_figure(figure, destination / "ablation.png")
    figures = [
        (destination / "validation_curves.png", "Validación manual por pérdida y fuente de supervisión."),
        (destination / "ablation.png", "Comparación de las seis configuraciones en ID y Bolivia."),
    ]
    baseline = source / "gold_bce"
    figure, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for axis, split in zip(axes, ("test", "bolivia"), strict=True):
        counts = read_json(baseline / split / "metrics.json")["pooled_pixels"]
        matrix_counts = np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]])
        normalized = matrix_counts / np.maximum(matrix_counts.sum(1, keepdims=True), 1)
        axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        for (row, column), value in np.ndenumerate(matrix_counts):
            axis.text(
                column,
                row,
                f"{value:,}\n{normalized[row, column]:.1%}",
                ha="center",
                va="center",
                fontsize=12,
                color="white" if normalized[row, column] > 0.6 else "black",
            )
        axis.set(
            title=split,
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["No agua", "Agua"],
            yticklabels=["No agua", "Agua"],
            xlabel="Predicción",
            ylabel="Referencia",
        )
        axis.tick_params(labelsize=12)
        axis.set_xlabel("Predicción", fontsize=12)
        axis.set_ylabel("Referencia", fontsize=12)
        axis.set_title(split, fontsize=13)
    save_figure(figure, destination / "confusion_baseline.png")
    figures.append(
        (
            destination / "confusion_baseline.png",
            "Baseline gold+BCE: conteos y porcentajes normalizados por clase real.",
        )
    )
    reviewed = []
    for split in ("test", "bolivia"):
        review_path = baseline / split / "qualitative_review.csv"
        if review_path.is_file():
            reviewed += [(split, row) for row in read_csv(review_path) if row.get("visual_evidence")]
    for split, row in reviewed[:3]:
        path = baseline / split / f"{row['chip_id']}_errors.png"
        if not path.is_file():
            raise FileNotFoundError(f"Missing reviewed error figure: {path}")
        figures.append((path, f"{row['chip_id']} ({split}): {row['visual_evidence']}"))
    lines = []
    for index, (path, caption) in enumerate(figures):
        target = destination / "figures" / f"flood_{index}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        lines.append(
            r"\begin{figure}[H]\centering\includegraphics[width=0.96\linewidth]{figures/"
            + target.name
            + r"}\caption{"
            + latex_escape(caption)
            + r"}\end{figure}"
        )
    (destination / "figures.tex").write_text("\n".join(lines[:3]), encoding="utf-8")
    (destination / "qualitative_figures.tex").write_text("\n".join(lines[3:]), encoding="utf-8")
    status = {
        "status": "assets_generated",
        "experiment_count": 6,
        "missing_figures": [],
        "qualitative_examples_reviewed": len(reviewed),
        "editorial_review": "pending",
        "source": str(source),
        "test_selected_baseline": False,
    }
    write_json(destination / "status.json", status)
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/flood.toml")
    arguments = parser.parse_args()
    print(prepare_report_assets(load_settings(arguments.config)))
