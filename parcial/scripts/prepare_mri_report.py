"""Generate MRI report tables and figures from audited scientific outputs."""

from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from neural_exam.common.figures import save_figure
from neural_exam.common.io import read_csv, read_json, write_csv, write_json
from neural_exam.config import load_settings
from neural_exam.mri.labels import LABELS
from neural_exam.reporting import _table, latex_escape


def main() -> None:
    """Export complete observations without selecting a model on the gold outcomes."""
    settings = load_settings("configs/mri.toml")
    plt.rcParams.update({"font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14})
    root = settings.output_root / "mri"
    destination = settings.output_root / "reports/mri"
    figures = destination / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    audit = read_json(settings.output_root / "audit/mri_results.json")
    assert audit["status"] == "passed"
    manifest = read_json(root / "manifest.json")
    data = read_json(settings.output_root / "audit/mri_data.json")
    runs = audit["experiments"]
    results = []
    for row in runs:
        architecture, loss, strategy = row["experiment"].split("_", 2)
        results.append(
            {
                **row,
                "name": ("CNN" if architecture == "scratch" else "ResNet")
                + "/"
                + loss
                + "/"
                + ("pond." if strategy == "weighted" else "filtro"),
            }
        )
    write_csv(root / "metrics_all_labels.csv", audit["per_label"])
    tables = _table(
        results,
        [
            ("name", "Variante"),
            ("epochs", "Épocas"),
            ("best_epoch", "Mejor"),
            ("macro_auc", "AUC 12"),
            ("partial_macro_auc", "AUC parcial"),
            ("macro_f1", "F1 macro"),
        ],
        "Ocho variantes con el mismo conjunto experto. ND no se sustituye por el promedio parcial.",
    )
    baseline = read_json(root / "scratch_bce_weighted/metrics.json")
    tables += _table(
        baseline["per_label"],
        [
            ("label", "Hallazgo"),
            ("positives", "Positivos"),
            ("auc", "AUC"),
            ("precision", "Precisión"),
            ("recall", "Recall"),
            ("f1", "F1"),
        ],
        "Baseline CNN/BCE/ponderación fijado previamente. Evaluación de 58 estudios, umbral 0,5.",
    )
    prevalence = []
    for index, label in enumerate(LABELS):
        item = {"label": label}
        for split in ("train", "val", "gold"):
            rows = [row for row in manifest["records"] if row["split"] == split]
            item[split] = sum(row["targets"][index] == 1 for row in rows)
        rows = [row for row in manifest["records"] if row["split"] == "train"]
        item["high_observed"] = sum(
            row["targets"][index] is not None and row["weak"]["confidence"][index] >= 0.8 for row in rows
        )
        item["high_positive"] = sum(
            row["targets"][index] == 1 and row["weak"]["confidence"][index] >= 0.8 for row in rows
        )
        prevalence.append(item)
    write_csv(root / "class_balance.csv", prevalence)
    train_count = manifest["split_counts"]["train"]
    prevalence_display = [
        {**row, "train_prevalence": f"{row['train']} ({100 * row['train'] / train_count:.1f}%)"}
        for row in prevalence
    ]
    tables += _table(
        prevalence_display,
        [
            ("label", "Hallazgo"),
            ("train_prevalence", "Train + (%)"),
            ("val", "Val +"),
            ("gold", "Gold +"),
            ("high_observed", "Filtro obs."),
            ("high_positive", "Filtro +"),
        ],
        "Positivos y porcentaje sobre 434 estudios de entrenamiento, incluidas abstenciones en el denominador. "
        "La ausencia de mención recibe peso 0,2; no es negación experta. Filtro obs. cuenta etiquetas observadas, no estudios.",
    )
    agreement = read_json(root / "label_agreement.json")["per_label"]
    agreement_rows = []
    for label in LABELS:
        all_rules = next(row for row in agreement if row["label"] == label and row["minimum_confidence"] == 0)
        high = next(row for row in agreement if row["label"] == label and row["minimum_confidence"] == 0.8)
        agreement_rows.append(
            {
                "label": label,
                "all_observed": all_rules["observed"],
                "all_disagreement": all_rules["estimated_disagreement"],
                "high_observed": high["observed"],
                "high_disagreement": high["estimated_disagreement"],
            }
        )
    tables += _table(
        agreement_rows,
        [
            ("label", "Hallazgo"),
            ("all_observed", "Todas obs."),
            ("all_disagreement", "Desacuerdo"),
            ("high_observed", "Filtro obs."),
            ("high_disagreement", "Desacuerdo"),
        ],
        "Comparación diagnóstica de reglas congeladas con los 58 expertos; observados/58 expresa cobertura.",
    )
    script_rows = [
        {
            "script": key,
            "reports": count,
            "supported": data["weak_supported_by_script"].get(key, 0),
            "coverage": data["weak_supported_by_script"].get(key, 0) / count,
        }
        for key, count in data["weak_reports_by_script"].items()
    ]
    tables += _table(
        script_rows,
        [
            ("script", "Escritura"),
            ("reports", "Reportes débiles"),
            ("supported", "Compatibles"),
            ("coverage", "Cobertura"),
        ],
        "Cobertura del extractor en todos los 4.349 reportes no expertos. Escritura no identifica idioma ni centro; Thai tiene solo tres casos.",
    )
    (destination / "tables.tex").write_text(tables, encoding="utf-8")
    matrix = np.array(
        [
            [
                next(
                    score
                    for score in audit["per_label"]
                    if score["experiment"] == run["experiment"] and score["label"] == label
                )["auc"]
                for run in results
            ]
            for label in LABELS
        ],
        dtype=float,
    )
    figure, axis = plt.subplots(figsize=(6.5, 5.6))
    artist = axis.imshow(matrix, vmin=0, vmax=1, cmap="cividis", aspect="auto")
    axis.set_xticks(
        range(8),
        [
            row["name"].replace("ResNet/", "R50/").replace("bce", "BCE").replace("focal", "FCL")
            .replace("pond.", "P").replace("filtro", "F").replace("/", "\n")
            for row in results
        ],
        fontsize=12,
    )
    axis.set_yticks(range(12), LABELS)
    for y in range(12):
        for x in range(8):
            value = matrix[y, x]
            axis.text(
                x,
                y,
                "ND" if np.isnan(value) else f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=12,
                color="white" if np.isnan(value) or value < 0.5 else "black",
            )
    figure.colorbar(artist, ax=axis, label="AUC-ROC")
    axis.set_title("Discriminación por hallazgo en las ocho variantes")
    save_figure(figure, figures / "auc_matrix.png")
    for architecture_index, architecture in enumerate(("scratch", "resnet50")):
        figure, axes = plt.subplots(2, 2, figsize=(6.5, 5.6))
        for axis, row in zip(axes.flat, results[architecture_index * 4 : (architecture_index + 1) * 4], strict=True):
            history = read_csv(root / row["experiment"] / "history.csv")
            epochs = [int(point["epoch"]) for point in history]
            axis.plot(epochs, [float(point["train_loss"]) for point in history], label="Train")
            axis.plot(epochs, [float(point["val_loss"]) for point in history], "--", label="Val débil")
            axis.axvline(row["best_epoch"], color="gray", linewidth=1, linestyle=":")
            axis.set(title=row["name"].split("/", 1)[1], xlabel="Época", ylabel="Pérdida")
            axis.tick_params(labelsize=12)
        axes.flat[0].legend(fontsize=11)
        save_figure(figure, figures / f"curves_{architecture}.png")
    figure, axes = plt.subplots(2, 3, figsize=(6.5, 4.8))
    cases = read_json(root / "scratch_bce_weighted/cam_cases.json")
    for column, case in enumerate(cases):
        with np.load(root / "scratch_bce_weighted" / f"cam_{case['label_index']}.npz", allow_pickle=False) as arrays:
            stack = case["selected_stack"]
            for row in (0, 1):
                axes[row, column].imshow(arrays["image"][stack], cmap="gray")
            axes[1, column].imshow(arrays["heatmaps"][stack], cmap="magma", alpha=0.5, vmin=0, vmax=1)
        axes[0, column].set_title(case["label"], fontsize=14)
        axes[1, column].set_title(f"p={case['probability']:.3f}, gold=1", fontsize=13)
        for row in (0, 1):
            axes[row, column].axis("off")
    save_figure(figure, figures / "cam_grid.png")
    captions = [
        (
            "auc_matrix.png",
            "AUC por hallazgo y variante: CNN propia y R50 (ResNet-50); FCL: focal; P: ponderación; F: filtro. "
            "Las cifras y ND también están disponibles en metrics_all_labels.csv.",
        ),
        (
            "curves_scratch.png",
            "Curvas completas de la CNN propia. La línea vertical marca el checkpoint elegido. Las escalas dependen de la pérdida y no deben compararse como una misma magnitud.",
        ),
        (
            "curves_resnet50.png",
            "Curvas completas de ResNet-50. La selección utiliza únicamente pérdida de validación débil, con paciencia de diez épocas.",
        ),
        (
            "cam_grid.png",
            "Grad-CAM del baseline para tres positivos expertos, con probabilidades al umbral fijo. Casos fijados por orden del manifiesto; no elegidos por apariencia ni por acierto.",
        ),
    ]
    (destination / "figures.tex").write_text(
        "\n".join(
            r"\begin{figure}[p]\centering\includegraphics[width=0.83\linewidth]{figures/"
            + filename
            + r"}\caption{"
            + latex_escape(caption)
            + r"}\label{fig:mri_"
            + filename.removesuffix(".png")
            + r"}\end{figure}"
            for filename, caption in captions
        ),
        encoding="utf-8",
    )
    scripts = Counter(row["weak"]["script"] for row in manifest["records"])
    planes = Counter(row["plane"] for row in manifest["records"])
    inventory = (
        f"Se validaron {data['all_metadata_studies']} estudios y {data['all_metadata_series']} descriptores. "
        f"La muestra efectiva tiene {manifest['split_counts']['train']} estudios de entrenamiento, "
        f"{manifest['split_counts']['val']} de validación débil y 58 expertos. "
        f"Se comprobaron hashes, identificadores y decodificación de {data['selected_dicom_files']} DICOM. "
        f"La caché coincide byte a byte con sus originales. Planos: {dict(planes)}. "
        f"Grupos de escritura de la muestra: {dict(scripts)}."
    )

    (destination / "inventory.tex").write_text(latex_escape(inventory), encoding="utf-8")
    undefined = [row["label"] for row in baseline["per_label"] if row["auc"] is None]
    auc_name = "AUC macro de doce etiquetas" if not undefined else "AUC parcial"
    official = (
        "definido para las doce etiquetas"
        if not undefined
        else "indefinido por falta de ambas clases en " + ", ".join(undefined)
    )
    abstract = (
        "Se ejecutó una comparación factorial de ocho variantes de clasificación de doce hallazgos "
        "con CNN propia y ResNet-50, BCE/focal y dos tratamientos del ruido textual. "
        f"Los 600 estudios incluyen 58 expertos reservados para evaluación. El AUC macro oficial quedó {official}. "
        f"El baseline previamente fijado obtuvo {auc_name} {baseline['partial_macro_auc']:.3f} "
        f"sobre {baseline['auc_labels_defined']} etiquetas y F1 macro {baseline['macro_f1']:.3f}. "
        "Se auditaron predicciones y checkpoints y se generaron explicaciones para tres hallazgos. "
        "La muestra pequeña, el ruido lingüístico y una única semilla limitan las conclusiones."
    )
    (destination / "abstract_results.tex").write_text(latex_escape(abstract), encoding="utf-8")
    defined = baseline["auc_labels_defined"]
    partial_scores = [row["partial_macro_auc"] for row in results if row["partial_macro_auc"] is not None]
    f1_scores = [row["macro_f1"] for row in results]
    analysis = (
        f"En esta evaluación se dispone de AUC para {defined} de los doce hallazgos. "
        f"El {auc_name} de las variantes varía entre {min(partial_scores):.3f} y {max(partial_scores):.3f}; "
        f"F1 macro varía entre {min(f1_scores):.3f} y {max(f1_scores):.3f}. "
        "Estos intervalos describen la matriz ejecutada y no son intervalos de confianza. "
        "No se elige una variante para despliegue por estos resultados de evaluación. "
        "Las curvas muestran el criterio aplicado para elegir cada checkpoint; no se comparan los valores "
        "absolutos de BCE y focal como si fueran la misma escala. "
    )
    effects = []
    for architecture in ("scratch", "resnet50"):
        for strategy in ("weighted", "high_confidence"):
            bce = next(row for row in results if row["experiment"] == f"{architecture}_bce_{strategy}")
            focal = next(row for row in results if row["experiment"] == f"{architecture}_focal_{strategy}")
            effects.append(
                f"{bce['name']}: diferencia focal menos BCE de AUC "
                f"{focal['partial_macro_auc'] - bce['partial_macro_auc']:+.3f} y de F1 macro "
                f"{focal['macro_f1'] - bce['macro_f1']:+.3f}"
            )
    analysis += " Comparaciones pareadas por arquitectura y tratamiento del ruido: " + "; ".join(effects) + ". "
    analysis += (
        "Con una semilla y una cohorte experta pequeña, estas diferencias no establecen superioridad general. "
        "La mitigación del desbalance debe evaluarse por hallazgo y soporte, incluyendo posibles pérdidas "
        "de precisión al elevar recall. El umbral 0,5 permanece fijo aun cuando una variante produce "
        "probabilidades mal calibradas. La distribución tras el filtro explica qué etiquetas "
        "conservan positivos y negativos explícitos; no se interpreta el filtrado como limpieza perfecta."
    )
    (destination / "results_analysis.tex").write_text(latex_escape(analysis), encoding="utf-8")
    resource_rows = []
    for row in results:
        receipt = read_json(root / row["experiment"] / "execution.json")
        resource_rows.append(
            {
                "name": row["name"],
                "seconds": receipt["training_and_evaluation_seconds"],
                "vram_gib": receipt["peak_cuda_memory_bytes"] / 2**30,
            }
        )
    (destination / "resources.tex").write_text(
        _table(
            resource_rows,
            [("name", "Variante"), ("seconds", "Segundos"), ("vram_gib", "VRAM tensor GiB")],
            "Tiempo de entrenamiento y evaluación, y máximo de memoria asignada por PyTorch; no equivalen al consumo total del dispositivo.",
        ),
        encoding="utf-8",
    )
    write_json(
        destination / "status.json",
        {
            "status": "assets_generated",
            "experiment_count": 8,
            "missing_figures": [],
            "editorial_review": "pending",
            "source": str(root / "experiments.csv"),
            "test_selected_baseline": False,
            "result_audit": "passed",
            "radiologist_review": "not_recorded_not_required_by_exam",
        },
    )
    print("MRI report assets generated from audited results.")


if __name__ == "__main__":
    main()
