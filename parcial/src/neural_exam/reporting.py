"""Generate LaTeX assets only from persisted scientific runs, without training."""

from __future__ import annotations

import shutil
from collections import Counter

from neural_exam.common.io import read_csv, read_json, write_json

TASKS = ("mri", "flood", "wildlife")
COLUMNS = {
    "mri": [("experiment", "Experimento"), ("macro_auc", "AUC macro"), ("macro_f1", "F1 macro")],
    "flood": [
        ("experiment", "Experimento"),
        ("test_iou", "IoU ID"),
        ("bolivia_iou", "IoU Bolivia"),
        ("geographic_gap", "Brecha"),
    ],
    "wildlife": [
        ("experiment", "Experimento"),
        ("cis_map50_95", "mAP cámaras vistas"),
        ("trans_map50_95", "mAP cámaras nuevas"),
        ("geographic_gap", "Brecha"),
    ],
}


def latex_escape(value) -> str:
    """Escape external identifiers so they cannot become LaTeX commands."""
    translations = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(translations.get(char, char) for char in str(value))


def _number(value) -> str:
    if value is None or value == "":
        return "ND"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, str) and value.lstrip("-").isdigit():
        return str(int(value))
    try:
        return f"{float(value):.4f}"
    except (ValueError, TypeError):
        return latex_escape(value)


def _table(rows: list[dict], columns: list[tuple[str, str]], caption: str) -> str:
    if not rows:
        return "\\paragraph{Pendiente.} No hay mediciones disponibles.\n"
    lines = [
        r"\begin{table}[ht]",
        r"\centering\small",
        r"\caption{" + latex_escape(caption) + "}",
        r"\begin{tabular}{" + "l" + "r" * (len(columns) - 1) + "}",
        r"\toprule",
        " & ".join(latex_escape(title) for _, title in columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        cells = [latex_escape(row.get(columns[0][0], ""))]
        cells += [_number(row.get(key)) for key, _ in columns[1:]]
        lines.append(" & ".join(cells) + r" \\")
    return "\n".join([*lines, r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])


def _wildlife_assets(source) -> tuple[str, list, dict]:
    from neural_exam.wildlife.review import label_review_status

    completed = read_json(source / "yolov8s/evaluation_complete.json")
    run = read_json(source / "yolov8s/run.json")
    if not run["scientific_run"]:
        raise ValueError("Smoke evaluation cannot generate scientific report assets.")
    manifest = read_json(source / "manifest.json")
    scores = {split: read_json(source / split / "metrics.json") for split in ("cis_test", "trans_test")}
    if any(score["protocol"]["checkpoint_sha256"] != completed["checkpoint_sha256"] for score in scores.values()):
        raise ValueError("Report evaluations do not use the same selected checkpoint.")
    tables = _table(
        [
            {
                "split": split,
                "images": summary["images"],
                "backgrounds": summary["backgrounds"],
                "cameras": len(summary["cameras"]),
                "sequences": summary["sequences"],
            }
            for split, summary in manifest["split_summary"].items()
        ],
        [
            ("split", "Partición"),
            ("images", "Fotos"),
            ("backgrounds", "Fondos"),
            ("cameras", "Cámaras"),
            ("sequences", "Secuencias"),
        ],
        "Inventario efectivo después de depurar anotaciones y secuencias. Train oficial no contiene imágenes empty.",
    )
    tables += _table(
        [
            {
                "split": split,
                "categories": score["protocol"]["categories_evaluated"],
                "map50": score["map50"],
                "map50_95": score["map50_95"],
            }
            for split, score in scores.items()
        ],
        [("split", "Partición"), ("categories", "Clases con soporte"), ("map50", "mAP50"), ("map50_95", "mAP50:95")],
        "Evaluación COCO global con el mismo checkpoint, 101 puntos de recall y máximo 300 detecciones por imagen.",
    )
    for split, score in scores.items():
        tables += _table(
            score["per_species"],
            [("species", "Especie"), ("support", "Cajas"), ("map50", "AP50"), ("map50_95", "AP50:95")],
            f"Rendimiento por especie: {split}. ND indica ausencia de referencia.",
        )
    robustness = read_json(source / "robustness.json")
    labels = {"small_lt_1pct": "Pequeño", "medium_1_to_5pct": "Mediano", "large_gt_5pct": "Grande"}
    for split, groups in robustness["size_by_split"].items():
        tables += _table(
            [{"group": labels[group], **score} for group, score in groups.items()],
            [("group", "Tamaño relativo"), ("eligible_objects", "Cajas"), ("map50", "AP50"), ("map50_95", "AP50:95")],
            f"Robustez por área relativa en {split}: menor de 1 %, 1--5 %, mayor de 5 %.",
        )
    tables += _table(
        [
            {"group": {"low": "Baja", "medium": "Media", "high": "Alta"}[group], **score}
            for group, score in robustness["occlusion"].items()
        ],
        [("group", "Oclusión"), ("reviewed_objects", "Revisados"), ("map50", "AP50"), ("map50_95", "AP50:95")],
        "Muestra de oclusión visual en trans_test; mínimo requerido: 30 objetos por nivel.",
    )
    if robustness["occlusion_status"] != "reviewed":
        tables += r"\pending{El análisis de oclusión no alcanza la revisión humana mínima en los tres niveles.}" + "\n"
    activity = read_json(source / "activity.json")
    tables += _table(
        [{"period": row["date"], **row} for row in activity["camera_days"][:10]],
        [
            ("period", "Día"),
            ("photos", "Fotos"),
            ("photos_with_fauna", "Con fauna"),
            ("distinct_species", "Especies"),
            ("empty_photo_rate", "Vacías"),
        ],
        f"Actividad predicha de la cámara {activity['camera_id']}: primeros diez días de la cohorte completa de esa cámara.",
    )
    counts, photos = Counter(), Counter()
    for row in activity["species_days"]:
        counts[row["species"]] += row["detections"]
        photos[row["species"]] += row["photos_with_species"]
    tables += _table(
        [{"species": species, "detections": counts[species], "photos": photos[species]} for species in sorted(counts)],
        [("species", "Especie"), ("detections", "Detecciones"), ("photos", "Fotos con especie")],
        "Resumen de todos los días de la cámara de demostración. Las detecciones no representan individuos únicos.",
    )
    sequence = read_csv(source / "activity_sequence.csv")
    tables += _table(
        sequence,
        [
            ("frame_number", "Fotograma"),
            ("timestamp", "Captura"),
            ("detections", "Detecciones"),
            ("species", "Especies"),
        ],
        "Secuencia completa con fecha de captura válida. Identificadores completos disponibles en activity_sequence.csv.",
    )
    figures = [
        (source / "class_balance.png", "Desbalance de cajas anotadas en entrenamiento."),
        (
            source / "training_curves.png",
            "Pérdidas por componente y mAP en cis_val. Las curvas de selección no usan los conjuntos de prueba.",
        ),
        (
            source / "cam_cis_test_grid.png",
            "Seis casos Grad-CAM en cámaras vistas: aciertos, errores y fondos. Día/noche se deriva de la hora de captura y requiere inspección visual.",
        ),
        (
            source / "cam_trans_test_grid.png",
            "Seis casos Grad-CAM en cámaras nuevas, con soporte de entrenamiento por especie. Los mapas no demuestran causalidad.",
        ),
    ]
    return (
        tables,
        figures,
        {
            "occlusion_status": robustness["occlusion_status"],
            "label_review_status": label_review_status(source / "label_review.csv")["status"],
            "checkpoint_sha256": completed["checkpoint_sha256"],
            "cam_cases": len(read_json(source / "cam_cases.json")),
        },
    )


def prepare_report_assets(settings, task="all") -> dict:
    """Reject smoke data and export actual tables and a fixed qualitative baseline."""
    if settings.output_root.name == "smoke":
        raise ValueError("Smoke outputs are excluded from scientific reports.")
    tasks = TASKS if task == "all" else (task,)
    results = {}
    for name in tasks:
        source = settings.output_root / name
        table_path = source / "experiments.csv"
        if not table_path.is_file():
            results[name] = {"status": "pending", "reason": "No experiments.csv from a scientific run"}
            continue
        destination = settings.output_root / "reports" / name
        destination.mkdir(parents=True, exist_ok=True)
        rows = read_csv(table_path)
        expected = {"mri": 8, "flood": 6, "wildlife": 1}[name]
        if len(rows) != expected:
            raise ValueError(f"Incomplete {name} experiment matrix: {len(rows)} of {expected}.")
        tables = _table(rows, COLUMNS[name], "Resultados de las ejecuciones registradas. ND: métrica no definida.")
        figures = []
        if name == "mri":
            baseline = source / "scratch_bce_weighted"
            metrics = read_json(baseline / "metrics.json")
            tables += _table(
                metrics["per_label"],
                [("label", "Hallazgo"), ("auc", "AUC"), ("precision", "Precisión"), ("recall", "Recall"), ("f1", "F1")],
                "Desglose del baseline fijado antes de evaluar.",
            )
            figures = [
                (source / "prevalence.png", "Prevalencia de etiquetas débiles en entrenamiento."),
                (baseline / "loss.png", "Pérdidas de entrenamiento y validación del baseline."),
            ]
            figures += [
                (baseline / f"cam_{index}.png", "Grad-CAM del baseline para revisión anatómica.") for index in (0, 2, 7)
            ]
            agreement = read_json(source / "label_agreement.json")
            tables += _table(
                [row for row in agreement["per_label"] if row["minimum_confidence"] == 0.8],
                [
                    ("label", "Hallazgo"),
                    ("observed", "Comparables"),
                    ("agreement", "Acuerdo"),
                    ("estimated_disagreement", "Desacuerdo"),
                ],
                "Reglas de alta confianza comparadas con etiquetas expertas.",
            )
        elif name == "flood":
            baseline = source / "gold_bce"
            detailed = []
            for split in ("test", "bolivia"):
                metrics = read_json(baseline / split / "metrics.json")
                detailed.append({"split": split, **metrics["pooled_pixels"]})
                figures.append((baseline / split / "confusion.png", f"Matriz de confusión del baseline: {split}."))
                figures += [
                    (path, "Ejemplo de error para análisis cualitativo.")
                    for path in sorted((baseline / split).glob("*_errors.png"))[:2]
                ]
            tables += _table(
                detailed,
                [
                    ("split", "Partición"),
                    ("iou", "IoU global"),
                    ("f1", "F1"),
                    ("precision", "Precisión"),
                    ("recall", "Recall"),
                    ("omission", "Omisión"),
                    ("commission", "Comisión"),
                ],
                "Métricas agregadas por píxel del baseline; distintas de IoU promedio por chip.",
            )
        else:
            wildlife_tables, figures, wildlife_status = _wildlife_assets(source)
            tables += wildlife_tables
        (destination / "tables.tex").write_text(tables, encoding="utf-8")
        figure_lines, missing = [], []
        for index, (path, caption) in enumerate(figures):
            if not path.is_file():
                missing.append(str(path))
                continue
            target = destination / "figures" / f"figure_{index}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            figure_lines.append(
                r"\begin{figure}[ht]\centering\includegraphics[width=0.94\linewidth]{figures/"
                + target.name
                + r"}\caption{"
                + latex_escape(caption)
                + r"}\end{figure}"
            )
        (destination / "figures.tex").write_text("\n".join(figure_lines), encoding="utf-8")
        status = {
            "status": "assets_generated",
            "experiment_count": len(rows),
            "missing_figures": missing,
            "editorial_review": "pending",
            "source": str(table_path),
            "test_selected_baseline": False,
        }
        if name == "wildlife":
            status.update(wildlife_status)
        write_json(destination / "status.json", status)
        results[name] = status
    return results
