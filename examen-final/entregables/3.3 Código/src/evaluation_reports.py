"""Build test-set tables and figures from saved predictions and example images."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def recipe(catalog: dict, class_name: str) -> dict:
    """Preserve the two documented columns and their orientative union."""
    entry = catalog["dishes"][class_name]
    codes = sorted(set(entry["efsa"] or []) | set(entry["usda"] or []))
    return {"paper_name": entry["paper_name"], "efsa_codes": entry["efsa"], "usda_codes": entry["usda"],
            "union": [{"code": code, "name": catalog["ingredients"][code]["es"]} for code in codes]}


def choose_example(predictions: pd.DataFrame, catalog: dict) -> dict:
    """Choose the first real error whose two documented unions differ."""
    for row in predictions.itertuples(index=False):
        if row.correct:
            continue
        actual = recipe(catalog, row.true_class)
        proposed = recipe(catalog, row.predicted_class)
        actual_codes = {item["code"] for item in actual["union"]}
        predicted_codes = {item["code"] for item in proposed["union"]}
        if actual_codes != predicted_codes:
            return {"relative_path": row.relative_path, "true_class": row.true_class,
                    "predicted_class": row.predicted_class, "score": float(row.score),
                    "uncertain": bool(row.uncertain), "true_recipe": actual, "predicted_recipe": proposed,
                    "only_true_codes": sorted(actual_codes - predicted_codes),
                    "only_predicted_codes": sorted(predicted_codes - actual_codes),
                    "source": catalog["source"], "photo_license": "CC BY 4.0",
                    "photo_source": "https://data.mendeley.com/datasets/hxhbbm497d/3",
                    "limitation": "La foto no detecta ingredientes; una lista vacía no garantiza ausencia."}
    raise ValueError("No classification error with different documented ingredients was found")


def generate_test_reports(predictions: pd.DataFrame, labels: list[str], catalog: dict,
                          example_image: Path, output: Path) -> dict:
    """Generate per-class results, confusion figures, pairs and one documented error."""
    output.mkdir(parents=True, exist_ok=True)
    targets = predictions["true_index"].to_numpy()
    predicted = predictions["predicted_index"].to_numpy()
    report = classification_report(targets, predicted, labels=list(range(len(labels))), target_names=labels,
                                   output_dict=True, zero_division=0)
    report.pop("accuracy", None)
    class_report = pd.DataFrame(report).transpose().rename(columns={"precision": "precisión", "recall": "sensibilidad",
                                                                  "f1-score": "f1", "support": "muestras"})
    class_report.rename(index={"accuracy": "exactitud", "macro avg": "promedio macro",
                               "weighted avg": "promedio ponderado"}).to_csv(output / "metricas_por_clase.csv", encoding="utf-8")
    if "cross_entropy" in predictions:
        recomputed = {"images": len(predictions), "accuracy": accuracy_score(targets, predicted),
                      "f1_macro": f1_score(targets, predicted, labels=list(range(len(labels))), average="macro", zero_division=0),
                      "loss": float(predictions["cross_entropy"].mean()),
                      "uncertain_count": int((predictions["score"] < 0.5).sum())}
        (output / "metricas_recalculadas.json").write_text(json.dumps(recomputed, indent=2), encoding="utf-8")
    matrix = confusion_matrix(targets, predicted, labels=list(range(len(labels))))
    pd.DataFrame(matrix, index=labels, columns=labels).rename_axis("clase_real").to_csv(output / "matriz_confusion.csv", encoding="utf-8")
    pairs = []
    for left in range(len(labels)):
        for right in range(left + 1, len(labels)):
            forward, backward = int(matrix[left, right]), int(matrix[right, left])
            if forward + backward:
                pairs.append({"clase_a": labels[left], "clase_b": labels[right], "a_predicha_b": forward,
                              "b_predicha_a": backward, "total": forward + backward})
    pairs.sort(key=lambda item: (-item["total"], item["clase_a"], item["clase_b"]))
    if len(pairs) < 2:
        raise ValueError("At least two nonzero confusion pairs are required")
    pd.DataFrame(pairs).to_csv(output / "pares_confundidos.csv", index=False, encoding="utf-8")
    cutoff = pairs[1]["total"]
    ranked = [item for item in pairs if item["total"] >= cutoff]
    case = choose_example(predictions, catalog)
    analysis = {"pair_rule": "Pares no ordenados; suma de ambas direcciones. Se incluyen empates hasta el segundo puesto.",
                "top_pairs_including_ties": ranked, "second_place_tied": sum(x["total"] == cutoff for x in pairs) > 1,
                "example_rule": "Primer error en el orden del manifiesto con uniones documentales distintas.",
                "example": case}
    (output / "analisis_errores.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, axis = plt.subplots(figsize=(18, 16))
    image = axis.imshow(matrix, cmap="Blues", interpolation="nearest")
    axis.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels,
             xlabel="Clase predicha", ylabel="Clase real", title="Matriz de confusión · 600 fotografías de prueba")
    axis.tick_params(axis="x", labelrotation=90, labelsize=7)
    axis.tick_params(axis="y", labelsize=7)
    fig.colorbar(image, ax=axis, label="Número de fotografías", fraction=0.046)
    fig.tight_layout()
    fig.savefig(output / "matriz_confusion.png", dpi=160)
    plt.close(fig)

    shown = pairs[:min(10, len(pairs))]
    fig, axis = plt.subplots(figsize=(10, max(4, len(shown) * .45)))
    names = [f"{x['clase_a']} / {x['clase_b']}" for x in shown]
    axis.barh(names, [x["total"] for x in shown], color="#285f91")
    axis.invert_yaxis()
    axis.set(xlabel="Errores en ambas direcciones", title="Pares de clases más confundidos")
    fig.tight_layout()
    fig.savefig(output / "pares_confundidos.png", dpi=160)
    plt.close(fig)

    with Image.open(example_image) as source:
        photo = source.convert("RGB")
    fig, axis = plt.subplots(figsize=(8, 6))
    axis.imshow(photo)
    axis.axis("off")
    axis.set_title(f"Real: {case['true_class']} · Predicción: {case['predicted_class']}\nPuntuación: {case['score']:.3f}")
    fig.text(.5, .02, "PeruFoodNet v3 · CC BY 4.0 · Foto del conjunto de prueba", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(output / "ejemplo_error.png", dpi=150)
    plt.close(fig)
    return analysis
