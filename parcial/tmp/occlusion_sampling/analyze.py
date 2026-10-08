"""Read-only population audit and a proposed occlusion review allocation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("E:/Kaggle/exam-output-wildlife-v6/wildlife")
OUTPUT = Path(__file__).resolve().parents[2] / "audits" / "occlusion-sampling-2026-10-05"
Z = 1.959963984540054
TARGET_MARGIN = 0.05
REFERENCE = "https://www.rti.org/publication/practools-computations-design-finite-population-samples/fulltext.pdf"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    paths = [ROOT / "manifest.json", ROOT / "occlusion_review.csv", ROOT / "label_review.csv"]
    before = {path.name: sha256(path) for path in paths}
    manifest = json.loads(paths[0].read_text(encoding="utf-8"))
    with paths[1].open(encoding="utf-8-sig", newline="") as stream:
        reviews = list(csv.DictReader(stream))
    review_ids = {row["annotation_id"] for row in reviews}
    assert len(review_ids) == len(reviews), "Duplicate review IDs"
    objects = []
    for record in manifest["records"]["trans_test"]:
        for box in record["boxes"]:
            if box["annotation_id"] not in review_ids:
                continue
            area = box["box"][2] * box["box"][3] * 1e6
            size = "small" if area < 10000 else "medium" if area <= 50000 else "large"
            objects.append(
                {
                    "annotation_id": box["annotation_id"],
                    "image_id": record["image_id"],
                    "sequence_id": record["sequence_id"],
                    "camera_id": record["camera_id"],
                    "species": manifest["names"][box["class_id"]],
                    "size": size,
                }
            )
    assert {obj["annotation_id"] for obj in objects} == review_ids
    total = len(objects)
    assert total == len(reviews)
    counts = Counter((obj["camera_id"], obj["species"], obj["size"]) for obj in objects)
    counts = dict(sorted(counts.items()))
    allocation = {key: min(2, count) for key, count in counts.items()}

    def variance_bound(quotas: dict) -> float:
        # S_h^2 <= N_h / (4 * (N_h - 1)) for binary outcomes.
        # Var(p_hat) = sum(W_h^2 * (1 - n_h/N_h) * S_h^2 / n_h).
        # Population size-one strata are censused and contribute no variance.
        return sum(
            (count / total) ** 2 * (count - quotas[key]) / (4 * quotas[key] * (count - 1))
            for key, count in counts.items()
            if count > 1
        )

    minimum_coverage = sum(allocation.values())
    previous_margin = None
    while Z * math.sqrt(variance_bound(allocation)) > TARGET_MARGIN:
        previous_margin = Z * math.sqrt(variance_bound(allocation))
        candidates = [key for key, count in counts.items() if allocation[key] < count]
        # Equal review cost and decreasing marginal variance gains make this
        # integer greedy allocation optimal under the declared lower bounds.
        key = max(
            candidates,
            key=lambda key: (counts[key] / total) ** 2
            * counts[key]
            / (4 * (counts[key] - 1) * allocation[key] * (allocation[key] + 1)),
        )
        allocation[key] += 1
    recommended = sum(allocation.values())
    margin = Z * math.sqrt(variance_bound(allocation))
    assert recommended == 600
    assert margin <= TARGET_MARGIN < previous_margin
    assert all(min(2, count) <= allocation[key] <= count for key, count in counts.items())
    assert len(allocation) == len(counts) == 209

    srs_raw = total * Z**2 * 0.25 / (TARGET_MARGIN**2 * (total - 1) + Z**2 * 0.25)
    summaries = {}
    for field, index in (("camera_id", 0), ("species", 1), ("size", 2)):
        population = Counter(obj[field] for obj in objects)
        sample = Counter()
        for key, quota in allocation.items():
            sample[key[index]] += quota
        summaries[field] = {
            value: {"population": population[value], "proposed_sample": sample[value]}
            for value in sorted(population)
        }
    output = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "size_and_allocation_proposal_only_no_sample_selected",
        "population": {
            "partition": "trans_test",
            "unit": "annotated animal appearance, not a unique individual animal",
            "objects": total,
            "images_with_boxes": len({obj["image_id"] for obj in objects}),
            "sequences_with_boxes": len({obj["sequence_id"] for obj in objects}),
            "cameras": len(summaries["camera_id"]),
            "species_present": len(summaries["species"]),
            "species_absent": sorted(set(manifest["names"]) - set(summaries["species"])),
            "reviewed_objects": sum(bool(row["level"]) for row in reviews),
            "manifest_identity": manifest["identity"],
        },
        "assumptions": {
            "confidence_level": 0.95,
            "z_normal": Z,
            "target_marginal_margin_proportion": TARGET_MARGIN,
            "binary_variance_worst_case": 0.25,
            "target_estimand": "finite-population proportion of each occlusion label, including ambiguity",
            "strata": ["camera_id", "species", "size"],
            "size_definitions": {"small": "area < 1%", "medium": "1% <= area <= 5%", "large": "area > 5%"},
            "minimum_per_non_census_stratum": 2,
            "selection": "independent simple random sampling without replacement within each stratum",
            "selection_excludes_model_predictions": True,
            "planned_random_seed": 42,
            "estimator": "sum_h (N_h / N) * label_proportion_h; equivalent object weights N_h/n_h",
            "normal_interval_is_approximate_and_marginal_not_simultaneous": True,
        },
        "calculations": {
            "simple_random_n_unrounded": srs_raw,
            "simple_random_n_rounded_up": math.ceil(srs_raw),
            "observed_joint_strata": len(counts),
            "singleton_strata": sum(count == 1 for count in counts.values()),
            "minimum_coverage_total": minimum_coverage,
            "recommended_sample": recommended,
            "recommended_fraction_percent": 100 * recommended / total,
            "stratified_worst_case_margin_percent": 100 * margin,
            "margin_percent_one_case_smaller": 100 * previous_margin,
            "stratified_variance_bound": variance_bound(allocation),
            "variance_bound_formula": "sum_h (N_h/N)^2 * (N_h-n_h) / (4*n_h*(N_h-1)); singleton census terms zero",
        },
        "distributions": summaries,
        "proposed_allocation": [
            {
                "camera_id": key[0],
                "species": key[1],
                "size": key[2],
                "population": counts[key],
                "sample": quota,
                "inclusion_probability": quota / counts[key],
                "object_weight": counts[key] / quota,
            }
            for key, quota in allocation.items()
        ],
        "limits": [
            "This target precision applies to overall weighted occlusion proportions, not mAP, recall, per-species or per-camera estimates.",
            "No human labels exist yet, so 30 low, medium and high cases cannot be guaranteed. Thirty is only the current local support gate, not a representativeness proof.",
            "The weighted design describes the existing trans_test object population. It does not guarantee performance on future cameras or animal identities.",
            "Frames from the same burst remain correlated for performance inference; resample whole sequences when reporting sampled detection uncertainty.",
            "Do not reject repeated sequences or replace difficult cases after drawing the sample without revising inclusion probabilities.",
            "Supplementary targeted cases, if needed, must be documented separately from the fixed probability sample; do not pool their unweighted proportions.",
            "Ambiguous cases are recorded as an outcome, not silently replaced or excluded from the population denominator.",
            "The existing COCO evaluation has no survey weighting implementation. Future sample mAP must be explicitly reported as a sample analysis, not a weighted population mAP.",
            "No frozen configuration, source, model, prediction, notebook, PDF or human review was changed.",
        ],
        "reference": {"title": "PracTools: Computations for Design of Finite Population Samples", "authors": "Richard Valliant, Jill A. Dever, Frauke Kreuter", "year": 2015, "url": REFERENCE},
        "source_sha256": before,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "analysis.json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def table(field: str, heading: str) -> str:
        lines = [f"## {heading}", "", "| Grupo | Población | Muestra propuesta |", "| --- | ---: | ---: |"]
        for value, entry in summaries[field].items():
            lines.append(f"| {value} | {entry['population']:,} | {entry['proposed_sample']} |")
        return "\n".join(lines)

    note = f"""# Muestra propuesta para revisar la oclusión: {recommended} casos

Análisis del CSV y manifiesto reales de fauna V6. Es una propuesta de tamaño y reparto: todavía no se seleccionaron casos ni se registraron valoraciones humanas. Los casos son apariciones de animales anotadas en recuadros, no individuos únicos.

La población de esta revisión es `trans_test`: {total:,} recuadros en {output['population']['images_with_boxes']:,} fotos y {output['population']['sequences_with_boxes']:,} secuencias, procedentes de nueve cámaras y 13 especies. `deer` está ausente en esta partición. La revisión de etiquetas de 90 imágenes es otro proceso.

## Cálculo y decisión

Se adopta como objetivo una confianza aproximada del 95 % y un margen marginal de ±5 puntos porcentuales para estimar la proporción global de cada nivel de oclusión, con el peor caso de variabilidad binaria (`p=0.5`). Esta precisión es una decisión metodológica propuesta.

Para muestreo aleatorio simple sin reemplazo:

`n = N*z²*p*(1-p) / [e²*(N-1) + z²*p*(1-p)]`

Con `N=17620`, `z≈1.96`, `p=0.5` y `e=0.05`, el resultado es `{srs_raw:.6f}`, que se redondea hacia arriba a **376**. Es el mínimo de referencia para porcentajes generales bajo ese diseño.

Los datos contienen **209 combinaciones presentes de cámara × especie × tamaño**. Para cubrirlas, el diseño propuesto asigna al menos dos casos a cada combinación; las {output['calculations']['singleton_strata']} combinaciones con un solo caso se revisan completas. Esto consume **{minimum_coverage} casos** antes de distribuir los restantes según la reducción de la varianza máxima.

Con selección aleatoria independiente dentro de cada combinación y porcentajes ponderados según su población, **600 casos** alcanzan un margen previsto de **±{100*margin:.4f} puntos**; con 599 es ±{100*previous_margin:.4f}. Por ello se recomiendan 600 para este diseño particular, equivalentes al {100*recommended/total:.2f} % de los recuadros. El mínimo de dos por combinación da cobertura y no garantiza precisión dentro de cada combinación.

Se propone semilla 42. La selección utilizaría únicamente las anotaciones originales y metadatos de cámara, especie y tamaño. Las predicciones del detector no intervienen en la selección. Los porcentajes globales requieren pesos `N_h/n_h`, porque las combinaciones escasas quedan sobrerrepresentadas. El reparto de las 209 combinaciones y sus pesos están en [analysis.json](analysis.json).

## Alcance y condición de finalización

La precisión calculada corresponde a proporciones globales de oclusión en esta población fija; no asegura un margen de ±5 puntos en mAP, por especie o por cámara. El mAP del subconjunto sería un análisis de muestra y tendría sus propios intervalos agrupados por secuencia. El evaluador COCO existente no incorpora estos pesos de encuesta.

Todavía no sabemos cuántos casos tienen oclusión alta: los {total:,} niveles humanos están vacíos. El protocolo local pide al menos 30 casos en cada nivel bajo, medio y alto, pero ese umbral no demuestra representatividad. Si un nivel queda por debajo de 30, cualquier suplemento se documentaría separado de esta muestra probabilística fija. No se mezclan sin ponderación los casos añadidos de forma dirigida con los porcentajes de la muestra base.

Las fotos de una misma ráfaga pueden estar correlacionadas; los casos no son individuos independientes. El muestreo aleatorio propuesto estima la población fija de recuadros y conserva probabilidades conocidas. Evitar repeticiones de secuencia mediante reemplazos posteriores alteraría esas probabilidades. Para incertidumbre de rendimiento se requiere considerar la secuencia como grupo.

Los casos ambiguos se conservan como resultado y no se sustituyen en silencio. Se registran nivel, evidencia visual y autoría humanos. La configuración V6 mantiene su revisión completa original: una ejecución por muestra necesitaría un protocolo separado. No se modifica V6 ni se vuelve a entrenar.

{table('camera_id', 'Distribución propuesta por cámara')}

{table('species', 'Distribución propuesta por especie')}

{table('size', 'Distribución propuesta por tamaño')}

Tamaños iguales a los del evaluador actual: pequeño <1 % de área de imagen, mediano entre 1 % y 5 % inclusive y grande >5 %.

## Fuente metodológica y trazabilidad

Valliant, Dever y Kreuter (2015), [PracTools: Computations for Design of Finite Population Samples]({REFERENCE}), ecuación (2) y sección de muestreo estratificado. La elección de las cuotas mínimas y el cálculo sobre los datos de fauna son específicos de este análisis.

Los hashes de `manifest.json`, `occlusion_review.csv` y `label_review.csv` se verificaron iguales antes y después. No se cambiaron el entrenamiento, los resultados, las revisiones humanas, el notebook ni el PDF. Los datos y cálculos auditables están en `analysis.json`.
"""
    (OUTPUT / "calculo_muestra_occlusion.md").write_text(note, encoding="utf-8")
    after = {path.name: sha256(path) for path in paths}
    assert before == after, "Source files changed during analysis"
    print(json.dumps({"population": output["population"], "calculations": output["calculations"], "distributions": summaries, "sources_unchanged": before == after, "output": str(OUTPUT / "calculo_muestra_occlusion.md")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
