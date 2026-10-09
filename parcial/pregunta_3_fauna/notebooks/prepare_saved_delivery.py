"""Prepara una copia portable de resultados guardados; no calcula predicciones."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V6 = Path("E:/Kaggle/exam-output-wildlife-v6")
OCCLUSION = Path("E:/Kaggle/exam-output-wildlife-v6-occlusion-sample-600")
ORIGINAL = Path("D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial")
SIGNATURE = "6b2983c33abccbc18e293cc3a10ca30a710669c2ba8055a76bb462b3348a13a9"


def read_json(path: Path) -> dict:
    """Lee un archivo de evidencia sin modificarlo."""
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    """Calcula SHA-256 sobre bytes físicos."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    """Escribe únicamente dentro de la carpeta de fauna del worktree."""
    assert path.resolve().is_relative_to(ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Verifica entradas, copia evidencias y adapta el notebook V6 existente."""
    delivery = ROOT / "entregable"
    data = delivery / "data"
    provenance = {}

    def copy(source: Path, target: Path) -> None:
        assert target.resolve().is_relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        assert digest(source) == digest(target)
        provenance[str(source)] = {
            "destination": target.relative_to(ROOT).as_posix(),
            "sha256": digest(source),
        }

    result_path = OCCLUSION / "accepted_evaluation/6b2983c33abccbc1/results.json"
    results = read_json(result_path)
    assert results["signature"] == SIGNATURE
    assert results["status"] == "evaluation_complete_with_documented_review_deviations"
    assert digest(result_path) == digest(OCCLUSION / "accepted_delivery/data/results.json")
    for level, support in {"low": 340, "medium": 69, "high": 48}.items():
        row = results["sample_detection"][level]
        assert row["metrics"]["eligible_objects"] == support
        assert row["uncertainty"]["replicates"] == 200
        assert all(x["defined_replicates"] == 200 for x in row["uncertainty"]["intervals"].values())

    for source in (OCCLUSION / "accepted_delivery/data").iterdir():
        copy(source, data / source.name)
    for name in ["04_occlusion_582.ipynb", "delivery_manifest.json"]:
        target = delivery / name if name.endswith("ipynb") else ROOT / "evidencias/occlusion_delivery_original.json"
        copy(OCCLUSION / "accepted_delivery" / name, target)
    copy(OCCLUSION / "accepted_delivery/occlusion_metrics_582.csv", delivery / "metricas_occlusion.csv")

    v6_names = [
        "wildlife/generalization.json", "wildlife/uncertainty.json", "wildlife/robustness.json",
        "wildlife/activity.json", "wildlife/cam_cases.json", "wildlife/class_balance.png",
        "wildlife/training_curves.png", "wildlife/cam_cis_test_grid.png",
        "wildlife/cam_trans_test_grid.png", "wildlife/label_contact_01.png",
        "wildlife/experiments.csv", "wildlife/cis_test/metrics.json", "wildlife/trans_test/metrics.json",
        "wildlife/label_review.csv", "wildlife/label_review_status.json", "wildlife/review_migration.json",
        "wildlife/activity_camera_days.csv", "wildlife/activity_species_days.csv", "wildlife/activity_sequence.csv",
        "wildlife/yolov8s/results.csv", "wildlife/yolov8s/args.yaml",
        "receipts/doctor_gpu.json", "receipts/amp.json", "receipts/source-freeze.json",
        "receipts/architecture-review.json", "receipts/yolo-loader-verification.json",
    ]
    for name in v6_names:
        copy(V6 / name, data / "v6" / name)

    manifest = read_json(V6 / "wildlife/manifest.json")
    summary = {key: manifest[key] for key in [
        "identity", "bbox_units", "integrity", "split_summary", "sequence_overlaps",
        "class_map", "train_class_boxes", "jpeg_materialization",
    ]}
    counts = Counter((row["split"], row["reason"]) for row in manifest["exclusions"])
    summary["exclusion_counts"] = [
        {"split": split, "reason": reason, "images": count}
        for (split, reason), count in sorted(counts.items())
    ]
    summary["derivation"] = {
        "source": str(V6 / "wildlife/manifest.json"),
        "sha256": digest(V6 / "wildlife/manifest.json"),
        "method": "Campos originales seleccionados; exclusiones agrupadas por partición y motivo. No contiene records."
    }
    write_json(data / "v6/wildlife/manifest_summary.json", summary)
    frozen = read_json(V6 / "receipts/source-freeze.json")["sha256"]
    physical = []
    for name, expected in frozen.items():
        original = ORIGINAL / name
        local = ROOT.parent / name
        assert digest(original) == expected, name
        original_bytes, local_bytes = original.read_bytes(), local.read_bytes()
        assert original_bytes.replace(b"\r\n", b"\n") == local_bytes.replace(b"\r\n", b"\n"), name
        physical.append({
            "path": name, "expected": expected, "original": digest(original),
            "worktree": digest(local), "same_after_newline_normalization": True,
        })
    write_json(ROOT / "evidencias/fuentes_congeladas.json", {"files": physical})
    input_hashes = {str(V6 / "notebooks/03_wildlife.ipynb"): digest(V6 / "notebooks/03_wildlife.ipynb"),
                    str(V6 / "reports/wildlife_draft.pdf"): digest(V6 / "reports/wildlife_draft.pdf"),
                    str(V6 / "wildlife/manifest.json"): digest(V6 / "wildlife/manifest.json"),
                    str(V6 / "wildlife/yolov8s/weights/best.pt"): digest(V6 / "wildlife/yolov8s/weights/best.pt")}
    assert input_hashes[str(V6 / "wildlife/yolov8s/weights/best.pt")] == results["checkpoint_sha256"]
    write_json(ROOT / "evidencias/procedencia.json", {"copied_files": provenance, "preserved_inputs": input_hashes})
    hashes = {p.relative_to(data).as_posix(): digest(p) for p in sorted(data.rglob("*")) if p.is_file()}
    write_json(delivery / "integridad_datos.json", {"sha256": hashes, "signature": SIGNATURE})

    notebook = read_json(V6 / "notebooks/03_wildlife.ipynb")

    def replace(index: int, source: str) -> None:
        notebook["cells"][index]["source"] = source.strip().splitlines(keepends=True)

    replace(0, """# Pregunta 3: fauna CCT20 y oclusión aceptada

Notebook final de consulta de resultados V6 guardados y de la revisión independiente de 582 casos.
No entrena, prepara datos científicos ni ejecuta inferencia. Ejecutar desde `entregable/`, junto con `data/`
e `integridad_datos.json`. Requiere Python, pandas e IPython; no importa PyTorch ni Ultralytics.
El notebook y las predicciones V6 originales permanecen intactos. Las salidas de este documento se vuelven a ejecutar.""")
    replace(2, '''from pathlib import Path
import csv
import hashlib
import json
from collections import Counter

import pandas as pd
from IPython.display import Image, display

delivery = Path.cwd()
if not (delivery / "integridad_datos.json").is_file():
    raise FileNotFoundError("Ejecutar desde la carpeta entregable, junto con data/.")
data = delivery / "data"
integrity = json.loads((delivery / "integridad_datos.json").read_text(encoding="utf-8"))
for name, expected in integrity["sha256"].items():
    assert hashlib.sha256((data / name).read_bytes()).hexdigest() == expected, name
v6 = data / "v6"
output = v6 / "wildlife"

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def show_json(path):
    display(read_json(path))

def show_figure(path, width=900):
    display(Image(filename=str(path), width=width))

print(f"Integridad verificada: {len(integrity['sha256'])} archivos de evidencia.")''')
    replace(3, """## Entorno y CUDA de la ejecución V6

Las constancias siguientes son evidencias guardadas de V6. Consultarlas no inicia operaciones GPU ni
vuelve a validar el entorno local de lectura. La prueba de reanudación histórica no implica equivalencia bit a bit.""")
    replace(4, '''for name in ["doctor_gpu.json", "amp.json"]:
    show_json(v6 / "receipts" / name)''')
    replace(6, '''manifest = read_json(output / "manifest_summary.json")
display({"identity": manifest["identity"], "bbox_units": manifest["bbox_units"], **manifest["integrity"]})
display(pd.DataFrame([{"split": split, "images": row["images"], "backgrounds": row["backgrounds"],
                      "cameras": len(row["cameras"]), "sequences": row["sequences"]}
                     for split, row in manifest["split_summary"].items()]))
display(pd.DataFrame(list(manifest["sequence_overlaps"].items()), columns=["partition_pair", "shared_sequences"]))
display(pd.DataFrame(manifest["class_map"]))
display(manifest["derivation"])''')
    replace(7, '''display(pd.DataFrame(manifest["exclusion_counts"]))
display(pd.DataFrame(list(manifest["train_class_boxes"].items()), columns=["species", "training_boxes"]))
show_figure(output / "class_balance.png")''')
    replace(8, """## Revisión humana antes del entrenamiento V6

Se conservan las 90 fotografías aprobadas por Miguel, incluida la corrección de la tercera caja de ave,
sin modificar las otras dos. Las notas individuales eran opcionales. La migración V6 conserva fotografías
y cajas aprobadas. La decisión de ajustar YOLOv8s existente y el control del cargador se consultan como
constancias históricas; este notebook no modifica revisiones ni galerías.""")
    replace(9, '''labels = pd.read_csv(output / "label_review.csv").fillna("")
assert len(labels) == 90
show_json(output / "label_review_status.json")
show_figure(output / "label_contact_01.png", width=1100)
show_json(v6 / "receipts/architecture-review.json")
show_json(output / "review_migration.json")
display(manifest["jpeg_materialization"])
show_json(v6 / "receipts/yolo-loader-verification.json")''')
    replace(10, """## Smoke aislado: alcance histórico

La prueba pequeña no aporta cifras al informe. El notebook V6 original ya indicaba que sus resultados
no estaban disponibles en esta raíz de salida; no se generan ni se atribuyen cifras nuevas a esa prueba.""")
    replace(11, '''print("Smoke: no incorporado; no interviene en las métricas científicas guardadas.")''')
    replace(12, """## Entrenamiento completo guardado

V6 terminó en 39 épocas y seleccionó el checkpoint de la época 29 mediante validación.
Las curvas y argumentos se conservan; este notebook no contiene una ruta de entrenamiento o inferencia.""")
    replace(13, '''training = pd.read_csv(output / "yolov8s/results.csv")
assert len(training) == 39
print(f"Épocas registradas: {len(training)}; checkpoint seleccionado: época 29 (V6).")
print((output / "yolov8s/args.yaml").read_text(encoding="utf-8"))
show_figure(output / "training_curves.png")''')
    replace(15, '''display(pd.read_csv(output / "experiments.csv"))
for split in ["cis_test", "trans_test"]:
    metrics = read_json(output / split / "metrics.json")
    display({"split": split, "map50": metrics["map50"], "map50_95": metrics["map50_95"],
             "protocol": metrics["protocol"]})
    display(pd.DataFrame(metrics["per_species"]))
show_json(output / "generalization.json")
uncertainty = read_json(output / "uncertainty.json")
display({key: uncertainty[key] for key in ["global", "size", "limitations"]})''')
    replace(16, """## Tamaño relativo: resultados V6

Pequeño <1 %, mediano 1–5 %, grande >5 % del área de la foto, en ambos tests.
Las cifras se conservan sin cambios. `robustness.json` es histórico: su estado de oclusión corresponde
a la galería completa original; el complemento aceptado se presenta en la sección siguiente.""")
    replace(17, '''robustness = read_json(output / "robustness.json")
for split, groups in robustness["size_by_split"].items():
    display(pd.DataFrame([{"split": split, "group": group, "objects": score["eligible_objects"],
                          "map50": score["map50"], "map50_95": score["map50_95"]}
                         for group, score in groups.items()]))''')
    replace(22, """## Informe, seis entregables y límites

Se conservan el detector ajustado y sus decisiones, métricas por especie y desbalance, comparación
conocidas/nuevas y doce Grad-CAM, robustez por tamaño y oclusión aceptada, monitoreo ecológico
y discusión responsable. La AP global V6 no es comparable directamente con la AP de la submuestra.
Las detecciones no son individuos; CCT20 no valida despliegue en Perú. No se atribuye aprobación del docente.
El informe principal se compone por separado en ODT y se exporta a PDF; el anexo de oclusión permanece independiente.""")
    replace(23, '''assert len(read_json(output / "cam_cases.json")) == 12
activity = read_json(output / "activity.json")
assert sum(row["photos"] for row in activity["camera_days"]) == 1853
print("Resultados guardados de V6 y complemento de oclusión consultados; sin entrenamiento ni inferencia.")''')
    annex = read_json(OCCLUSION / "accepted_delivery/04_occlusion_582.ipynb")
    added = annex["cells"][:8]
    added[0]["source"] = ["## Oclusión: 582 casos aceptados\n\n",
        "Se usan todos los 18.647 fotogramas de trans_test, incluidos 1.778 vacíos. La AP no usa pesos de muestreo.\n",
        "Las cajas no revisadas, excluidas, ambiguas o de otro nivel se ignoran como crowd.\n",
        "Las coincidencias de la misma clase pueden ignorarse; las detecciones no absorbidas aún pueden ser falsos positivos.\n",
        "La reducción del soporte con estos fotogramas cambia la tarea: la mAP no es comparable directamente con la global V6.\n"]
    added[5]["source"] = "".join(added[5]["source"]).replace(
        '0 < bounds["defined_replicates"] <= 200', 'bounds["defined_replicates"] == 200').splitlines(keepends=True)
    for level in ["low", "medium", "high"]:
        added[5]["source"].append(f'\ndisplay(pd.DataFrame(results["sample_detection"]["{level}"]["metrics"]["per_species"]).assign(level="{level}"))\n')
    added.append({"cell_type": "markdown", "metadata": {}, "source": [
        "### Comparaciones y límites de oclusión\n\n",
        "mAP50:95 baja 6,04 puntos porcentuales de baja a media y 6,31 de baja a alta. ",
        "Media y alta difieren 0,28 puntos y sus intervalos se superponen ampliamente; no se calculó IC de la diferencia. ",
        "Las clases con soporte son 13, 10 y 11: composición y soporte difieren. No es un efecto causal aislado. ",
        "Los IC no incluyen variación de entrenamiento, desacuerdo de anotación ni cámaras futuras.\n"]})
    notebook["cells"][18:18] = added
    for index, cell in enumerate(notebook["cells"]):
        cell["id"] = f"fauna-final-{index:02d}"
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    notebook["metadata"]["execution_policy"] = "Solo lectura de resultados guardados verificados; sin entrenamiento ni inferencia"
    write_json(ROOT / "notebooks/03_wildlife.ipynb", notebook)
    write_json(delivery / "03_wildlife.ipynb", notebook)
    print(f"Entradas verificadas: 44 fuentes; {len(hashes)} archivos portables; 3 niveles completos.")


if __name__ == "__main__":
    main()
