"""Populate skill-scaffolded notebooks; this authoring helper runs no experiments."""

import json
from pathlib import Path

TASKS = [
    (
        "01_mri",
        "mri",
        "MRI de rodilla",
        "Comparar CNN propia y ResNet-50, BCE/focal y manejo de ruido.",
        "Los 58 estudios expertos quedan separados de la selección del modelo. La confianza textual es heurística.",
        "Revisar AUC por etiqueta, cobertura de reglas y Grad-CAM para LCA, menisco medial y derrame.",
    ),
    (
        "02_flood",
        "flood",
        "Inundaciones Sentinel-1",
        "Comparar supervisión manual/manual más débil y BCE/Dice/combinada.",
        "Bolivia se excluye de entrenamiento y estadísticas. Los píxeles sin datos no entran en pérdidas ni métricas.",
        "Revisar IoU promedio por chip y métricas globales; documentar tres tipos de fallo SAR con evidencia.",
    ),
    (
        "03_wildlife",
        "wildlife",
        "Fauna CCT20",
        "Evaluar YOLOv8s en cámaras vistas/nuevas y generar actividad ecológica.",
        "Vacíos son fondo; animales sin cajas completas se excluyen; se depuran secuencias compartidas con validación.",
        "Revisar AP por especie, tamaño relativo, oclusión manual y actividad por cámara, especie y día.",
    ),
]


def cell(kind, content, index):
    result = {"cell_type": kind, "id": f"cell-{index:02d}", "metadata": {}, "source": content.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def main() -> None:
    folder = Path(__file__).resolve().parents[1] / "notebooks"
    for stem, task, title, hypothesis, integrity, review in TASKS:
        if task == "wildlife":
            from populate_wildlife_notebook import create_notebook

            create_notebook()
            continue
        path = folder / f"{stem}.ipynb"
        document = json.loads(path.read_text(encoding="utf-8"))
        contents = [
            (
                "markdown",
                f"# Pregunta: {title}\n\n**Estado: preparado, sin ejecutar.**\n\n{hypothesis}\n\n"
                "Las celdas no incluyen resultados. Los controles siguientes evitan iniciar experimentos al abrir el archivo.\n",
            ),
            (
                "markdown",
                "## Configuración y decisiones previas\n\nSemilla 42, hasta 50 épocas y paciencia 10. "
                "Los resultados científicos y smoke usan directorios diferentes. Para ejecución completa se recomienda el comando del README.\n",
            ),
            (
                "code",
                "from pathlib import Path\nimport os\nimport pandas as pd\nfrom IPython.display import display, Image\n"
                "from neural_exam.config import load_settings\nfrom neural_exam.cli import doctor\n"
                "from neural_exam.common.io import read_json\nfrom neural_exam.common.locking import execution_lock\n"
                f"from neural_exam.{task}.data import prepare\n"
                f"from neural_exam.{task}.pipeline import run\n"
                "from neural_exam.reporting import prepare_report_assets\n\n"
                "RUN_PREPARE = False\nRUN_TRAINING = False\nRESUME = False\nGENERATE_REPORT_ASSETS = False\n"
                "settings = load_settings()\n"
                "os.environ.setdefault('YOLO_OFFLINE', 'true')\n"
                "os.environ.setdefault('YOLO_AUTOINSTALL', 'false')\n"
                f"task = '{task}'\noutput = settings.output_root / task\n",
            ),
            (
                "markdown",
                "## Diagnóstico sin operaciones GPU\n\nLa comprobación CUDA se realizará por separado mediante "
                "doctor --gpu-check. Esta celda consulta versiones y rutas.\n",
            ),
            ("code", "diagnostics = doctor(settings, gpu_check=False)\ndisplay(diagnostics)\n"),
            (
                "markdown",
                f"## Integridad y preparación\n\n{integrity}\n\n"
                "Activar RUN_PREPARE únicamente después de completar las descargas y extracción. Preparar lee datos y escribe derivados locales.\n",
            ),
            (
                "code",
                "manifest = None\nif RUN_PREPARE:\n    with execution_lock(settings.output_root):\n"
                "        manifest = prepare(settings)\nelif (output / 'manifest.json').is_file():\n"
                "    manifest = read_json(output / 'manifest.json')\n"
                "else:\n    print('Preparación pendiente: no hay manifiesto de datos.')\n",
            ),
            (
                "markdown",
                "## Inventario de particiones\n\nLos conteos provienen del manifiesto real; no se usan números simulados.\n",
            ),
            (
                "code",
                "if manifest:\n    records = manifest['records']\n"
                "    if isinstance(records, dict):\n"
                "        counts = {split: len(rows) for split, rows in records.items()}\n"
                "    else:\n"
                "        from collections import Counter\n"
                "        counts = dict(Counter(row['split'] for row in records))\n"
                "    display(pd.DataFrame(list(counts.items()), columns=['partition', 'examples']))\n",
            ),
            (
                "markdown",
                "## Experimentos\n\nEsta celda importa la implementación del paquete. "
                "No duplica arquitectura, entrenamiento o evaluación. No ejecutar simultáneamente con otros entrenamientos.\n",
            ),
            (
                "code",
                "if RUN_TRAINING:\n    with execution_lock(settings.output_root):\n"
                "        experiment_summary = run(settings, resume=RESUME, quick=False, manifest=manifest)\n"
                "    display(experiment_summary)\nelse:\n    print('Entrenamiento pendiente; RUN_TRAINING permanece desactivado.')\n",
            ),
            (
                "markdown",
                f"## Evaluación y revisión\n\n{review}\n\n"
                "No elegir hiperparámetros por rendimiento de prueba. Una métrica ND debe conservarse como indefinida.\n",
            ),
            (
                "code",
                "if (output / 'experiments.csv').is_file():\n"
                "    display(pd.read_csv(output / 'experiments.csv'))\n"
                "else:\n    print('No hay mediciones de experimentos completos.')\n",
            ),
            (
                "code",
                "figures = sorted(output.rglob('*.png'))\n"
                "for figure in figures[:6]:\n"
                "    display(Image(filename=str(figure), width=700))\n",
            ),
            (
                "markdown",
                "## Fuentes del informe\n\nLas tablas y figuras se exportan desde ejecuciones reales. "
                "Completar la discusión y la revisión humana antes de compilar el informe final.\n",
            ),
            ("code", "if GENERATE_REPORT_ASSETS:\n    display(prepare_report_assets(settings, task))\n"),
            (
                "markdown",
                "## Limitaciones y conclusiones pendientes\n\n"
                "Registrar aquí las observaciones sustentadas en resultados, posibles sesgos y próximos experimentos. "
                "No completar conclusiones empíricas antes de ejecutar y revisar.\n",
            ),
        ]
        document["cells"] = [cell(kind, content, index) for index, (kind, content) in enumerate(contents)]
        document["metadata"].update(
            preparation_status="not_executed",
            kernelspec={"display_name": "Python 3", "language": "python", "name": "python3"},
        )
        path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
