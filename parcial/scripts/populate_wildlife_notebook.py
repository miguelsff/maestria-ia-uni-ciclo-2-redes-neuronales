"""Build the wildlife notebook with explicit gates and persisted evidence only."""

import json
from pathlib import Path


def create_notebook() -> Path:
    path = Path(__file__).resolve().parents[1] / "notebooks/03_wildlife.ipynb"
    contents = [
        (
            "markdown",
            "# Pregunta 3: fauna CCT20\n\nEste notebook carga auditorías y resultados reales desde la salida de fauna. Las tablas científicas se muestran cuando termina su evaluación real; cada revisión humana pendiente conserva su estado explícito. La prueba pequeña (smoke) se identifica por separado y no aporta cifras al informe.\n",
        ),
        (
            "markdown",
            "## Protocolo v6\n\nYOLOv8s preentrenado, 14 clases animales, semilla 42, resolución 640, batch 16, hasta 50 épocas y paciencia 10. AdamW: tasa inicial 3e-4 y weight decay 1e-4. Se conservan los 1.000 fondos oficiales adicionales de v2 y todos los holdouts. V5 corrige la tercera caja de una foto de train según el recuadro rojo proporcionado por el revisor; conserva las otras dos cajas. Conserva las 89 aprobaciones sin exigir notas por fotografía, conforme a la confirmación humana registrada. Compose v6 selecciona configs/wildlife_v6.toml. V6 conserva los píxeles y las etiquetas aprobadas de V5: para JPEG con espacios tras el marcador final, genera una copia sin esos espacios y verifica igualdad exacta de los píxeles decodificados. El comprobador real de Ultralytics valida todas las imágenes y cajas antes de entrenar. cis_val depurado selecciona best.pt. El usuario definió usar el detector existente de Ultralytics y ajustarlo con CCT20. Esa decisión queda en receipts/architecture-review.json, junto con la revisión completa de las 90 fotos. Las tablas y mapas científicos se generan a partir de la ejecución real.\n",
        ),
        (
            "code",
            "from pathlib import Path\nimport os\nimport pandas as pd\nfrom IPython.display import HTML, Image, display\nfrom neural_exam.config import load_settings\nfrom neural_exam.common.io import read_json\nfrom neural_exam.common.locking import execution_lock\nfrom neural_exam.wildlife.data import prepare\nfrom neural_exam.wildlife.review import label_review_status, prepare_label_review\nfrom neural_exam.reporting import prepare_report_assets\n\nRUN_PREPARE = False\nRUN_TRAINING = False\nRESUME = False\nGENERATE_REPORT_ASSETS = False\nsettings = load_settings()\noutput = settings.output_root / 'wildlife'\nos.environ.setdefault('YOLO_OFFLINE', 'true')\nos.environ.setdefault('YOLO_AUTOINSTALL', 'false')\nPath(os.environ.setdefault('YOLO_CONFIG_DIR', str(settings.output_root / 'cache/ultralytics'))).mkdir(parents=True, exist_ok=True)\n\ndef show_json(path):\n    if path.exists():\n        display(read_json(path))\n    else:\n        print(f'Pendiente: {path.name}')\n\ndef show_figure(path, width=900):\n    if path.exists():\n        display(Image(filename=str(path), width=width))\n    else:\n        print(f'Figura pendiente: {path.name}')\n",
        ),
        (
            "markdown",
            "## Entorno, CUDA y reanudación\n\nLas constancias siguientes provienen de las comprobaciones explícitas. Esta celda no inicia operaciones GPU. La interrupción sintética v3 verifica época, estados del optimizador y paciencia mediante un archivo ligado al hash de last.pt; no afirma equivalencia bit a bit ni rendimiento científico.\n",
        ),
        (
            "code",
            "for path in [settings.output_root / 'receipts/doctor_gpu.json', settings.output_root / 'receipts/amp.json', settings.output_root / 'resume-check/passed.json']:\n    show_json(path)\n",
        ),
        (
            "markdown",
            "## Integridad y particiones\n\nSe decodifican las imágenes, se registra SHA-256 y se comprueban dimensiones y aspecto respecto a metadatos originales. Las cajas se normalizan con las dimensiones originales. Los JPG se enlazan en un volumen Linux; las imágenes de origen permanecen en solo lectura. Se excluyen animales sin cajas completas y vehículos sin animales. Solo empty explícito genera etiquetas vacías. El train ECCV oficial no contiene empty y todas las fotos vacías de sus diez cámaras estaban reservadas. La ampliación toma 125 secuencias independientes por cada una de ocho cámaras del train de la colección completa, con selección basada en metadatos. No modifica los holdouts. Los negativos tienen resolución original y proceden de cámaras sin positivos añadidos: ambas diferencias deben declararse al interpretar los resultados.\n",
        ),
        (
            "code",
            "manifest = None\nif RUN_PREPARE:\n    with execution_lock(settings.output_root):\n        manifest = prepare(settings)\n        prepare_label_review(manifest, output, settings.training['seed'])\nelif (output / 'manifest.json').exists():\n    manifest = read_json(output / 'manifest.json')\nelse:\n    print('Preparación pendiente.')\n\nif manifest:\n    display({'identity': manifest['identity'], 'bbox_units': manifest['bbox_units'], **manifest['integrity']})\n    display(pd.DataFrame([{'split': split, 'images': row['images'], 'backgrounds': row['backgrounds'], 'cameras': len(row['cameras']), 'sequences': row['sequences']} for split, row in manifest['split_summary'].items()]))\n    display(pd.DataFrame(list(manifest['sequence_overlaps'].items()), columns=['partition_pair', 'shared_sequences']))\n    display(pd.DataFrame(manifest['class_map']))\n",
        ),
        (
            "code",
            "if manifest:\n    exclusions = pd.DataFrame(manifest['exclusions'])\n    if not exclusions.empty:\n        display(exclusions.groupby(['split', 'reason']).size().rename('images').reset_index())\n    display(pd.DataFrame(list(manifest['train_class_boxes'].items()), columns=['species', 'training_boxes']))\n    show_figure(output / 'class_balance.png')\n",
        ),
        (
            "markdown",
            "## Revisión humana antes de entrenar\n\nV5 conserva las 89 decisiones de Miguel cuyos objetivos de revisión no cambiaron. Su confirmación global queda registrada en receipts/human-label-confirmation.json. El CSV humano contiene 90 fotografías aprobadas por Miguel, incluida 596bdb10-23d2-11e8-a6a3-ec086b02610b tras convertir su tercera caja desde el recuadro rojo. La aprobación original y el hash del CSV constan en la salida preservada v5, receipts/human-label-review-completion.json. V6 conserva las 90 revisiones auténticas mediante review_migration.json: las fotografías y cajas permanecen idénticas. La referencia y su hash quedan archivados en inputs/user-reference/ave.png y configs/wildlife_label_corrections_v6.json. La revisión anterior queda en parent_label_review.csv. Las filas correctas requieren aprobación y evaluador; las notas visuales son opcionales. La observación humana inicial se conserva en el CSV histórico de v3. Guardar el CSV exportado en wildlife/label_review.csv; al repetir el comando se actualiza la galería desde ese archivo. La decisión de usar YOLOv8s existente está registrada en receipts/architecture-review.json como instrucción directa del usuario. No se asignan juicios ni autoría de forma automática.\n",
        ),
        (
            "code",
            "if (output / 'label_review.csv').exists():\n    display(label_review_status(output / 'label_review.csv', require_evidence=settings.wildlife.get('require_label_evidence', False)))\n    display(HTML('<a href=\"/files/wildlife/label_review.html\" target=\"_blank\">Abrir galería completa de etiquetas en Jupyter</a>'))\n    show_figure(output / 'label_contact_01.png', width=1100)\nelse:\n    print('Galería pendiente: ejecutar review-labels.')\nfrom neural_exam.wildlife.review import architecture_review_status\ndisplay(architecture_review_status(settings.output_root / 'receipts/architecture-review.json'))\nshow_json(output / 'review_migration.json')\ndisplay(manifest.get('jpeg_materialization', {})) if manifest else None\nshow_json(settings.output_root / 'receipts/yolo-loader-verification.json')\n",
        ),
        (
            "markdown",
            "## Smoke aislado\n\nUna época, muestreo estratificado de train/cis_val, con al menos un caso por clase disponible y algunos fondos. No se predice sobre cis_test ni trans_test. Esta evidencia sólo valida la ejecución y no mide el rendimiento científico final.\n",
        ),
        (
            "code",
            "show_json(settings.output_root / 'smoke/execution_summary.json')\nsmoke_path = settings.output_root / 'smoke/wildlife/smoke_validation.json'\nif smoke_path.exists():\n    smoke = read_json(smoke_path)\n    display({'scientific_run': smoke['scientific_run'], 'test_partitions_used': smoke['test_partitions_used'], 'eligible_validation_objects': smoke['metrics']['eligible_objects']})\nelse:\n    print('Smoke pendiente.')\n",
        ),
        (
            "markdown",
            "## Entrenamiento completo\n\nActivar RUN_TRAINING únicamente tras la revisión de etiquetas. El comando documentado también ejecuta evaluación, monitoreo y mapas. No iniciar otro entrenamiento simultáneo. RESUME recupera last.pt o reutiliza best.pt cuando el entrenamiento ya terminó.\n",
        ),
        (
            "code",
            "if RUN_TRAINING:\n    from neural_exam.wildlife.pipeline import run\n    with execution_lock(settings.output_root):\n        display(run(settings, resume=RESUME, quick=False, manifest=manifest))\nelse:\n    print('La ejecución completa no se inicia desde este notebook; se muestran resultados guardados cuando existen.')\nshow_figure(output / 'training_curves.png')\n",
        ),
        (
            "markdown",
            "## Cámaras vistas y nuevas\n\nAP COCO a 101 puntos de recall, IoU 0.50:0.95, confianza 0.001, NMS 0.7 y máximo 300 detecciones. AP de una categoría sin referencia queda indefinida. Se conserva la brecha global y se agrega una comparación sobre categorías con soporte en ambos tests. Los intervalos percentiles del 95 % remuestrean secuencias completas con 200 réplicas, semilla 42 y checkpoint fijo. No miden variación entre semillas ni generalización a regiones nuevas. Su tiempo de cálculo sobre predicciones completas permanece por medir.\n",
        ),
        (
            "code",
            "if (output / 'experiments.csv').exists():\n    display(pd.read_csv(output / 'experiments.csv'))\nelse:\n    print('Evaluación científica pendiente.')\nfor split in ['cis_test', 'trans_test']:\n    path = output / split / 'metrics.json'\n    if path.exists():\n        metrics = read_json(path)\n        display({'split': split, 'map50': metrics['map50'], 'map50_95': metrics['map50_95'], 'protocol': metrics['protocol']})\n        display(pd.DataFrame(metrics['per_species']))\nshow_json(output / 'generalization.json')\nshow_json(output / 'uncertainty.json')\n",
        ),
        (
            "markdown",
            "## Tamaño relativo y oclusión\n\nTamaño: pequeño <1 %, mediano 1–5 %, grande >5 % del área de la foto, en ambos tests. Oclusión v2: cohorte completa de objetos elegibles de trans_test, al menos 30 por nivel y ningún objeto sin revisión. Baja >75 % visible; media 25–75 %; alta <25 %. Visibilidad incierta o truncamiento: ambiguous, con evidencia y evaluador. Área y truncamiento no sustituyen oclusión. Los otros niveles y ambiguos son regiones crowd ignoradas en la comparación por grupo. La cobertura parcial conserva estado pendiente. La galería usa las imágenes existentes y superpone la caja del objeto; no duplica miles de fotografías.\n",
        ),
        (
            "code",
            "path = output / 'robustness.json'\nif path.exists():\n    robustness = read_json(path)\n    for split, groups in robustness['size_by_split'].items():\n        display(pd.DataFrame([{'split': split, 'group': group, 'objects': score['eligible_objects'], 'map50': score['map50'], 'map50_95': score['map50_95']} for group, score in groups.items()]))\n    display({'occlusion_status': robustness['occlusion_status']})\n    display(pd.DataFrame([{'level': level, 'reviewed_objects': score['reviewed_objects'], 'meets_minimum': score['meets_minimum'], 'map50': score['map50'], 'map50_95': score['map50_95']} for level, score in robustness['occlusion'].items()]))\nelse:\n    print('Robustez pendiente de evaluación y revisión humana.')\n",
        ),
        (
            "markdown",
            "## Doce casos Grad-CAM\n\nSeis por test: detección correcta a IoU ≥0.5, omisión/clase incorrecta y falsos positivos en fondo, cuando estén disponibles. La selección contrasta soporte raro/frecuente y horarios diurno/nocturno, con alternativas deterministas si falta un tipo de caso. La hora de captura es una aproximación de iluminación que se inspecciona visualmente. Se explica una propuesta y clase antes de NMS en la capa 9. Los mapas no prueban memorización causal del fondo.\n",
        ),
        (
            "code",
            "if (output / 'cam_cases.json').exists():\n    display(pd.DataFrame(read_json(output / 'cam_cases.json'))[['split', 'image_id', 'camera_id', 'case', 'class', 'capture_period', 'training_support', 'score']])\nfor split in ['cis_test', 'trans_test']:\n    show_figure(output / f'cam_{split}_grid.png', width=1100)\n",
        ),
        (
            "markdown",
            "## Demostración de actividad ecológica\n\nUna cámara nueva elegida por ID, con todos sus fotogramas oficiales, incluidos los excluidos del análisis de cajas. Se requiere fecha de captura válida y una ráfaga completa de al menos dos fotos. Confianza 0.25. Las detecciones no son individuos; las fotografías por especie se cuentan una vez aunque existan varias cajas. Las fechas sin zona no se convierten a Lima.\n",
        ),
        (
            "code",
            "path = output / 'activity.json'\nif path.exists():\n    activity = read_json(path)\n    display({key: activity[key] for key in ['camera_id', 'sequence_id', 'cohort', 'timezone', 'confidence', 'interpretation']})\n    display(pd.DataFrame(activity['camera_days']))\n    display(pd.DataFrame(activity['species_days']))\n    display(pd.DataFrame(activity['sequence_frames']))\nelse:\n    print('Demostración con predicciones finales pendiente.')\n",
        ),
        (
            "markdown",
            "## Informe y límites\n\nLos activos se generan desde resultados completos con el mismo checkpoint. Antes del PDF final se requieren revisión humana de etiquetas y oclusión, los doce mapas, discusión sustentada en cifras, cinco autores reales y revisión editorial de 6–10 páginas. La evaluación de CCT20 no valida un despliegue en hábitats peruanos ni permite estimar población.\n",
        ),
        (
            "code",
            "if GENERATE_REPORT_ASSETS:\n    display(prepare_report_assets(settings, 'wildlife'))\nshow_json(settings.output_root / 'reports/wildlife/status.json')\n",
        ),
    ]
    cells = []
    for index, (kind, source) in enumerate(contents):
        item = {
            "cell_type": kind,
            "id": f"wildlife-{index:02d}",
            "metadata": {},
            "source": source.splitlines(keepends=True),
        }
        if kind == "code":
            item.update(execution_count=None, outputs=[])
        cells.append(item)
    document = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "execution_policy": "saved evidence; training disabled by default",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    create_notebook()
