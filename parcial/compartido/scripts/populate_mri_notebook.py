"""Update only the existing MRI notebook for reproducible persisted-result review."""

import json
from pathlib import Path


def main() -> None:
    """Preserve notebook cells while replacing preparation placeholders with real-output views."""
    path = Path(__file__).resolve().parents[1] / "notebooks/01_mri.ipynb"
    document = json.loads(path.read_text(encoding="utf-8"))
    replacements = {
        0: "# Pregunta 1: MRI de rodilla\n\nComparación factorial de CNN propia/ResNet-50, BCE/focal y dos estrategias de confianza. "
        "Este cuaderno consulta exclusivamente las salidas persistidas y auditadas. Los controles de entrenamiento permanecen desactivados.\n",
        2: """from pathlib import Path
import os
import pandas as pd
from IPython.display import display, Image
from neural_exam.config import load_settings, PROJECT_ROOT
from neural_exam.cli import doctor
from neural_exam.common.io import read_json
from neural_exam.common.locking import execution_lock
from neural_exam.mri.data import prepare
from neural_exam.mri.pipeline import run
from neural_exam.reporting import prepare_report_assets

RUN_PREPARE = False
RUN_TRAINING = False
RESUME = False
GENERATE_REPORT_ASSETS = False
os.environ.setdefault('OUTPUT_DIR', '/outputs' if Path('/outputs').is_dir() else 'E:/Kaggle/exam-output-mri/stratified-v2')
settings = load_settings(PROJECT_ROOT / 'configs/mri.toml')
task = 'mri'
output = settings.output_root / task
""",
        4: """diagnostics = doctor(settings, gpu_check=False)
display(pd.DataFrame(list(diagnostics['dependencies'].items()), columns=['dependency', 'version']))
runtime = read_json(settings.output_root / 'audit/mri_runtime.json')
display(pd.DataFrame(runtime['checks']))
""",
        5: "## Integridad y preparación\n\nLos mismos 58 expertos se evaluaron en la ejecución inicial y en la partición corregida. "
        "La corrección empleó exclusivamente etiquetas débiles; esta cohorte no es una nueva prueba externa nunca consultada. "
        "El léxico, el umbral y los hiperparámetros permanecieron fijados. La auditoría compara bytes de la caché con originales, "
        "valida identificadores y evita compartir estudios entre particiones. Los identificadores anonimizados no prueban independencia de pacientes reales.\n",
        6: """if RUN_PREPARE:
    with execution_lock(settings.output_root):
        manifest = prepare(settings)
else:
    manifest = read_json(output / 'manifest.json')
audit = read_json(settings.output_root / 'audit/mri_data.json')
display({key: audit[key] for key in ['all_metadata_studies', 'all_metadata_series', 'selected_dicom_files',
                                    'docker_cache_byte_matches', 'split_counts', 'patient_split_overlaps']})
""",
        8: """display(pd.DataFrame(list(manifest['split_counts'].items()), columns=['partition', 'studies']))
display(pd.read_csv(output / 'class_balance.csv'))
display(pd.DataFrame({'script': list(audit['weak_reports_by_script']),
                      'weak_reports': list(audit['weak_reports_by_script'].values()),
                      'supported': [audit['weak_supported_by_script'].get(key, 0) for key in audit['weak_reports_by_script']]}))
""",
        10: """if RUN_TRAINING:
    with execution_lock(settings.output_root):
        display(run(settings, resume=RESUME, quick=False, manifest=manifest))
else:
    print('RUN_TRAINING=False: se leen resultados existentes, sin repetir entrenamientos.')
""",
        12: """result_audit = read_json(settings.output_root / 'audit/mri_results.json')
assert result_audit['status'] == 'passed'
display(pd.DataFrame(result_audit['experiments']).drop(columns=['checkpoint_sha256']))
scores = pd.read_csv(output / 'metrics_all_labels.csv')
display(scores.pivot(index='label', columns='experiment', values='auc'))
display(scores[scores['experiment'] == 'scratch_bce_weighted'])
agreement = pd.DataFrame(read_json(output / 'label_agreement.json')['per_label'])
agreement['coverage_of_58'] = agreement['observed'] / 58
display(agreement)
""",
        13: """figure_root = settings.output_root / 'reports/mri/figures'
for filename in ['auc_matrix.png', 'curves_scratch.png', 'curves_resnet50.png', 'cam_grid.png']:
    display(Image(filename=str(figure_root / filename), width=950))
display(pd.DataFrame(read_json(output / 'scratch_bce_weighted/cam_cases.json')))
""",
        14: "## Informe y trazabilidad\n\nLas tablas, figuras y PDF se generan desde métricas reales y auditadas. "
        "La configuración, hashes de código, pesos y checkpoints se conservan junto a los resultados. "
        "La entrega incorpora los cinco autores y revisión técnica/visual. La aprobación editorial del grupo y la entrega por la plataforma "
        "siguen pendientes; no se registra validación clínica profesional. Las métricas se recalculan desde predicciones persistidas, sin nueva inferencia.\n",
        16: "## Limitaciones y conclusiones\n\nUna única semilla y 58 expertos limitan la precisión de la comparación. "
        "Si una etiqueta tiene solo una clase, su AUC y el macro oficial de doce etiquetas quedan indefinidos; "
        "el macro parcial se identifica por separado. Precisión, recall y F1 usan zero_division=0. "
        "El filtro de confianza se aplica por etiqueta y conserva 434 estudios de entrenamiento y 108 de validación. "
        "En sinovitis deja 140 etiquetas de entrenamiento, todas positivas, sin negativos explícitos. "
        "La cobertura de escritura no equivale a detección de idioma, procedencia o validación clínica. "
        "Los Grad-CAM son observaciones espaciales y no acreditan causalidad ni localización radiológica correcta.\n",
    }
    for index, source in replacements.items():
        document["cells"][index]["source"] = source.splitlines(keepends=True)
    for cell in document["cells"]:
        if cell["cell_type"] == "code":
            cell.update(execution_count=None, outputs=[])
    document["metadata"]["preparation_status"] = "ready_for_persisted_result_execution"
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
