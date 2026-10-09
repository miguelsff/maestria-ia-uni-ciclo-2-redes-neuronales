# Pregunta 3: ejecución local de fauna

> Guía de la preparación original. Para la ejecución actual con fondos adicionales y revisión completa de oclusión, utilizar [WILDLIFE_V2_RUN.md](WILDLIFE_V2_RUN.md). La salida y el snapshot de fuentes originales se conservan como evidencia histórica.

El entorno, la auditoría de CCT20 y la prueba pequeña están validados. El entrenamiento científico requiere primero revisar las etiquetas. Este flujo monta únicamente las imágenes y anotaciones de fauna; MRI e inundaciones conservan su estado previo.

## Continuar desde la revisión

Desde `parcial/`:

```powershell
$wildlifeCompose = @('-f', 'compose.wildlife.yaml', '--env-file', '.env.wildlife')
docker compose @wildlifeCompose run --rm runner review-labels
```

Abrir `E:/Kaggle/exam-output-wildlife/wildlife/label_review.html` en un navegador. Revisar las 90 fotos, ampliar las imágenes y verificar cajas, categoría y ausencia de animales en los fondos. Las hojas `label_contact_01.png` a `label_contact_08.png` permiten localizar los ejemplos. La muestra original queda registrada y no se puede acortar para omitir revisiones.

Registrar el evaluador y guardar `label_review.csv` desde la galería en esa misma carpeta. Todos los ejemplos deben estar `reviewed`. Una fila `needs_changes` mantiene detenido el entrenamiento y exige resolver la observación. La implementación no fabrica revisiones humanas.

```powershell
docker compose @wildlifeCompose run --rm runner run --task wildlife
```

El comando comprueba datos, entrena y evalúa con `best.pt`, seleccionado en cis_val depurado. Produce mAP50 y mAP50:95 global y por especie, tamaño por test, doce Grad-CAM, curvas y actividad de una cámara nueva con una secuencia completa. Los tests se reservan para la evaluación final.

## Rutas y aislamiento

| Activo | Ubicación del host |
|---|---|
| Imágenes originales, solo lectura | `E:/Kaggle/eccv_18_all_images_sm/eccv_18_all_images_sm` |
| Anotaciones oficiales | `E:/Kaggle/CCT20/eccv_18_annotation_files` |
| Resultados y activos descargados | `E:/Kaggle/exam-output-wildlife` |
| Configuración privada ignorada por Git | `.env.wildlife` |
| Ejemplo de configuración | `.env.wildlife.example` |

Los JPG del dataset derivado son enlaces simbólicos en volúmenes Linux persistentes. `wildlife/dataset` del host no contiene otra colección de fotografías: se monta dentro de Docker. Las galerías son imágenes reducidas de revisión. Los originales no se mueven ni reorganizan.

Runner tiene red deshabilitada, ocho CPU, 12 GB de RAM y 2 GB de memoria compartida. Downloads sólo escribe anotaciones y resultados de fauna. No monta MRI, su ZIP ni inundaciones. `CCT_IMAGES_DIR` y `CCT_ANNOTATIONS_DIR` separan las rutas; `CCT_DATA_DIR` conserva el comportamiento anterior como alternativa.

## Evidencia verificada

| Comprobación | Evidencia en resultados |
|---|---|
| CUDA, gradientes, NMS y AMP en RTX 5090 | `receipts/doctor_gpu.json`, `receipts/amp.json` |
| Dependencias instaladas y descargas con URL/tamaño/SHA-256 | `receipts/` |
| Decodificación de 57.864 originales y SHA-256 | `cache/wildlife-integrity.json`, `wildlife/manifest.json` |
| Particiones, exclusiones y clases | `wildlife/split_audit.csv`, `exclusions.csv`, `class_map.csv` |
| Época y optimizador recuperados tras interrupción | `resume-check/passed.json` |
| Una época de smoke, sin predicción en tests | `smoke/execution_summary.json`, `smoke/wildlife/smoke_validation.json` |
| Grad-CAM del checkpoint de smoke en cis_val | `smoke/wildlife/cam_diagnostic.json` |
| Notebook ejecutado con auditoría y estado real | `notebooks/03_wildlife.ipynb` |

La auditoría deja 11.399 fotos en train, 2.543 en cis_val, 13.073 en cis_test, 1.592 en trans_val y 18.647 en trans_test. Se retiran 270 fotos de cis_val por secuencias compartidas con train; después no hay solapamiento de secuencias entre particiones. Las 668 fotos car de train se excluyen. Train oficial no incluye empty: no se trasladan fondos de validación o prueba a entrenamiento.

Fox tiene cinco cajas y badger nueve en train. cis_val no tiene deer ni fox; trans_test no tiene deer. Las clases sin referencia conservan AP indefinida; la mAP indica cuántas categorías tienen soporte.

## Reanudar y revisar oclusión

```powershell
docker compose @wildlifeCompose run --rm runner run --task wildlife --resume
docker compose @wildlifeCompose run --rm runner review-occlusion
docker compose @wildlifeCompose run --rm runner review-occlusion --extend 120
```

La reanudación valida configuración, datos, versiones y pesos. Recupera `last.pt` si el entrenamiento se interrumpió; si terminó, reutiliza `best.pt`. La evaluación conserva lotes terminados. La verificación demuestra época y optimizador, sin afirmar equivalencia bit a bit.

La galería `occlusion_review.html` comienza con hasta 120 objetos de trans_test. La ampliación conserva evidencia existente. Marcar visibilidad corporal: baja >75 %, media 25–75 %, alta <25 %. Registrar evidencia y evaluador; dejar ambiguos o truncamientos sin nivel. Se requieren 30 objetos por nivel. Tamaño no sustituye oclusión. Otros niveles y objetos sin revisar son regiones ignoradas de tipo crowd al evaluar un grupo.

Después del CSV, `run --task wildlife --resume` regenera métricas y activos. La demostración usa todos los fotogramas oficiales de una cámara nueva elegida por ID, fechas válidas y una ráfaga completa. Cuenta detecciones y fotografías, no individuos únicos.

Los nombres son categorías del benchmark; bird y rodent agrupan taxones. La diversidad de la demostración cuenta categorías detectadas, sin afirmar identificación exhaustiva de especies biológicas.

## Notebook e informe

El notebook fuente tiene entrenamiento desactivado. La copia ejecutada muestra las constancias. Después del entrenamiento completo se vuelve a ejecutar para incorporar resultados científicos:

```powershell
docker compose @wildlifeCompose run --rm --entrypoint jupyter runner nbconvert --execute --to notebook notebooks/03_wildlife.ipynb --output 03_wildlife --output-dir /outputs/notebooks --ExecutePreprocessor.timeout=180
docker compose @wildlifeCompose --profile notebooks up -d notebooks
```

Abrir `http://localhost:8890`, usar el token privado de `.env.wildlife` y abrir `notebooks/03_wildlife.ipynb`. Se copia únicamente el notebook de fauna y se muestran sus carpetas de resultados. Evitar entrenamiento simultáneo en notebook y consola.

```powershell
docker compose @wildlifeCompose run --rm runner report --task wildlife
docker compose @wildlifeCompose run --rm reports --task wildlife --draft
```

El informe incluye ambas mAP globales, AP y soporte por especie, tamaños en ambos tests, oclusión, actividad, curvas y los doce mapas. Sólo se genera desde evaluaciones completas del mismo checkpoint. El PDF final requiere revisiones de etiquetas y oclusión, doce CAM, cinco autores reales, discusión sustentada en resultados, revisión editorial y 6–10 páginas. Autores y discusión empírica siguen pendientes. Registrar `{"reviewed": true}` en `reports/wildlife/finalized.json` únicamente después de esa revisión y compilar sin `--draft`.

## Repetir verificaciones

```powershell
docker compose @wildlifeCompose build runner reports
docker compose @wildlifeCompose run --rm runner doctor --gpu-check --amp-check
docker compose @wildlifeCompose run --rm --entrypoint python runner -m pytest -q tests/test_wildlife_data.py tests/test_wildlife_workflow.py tests/test_utilities.py -p no:cacheprovider
docker compose @wildlifeCompose run --rm runner verify-wildlife-resume
docker compose @wildlifeCompose run --rm runner smoke --task wildlife --resume
```

Smoke usa 56 fotos de train y 31 de cis_val, una época, batch dos y cero workers. Comparte sólo pesos locales y caché de integridad; no aporta cifras al informe científico.

Las descargas acotadas son `download --dataset wildlife-weights` y `download --dataset cct-annotations`, usando downloads. No se ejecuta la descarga general de todos los pesos o imágenes. Los SHA-256 son verificaciones locales; no se presentan como MD5 publicados por el proveedor.

Para otra configuración o dataset, usar otra carpeta de salida y otro nombre de proyecto Compose mediante `--project-name`, obteniendo volúmenes derivados independientes. Preservar resultados y no ejecutar `down --volumes` sobre una ejecución que se quiera conservar.
