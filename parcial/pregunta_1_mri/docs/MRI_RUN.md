# Pregunta 1: ejecución local de MRI

Estado actual: ocho experimentos científicos completos y conservados. La entrega técnica tiene PDF de nueve páginas con cinco autores y notebook ejecutado sin errores; ver [MRI_DELIVERY.md](MRI_DELIVERY.md) para publicación y pendientes del grupo. La ejecución científica utiliza la partición weak-multilabel-quota-v2.

Esta guía conserva el registro de ejecución y sus comprobantes originales. Las menciones posteriores al borrador de ocho páginas y a pendientes de autoría describen esa versión histórica; la entrega actual está identificada por separado en MRI_DELIVERY.md.

## Resultados observados

Evaluación de los mismos 58 estudios expertos, doce AUC definidos y umbral fijo 0,5:

| Arquitectura | Pérdida | Supervisión | Épocas / mejor | AUC macro 12 | F1 macro |
|---|---|---|---|---|---|
| CNN propia | BCE | Ponderada | 48 / 38 | 0,5776 | 0,3610 |
| CNN propia | BCE | Alta confianza | 23 / 13 | 0,5130 | 0,4535 |
| CNN propia | Focal | Ponderada | 29 / 19 | 0,5723 | 0,4961 |
| CNN propia | Focal | Alta confianza | 23 / 13 | 0,5204 | 0,4253 |
| ResNet-50 | BCE | Ponderada | 12 / 2 | 0,6189 | 0,3614 |
| ResNet-50 | BCE | Alta confianza | 11 / 1 | 0,6020 | 0,4513 |
| ResNet-50 | Focal | Ponderada | 12 / 2 | 0,6199 | 0,5056 |
| ResNet-50 | Focal | Alta confianza | 12 / 2 | 0,5877 | 0,4863 |

ResNet/focal/ponderada tiene el mayor AUC observado; esta comparación no elige un modelo para despliegue ni establece superioridad con una sola semilla. Los checkpoints se eligieron exclusivamente por pérdida de validación débil. El baseline cualitativo fijado previamente es CNN/BCE/ponderada.

La mitigación de desbalance tiene costos: en ResNet ponderada, focal eleva recall de Medial OA de 0,133 a 0,867 y F1 de 0,222 a 0,473, pero reduce precisión de 0,667 a 0,325. Los tres Grad-CAM incluyen respuestas difusas, periféricas o de fondo; no demuestran localización anatómica fiable. El informe documenta ambos límites.

La primera ejecución dejó todos los positivos débiles de Lateral OA en validación (0 train / 14 val). Se conserva íntegra en E:/Kaggle/exam-output-mri, marcada como reemplazada para entrega en audit/protocol_status.json. La corrección utilizó únicamente etiquetas débiles, conservó los 600 estudios y repitió las ocho variantes en una salida distinta. Los mismos expertos se consultaron en ambas ejecuciones: la evaluación final no es una prueba externa nueva nunca observada. No se cambiaron reglas, umbral ni hiperparámetros por los resultados expertos.

## Evidencia disponible

- 4.407 estudios y 24.371 descriptores de serie corresponden al inventario local.
- Selección fija de 600 estudios: 434 entrenamiento, 108 validación débil y 58 evaluación experta. No se comparten estudios ni los PatientID disponibles entre particiones; los identificadores anonimizados no garantizan independencia de personas reales.
- 22.954 DICOM seleccionados copiados a una caché local y comparados con hashes SHA-256 de origen. Los originales siguen en E:/Kaggle/rsna-knee-abnormality-detection y se montan en solo lectura.
- Python 3.12.3, PyTorch 2.10.0+cu128, torchvision 0.25.0+cu128 y scikit-learn 1.9.1. CUDA reconoce la RTX 5090 y sm_120; gradientes y prueba de torchvision aprobados.
- Batch completo 4×8×3×224×224 validado con ambas arquitecturas y BCE/focal. Máximo asignado en esta comprobación: aproximadamente 1,55 GiB de VRAM; no es consumo total ni resultado científico.
- 32 pruebas aprobadas, incluidas pérdidas, separación, normalización DICOM, tipos de objetivos, Grad-CAM y reanudación exacta del fixture CPU.
- 464 predicciones guardadas verificadas: métricas recalculadas con scikit-learn, checkpoints de mínima pérdida débil, hashes del código congelado y 600 cachés float32 finitas. Véase audit/mri_results.json.
- Ocho celdas de código del notebook ejecutadas sin errores; controles de preparación, entrenamiento y generación de activos desactivados.
- PDF NeurIPS de ocho páginas inspeccionadas, con siete referencias y sin referencias indefinidas ni cajas desbordadas. Autoría y revisiones profesionales permanecen pendientes.

Las correcciones y su momento están documentados en [MRI_PLAN.md](MRI_PLAN.md): alpha después del filtro, salida float32, objetivos débiles flotantes y partición corregida después de la primera matriz. El extractor textual no se modifica después de evaluar expertos.

## Ejecutar o reanudar

Desde parcial, con Docker Desktop, GPU NVIDIA y los originales descargados:

```powershell
./scripts/run_mri.ps1
```

El comando audita, prepara, ejecuta smoke y las ocho variantes secuenciales, verifica resultados y genera notebook e informe borrador. Para reanudar checkpoints compatibles:

```powershell
./scripts/run_mri.ps1 -Resume
```

Si existe un contenedor detenido de una ejecución anterior, inspeccionar primero su salida y código de retorno. Tras conservar los logs, remover únicamente ese contenedor antes de invocar el script de reanudación; los volúmenes y checkpoints no se eliminan:

```powershell
docker logs neural-exam-mri-training
docker inspect neural-exam-mri-training --format '{{.State.Status}} {{.State.ExitCode}} {{.State.OOMKilled}}'
docker rm neural-exam-mri-training
```

Para detener únicamente el entrenamiento MRI:

```powershell
docker stop --time 30 neural-exam-mri-training
```

La reanudación recupera la última época completamente guardada. No ejecutar junto con otros entrenamientos de GPU; el script espera antes de comenzar y detiene únicamente MRI si aparece otro runner de entrenamiento.

## Datos, caché y resultados

| Ubicación | Contenido |
|---|---|
| E:/Kaggle/exam-output-mri/stratified-v2/audit/ | Inventario, hashes originales, datos auditados, logs, entorno, código y configuración congelados. |
| Volumen neural-exam-mri_dicom_cache | Copia byte a byte de las 600 series seleccionadas, montada en solo lectura. |
| Volumen neural-exam-mri_tensor_cache | Tensores normalizados usados para evitar lectura lenta de archivos pequeños desde Windows. |
| E:/Kaggle/exam-output-mri/stratified-v2/mri/cache/ | Copia exportada de los 600 NPZ, verificada byte a byte contra el volumen mediante audit/exported_tensor_cache.json. |
| E:/Kaggle/exam-output-mri/stratified-v2/mri/ | Manifiesto, ocho directorios de experimentos, métricas, predicciones, curvas y checkpoints best.pt/last.pt. |
| E:/Kaggle/exam-output-mri/stratified-v2/notebooks/01_mri.ipynb | Notebook ejecutado con los flags de entrenamiento desactivados. |
| E:/Kaggle/exam-output-mri/stratified-v2/reports/mri_draft.pdf | Borrador NeurIPS; el estado editorial se registra por separado. |

experiments.csv resume la matriz; metrics_all_labels.csv conserva AUC, precisión, recall y F1 de las doce etiquetas en las ocho variantes. class_balance.csv registra positivos y supervisión efectiva. Cada experimento conserva sus predicciones, historial, configuración, execution.json y checkpoints. audit/delivery.json registra hashes de los entregables y verificación final. La copia ejecutada del notebook también está en parcial/notebooks/01_mri.ipynb.

Se utilizan exclusivamente configs/mri.toml y compose.mri.yaml. Los límites del runner son 8 CPU, 12 GiB RAM y 2 GiB shm. No se modifican configuraciones ni resultados de inundaciones/fauna. PRETRAINED_WEIGHTS_DIR apunta a /outputs/cache/weights; los únicos activos descargados son ResNet-50 ImageNet V2 y el kit oficial NeurIPS 2026, con SHA-256 en receipts/mri_assets.json.

La primera copia de caché requiere Python con biblioteca estándar en Windows; MRI_PYTHON permite indicar su ejecutable. RSNA_HOST_DIR y MRI_OUTPUT_DIR permiten cambiar ubicaciones antes de comenzar. No cambiar configuración, código de entrenamiento, particiones o rutas de caché para reanudar un resultado existente.

## Entrega editorial pendiente

Se requieren los cinco nombres y correos reales, revisión anatómica de los mapas y revisión editorial del grupo. No se inventan autores ni revisiones humanas. El repositorio existente es https://github.com/miguelsff/maestria-ia-uni-ciclo-2-redes-neuronales; los cambios locales no se publican automáticamente.
