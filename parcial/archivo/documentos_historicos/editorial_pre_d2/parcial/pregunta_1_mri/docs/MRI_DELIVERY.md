# Entrega técnica de MRI — 4 de octubre de 2026

El PDF de pregunta 1 tiene nueve páginas, cinco autores y siete referencias. Se compiló sin `--draft` y se inspeccionaron visualmente sus nueve páginas. La revisión registrada es técnica y de diagramación; la aprobación editorial del grupo sigue pendiente.

## Archivos de entrega

| Archivo | Ubicación local |
|---|---|
| PDF preparado para aprobación del grupo | `E:/Kaggle/exam-output-mri/submission-2026-10-04/reports/mri.pdf` |
| Copia con el nombre previsto en el plan | `E:/Kaggle/exam-output-mri/stratified-v2/reports/mri.pdf` |
| Notebook ejecutado | `E:/Kaggle/exam-output-mri/submission-2026-10-04/notebooks/01_mri.ipynb` |
| Constancia y hashes de esta entrega | `E:/Kaggle/exam-output-mri/submission-2026-10-04/audit/final_delivery.json` |
| Guion para la exposición | [MRI_DEFENSE.md](MRI_DEFENSE.md) |

Código de entrega: [rama codex/mri-delivery](https://github.com/miguelsff/maestria-ia-uni-ciclo-2-redes-neuronales/tree/codex/mri-delivery), publicada y comprobada desde un clon de GitHub. El código del commit [45efe9d](https://github.com/miguelsff/maestria-ia-uni-ciclo-2-redes-neuronales/tree/45efe9d905f4a5a034334de18be3093fcd7d9146) pasó las 32 pruebas CPU y reprodujo exactamente la selección de los 600 estudios. [MRI_VERIFICATION.json](MRI_VERIFICATION.json) registra esa comprobación. El repositorio es privado; los integrantes deben comprobar que el docente pueda acceder al enlace.

El notebook publicado conserva las celdas y tiene las salidas vacías para no incluir imágenes clínicas en GitHub. El notebook local conserva las ocho celdas ejecutadas, las figuras y cero errores. Las fuentes LaTeX se publican; las figuras se generan localmente desde las salidas científicas. El PDF, los DICOM, los textos clínicos, las predicciones individuales y los pesos permanecen fuera de Git.

## Cambios y evidencia

| Unidad de trabajo | Resultado | Verificación |
|---|---|---|
| Reproducción | Compose lee CSV desde `/datasets/rsna`; README incluye comando y prerrequisitos. | Paquete sin `parcial/data/`: selección idéntica de 600 estudios y particiones; configuración Compose comprobada. |
| Estado terminal | Futuras corridas cierran `run.json` con `status=complete`, épocas y mejor pérdida. | 32 pruebas aprobadas en CPU, incluida interrupción/reanudación e igualdad exacta de parámetros con corrida continua. |
| Informe y notebook | Autoría, porcentajes con denominador 434, filtro por etiqueta/sinovitis, expertos reutilizados, métricas desde predicciones y conclusiones limitadas. | Ocho celdas ejecutadas sin errores, entrenamiento desactivado; cifras científicas conservadas. |
| Diagramación | Tablas y figuras no interrumpen párrafos; gráficos legibles y referencias enlazadas. | Nueve páginas inspeccionadas; cero `Overfull`, cero referencias indefinidas; un `Underfull` sin recorte visible. |
| Conservación | Nueva salida de entrega y versiones históricas preservadas. | 131 archivos científicos inventariados mantienen sus SHA-256; no se reentrenó la matriz. |

La prueba de selección verifica la corrección del empaquetado. El runner rechaza una salida con checkpoints antes de escribir logs o auditorías cuando no se solicita reanudación; se comprobó esa protección contra la salida histórica. No se repitió el entrenamiento de las ocho variantes desde el clon. Una reproducción científica nueva puede variar entre equipos y versiones; la prueba sintética no garantiza identidad de todos los resultados reales.

## Reproducción

Desde un clon de la rama de entrega, entrar en `parcial/` y ejecutar:

```powershell
$env:RSNA_HOST_DIR = 'E:/Kaggle/rsna-knee-abnormality-detection'
$env:MRI_OUTPUT_DIR = 'E:/Kaggle/exam-output-mri-reproduction'
.\scripts\run_mri.ps1
```

Se requieren Python 3.12, Docker Desktop/WSL2, GPU NVIDIA compatible con CUDA 12.8, acceso a descargas de pesos/estilo y los originales RSNA autorizados. La carpeta de originales contiene `train.csv`, `train_series.csv` y `train_series/`. La salida debe ser nueva. Consultar [README.md](README.md) para el alcance del comando y `MRI_PYTHON` si el intérprete tiene otro nombre.

Para regenerar únicamente el informe desde una salida científica auditada, sin entrenar:

```powershell
$env:MRI_OUTPUT_DIR = 'E:/Kaggle/exam-output-mri/submission-2026-10-04'
docker compose -f compose.mri.yaml run --rm --entrypoint python runner scripts/prepare_mri_report.py
docker compose -f compose.mri.yaml run --rm reports --task mri
```

El registro `reports/mri/finalized.json` identifica la revisión técnica y mantiene pendiente la aprobación del grupo. No certifica revisión radiológica ni aceptación de reglas Kaggle.

## Trazabilidad y reversión

El experimento del PDF se ejecutó con el código congelado en `E:/Kaggle/exam-output-mri/stratified-v2/audit/source_snapshot`. El cambio posterior en `common/training.py` solo cierra metadatos para futuras corridas; no modifica el algoritmo de optimización ni los checkpoints existentes. Sus hashes actuales difieren deliberadamente de los registrados en las corridas históricas. Para verificar esas corridas contra su código, usar la instantánea histórica.

La constancia anterior `stratified-v2/audit/delivery.json` permanece intacta. Sus referencias al notebook y la guía apuntan a versiones que ahora han evolucionado; las versiones originales correspondientes están conservadas en `audit/delivery_source_snapshot`. El PDF borrador, las métricas, los datos auditados y la configuración mantienen sus hashes históricos. La nueva constancia registra los archivos actuales.

Para revertir la edición del informe, usar la instantánea editorial histórica y conservar la nueva salida como evidencia. Para revertir el empaquetado o estado terminal, revertir el commit de entrega en una rama de trabajo. No borrar ni reescribir corridas, cachés o comprobantes científicos para esa reversión.

## Pendientes del grupo

El grupo debe aprobar el contenido final, incluidas las limitaciones de los mapas. Cada integrante debe confirmar su cuenta/acceso y aceptación de reglas Kaggle; el grupo debe revisar las condiciones de uso externo. El repositorio privado debe ser accesible al docente. Falta el comprobante de entrega del PDF y el enlace GitHub en la plataforma, así como la exposición del 30 de octubre de 2026. El examen no exige correo de los autores, leaderboard ni firma de un radiólogo.
