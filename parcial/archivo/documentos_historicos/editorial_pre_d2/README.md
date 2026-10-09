# Redes Neuronales y Aprendizaje Profundo — UNI

Repositorio de materiales y proyectos del segundo ciclo de la Maestría en Inteligencia Artificial de la Universidad Nacional de Ingeniería.

El examen parcial reúne tres aplicaciones de visión por computadora: clasificación multietiqueta de MRI de rodilla, segmentación de agua con radar Sentinel-1 y detección de fauna con cámaras trampa. La implementación se organiza como un paquete Python compartido, con ejecución local y experimentos secuenciales.

**Pregunta 1:** ocho experimentos MRI completos sobre 600 estudios y 464 predicciones auditadas. El informe incorpora los cinco autores, métricas reales, ablaciones, ruido textual, sesgo y tres mapas Grad-CAM. La [entrega MRI](parcial/MRI_DELIVERY.md) identifica el PDF, las verificaciones y los pendientes del grupo; la [guía de ejecución](parcial/MRI_RUN.md) conserva la historia experimental.

Para reproducir MRI desde `parcial/`, con Python 3.12, Docker Desktop/WSL2, GPU NVIDIA compatible con CUDA 12.8 y los originales RSNA descargados:

```powershell
.\scripts\run_mri.ps1
```

`RSNA_HOST_DIR` señala la carpeta con `train.csv`, `train_series.csv` y `train_series/` (por defecto `E:/Kaggle/rsna-knee-abnormality-detection`). El comando lee esos metadatos directamente del montaje de originales; no requiere `parcial/data/`. Para repetir las ocho corridas, elegir una carpeta nueva mediante `MRI_OUTPUT_DIR`. Los resultados científicos existentes se conservan.

**Pregunta 2:** seis experimentos científicos terminados, 618 predicciones georreferenciadas auditadas y notebook ejecutado sin errores. La autoría del informe incorpora los cinco integrantes y la afiliación proporcionados por el grupo; quedan pendientes su aprobación editorial y la entrega por la plataforma. La [guía de inundaciones](parcial/FLOOD_RUN.md) contiene resultados, evidencia y PDF de revisión; el [paquete reproducible en GitHub](https://github.com/miguelsff/maestria-ia-uni-ciclo-2-redes-neuronales/tree/codex/flood-delivery) está publicado y verificado.

Para reproducir la pregunta 2, desde `parcial/` y con los datos completos y Docker/WSL2/NVIDIA configurados:

```powershell
.\scripts\run_flood.ps1
```

El comando genera las seis corridas, evaluaciones, notebook y PDF borrador. Para repetir la ejecución científica, configurar una salida nueva mediante `FLOOD_OUTPUT_DIR`; la [guía local](parcial/README.md) documenta las rutas y los prerrequisitos. Los resultados ya auditados se conservan para el cierre editorial.

**Pregunta 3:** la preparación v2 conserva CCT20 y añade 1.000 fondos reales de ocho cámaras oficiales adicionales de entrenamiento. Los controles CPU/GPU, la prueba pequeña y el notebook pasaron; el entrenamiento científico sigue pendiente de revisar 90 fotografías. Continuar con la [guía v2 de fauna](parcial/WILDLIFE_V2_RUN.md) y su comando de ejecución; la salida original permanece conservada.

## Organización

| Ruta | Contenido |
|---|---|
| [docs/](docs/) | Material del curso: CNN, backpropagation, ResNet, transfer learning y YOLO. |
| [parcial/examen-parcial.pdf](parcial/examen-parcial.pdf) | Enunciado y rúbrica del examen. |
| [parcial/ejemplo.pdf](parcial/ejemplo.pdf) | Referencia de estructura de un informe; sus resultados no se trasladan a estos proyectos. |
| [parcial/src/neural_exam/](parcial/src/neural_exam/) | Módulos MRI, inundaciones y fauna, más utilidades de entrenamiento y evaluación. |
| [parcial/notebooks/](parcial/notebooks/) | Notebooks de MRI e inundaciones ejecutados; la copia de fauna ejecutada con auditoría queda fuera de Git, en resultados. |
| [parcial/reports/](parcial/reports/) | Fuentes NeurIPS en español; MRI e inundaciones incluyen resultados reales y limitaciones. |
| [parcial/tests/](parcial/tests/) | Pruebas; las de fauna y utilidades ya se ejecutaron, incluida reanudación YOLO en GPU. |
| [parcial/WILDLIFE_RUN.md](parcial/WILDLIFE_RUN.md) | Estado real y continuación de la pregunta 3. |
| [parcial/README.md](parcial/README.md) | Prerrequisitos, descargas, ubicación de datos y pasos de ejecución local. |
| [parcial/PREPARATION_STATUS.md](parcial/PREPARATION_STATUS.md) | Alcance implementado y verificaciones pendientes. |

## Proyectos y entregables

| Pregunta | Diseño preparado | Entregables tras ejecutar |
|---|---|---|
| MRI | Reglas multilingües, CNN 2.5D propia y ResNet-50, BCE/focal, estrategias de confianza. | AUC por etiqueta y macro, precisión/recall/F1, acuerdo textual, prevalencia y Grad-CAM. |
| Inundaciones | U-Net propia con VV/VH, máscaras sin datos y seis ablaciones. | IoU promedio por chip, métricas globales, evaluación en Bolivia, matrices y revisión de fallos. |
| Fauna | YOLOv8s, benchmark CCT20 oficial, vacíos como fondo y particiones depuradas. | AP por especie, cámaras vistas/nuevas, tamaño/oclusión, Grad-CAM y actividad por cámara, especie y día. |

Los parámetros base son semilla 42, hasta 50 épocas y parada temprana con paciencia 10. Docker Compose está preparado para RTX 5090 con PyTorch 2.10.0, torchvision 0.25.0 y CUDA 12.8, limitado a 8 CPU, 12 GB de RAM y 2 GB de memoria compartida.

La [guía local del parcial](parcial/README.md) documenta el flujo de los tres proyectos. MRI e inundaciones tienen entregables locales verificados; la plantilla común ya contiene la autoría real y los PDF requieren las revisiones indicadas en sus guías. Fauna mantiene las etapas pendientes indicadas en su guía.

Los datasets, reportes clínicos, pesos y predicciones pesadas se mantienen fuera de Git. La entrega de revisión de inundaciones incluye evidencia ligera y PDF; no hay envíos automáticos a Kaggle.
