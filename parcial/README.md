# Examen parcial: preparación y ejecución local

La pregunta 1 tiene ocho experimentos MRI completos y auditoría aprobada de 464 predicciones. El informe incorpora los cinco autores y revisión técnica/visual. [MRI_DELIVERY.md](MRI_DELIVERY.md) identifica la entrega y sus verificaciones; [MRI_RUN.md](MRI_RUN.md) conserva resultados e historia de la partición. Quedan pendientes la aprobación editorial del grupo, sus obligaciones personales de Kaggle y la entrega por la plataforma. No se registra validación clínica.

## Reproducción de la pregunta 1 con un comando

Desde `parcial/`, con Python 3.12 disponible como `python` (o indicado en `MRI_PYTHON`), Docker Desktop, WSL2 y una GPU NVIDIA compatible con CUDA 12.8:

```powershell
.\scripts\run_mri.ps1
```

Descargar previamente los datos originales de la competición RSNA con una cuenta autorizada. `RSNA_HOST_DIR` debe contener `train.csv`, `train_series.csv` y `train_series/`; el valor predeterminado es `E:/Kaggle/rsna-knee-abnormality-detection`. Los CSV se leen directamente de los originales montados en solo lectura; no es necesario copiar archivos a `parcial/data/`.

El comando construye las imágenes si faltan, selecciona y verifica las series, copia una caché aislada, prepara tensores, descarga los pesos y el estilo, verifica CUDA, ejecuta pruebas y una corrida pequeña, entrena y evalúa ocho variantes, audita sus resultados y genera tablas, figuras, notebook ejecutado y PDF borrador. Para una reproducción independiente:

```powershell
$env:RSNA_HOST_DIR = 'E:/Kaggle/rsna-knee-abnormality-detection'
$env:MRI_OUTPUT_DIR = 'E:/Kaggle/exam-output-mri-reproduction'
.\scripts\run_mri.ps1
```

La carpeta de salida debe ser nueva. `-Resume` recupera una ejecución interrumpida con la misma firma de código, configuración y datos. La entrega científica congelada mantiene su código histórico; el ajuste posterior del estado `run.json` se verifica en un caso sintético y no modifica esas ocho corridas. No ejecutar dos entrenamientos que compartan GPU o caché.

