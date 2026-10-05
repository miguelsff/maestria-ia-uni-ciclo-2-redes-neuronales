# Ejecutar la pregunta 2

El [README principal](../README.md) documenta los datos, el entorno y los resultados publicados. Desde esta carpeta, con Sen1Floods11 v1.1 completo, Docker Desktop/WSL2 y NVIDIA configurados:

```powershell
.\scripts\run_flood.ps1
```

Es el comando completo: construcción, caché verificada, controles, smoke, seis corridas científicas, evaluación, auditoría, notebook y PDF borrador. `requirements.txt` fija el entorno Python y `Dockerfile` la base CUDA 12.8.1.

Para una reproducción independiente:

```powershell
$env:SEN1_HOST_DIR = 'E:/Kaggle/Sen1Floods11'
$env:FLOOD_OUTPUT_DIR = 'E:/Kaggle/exam-output-flood-reproduction'
.\scripts\run_flood.ps1
```

Los resultados científicos originales ya terminaron y están acompañados por [evidencia ligera](delivery/flood/evidence/). El [notebook](notebooks/02_flood.ipynb) está ejecutado y mantiene desactivado el entrenamiento automático. El [PDF de revisión](delivery/flood/flood_review.pdf) incluye los cinco integrantes y siete referencias; la aprobación editorial del grupo está pendiente.

No mezclar una nueva configuración con una salida existente. `-Resume` recupera estado compatible y puede volver a escribir evaluaciones; conservar los originales y las constancias anteriores. [FLOOD_RUN.md](FLOOD_RUN.md) detalla particiones, límites y mecanismos de reanudación.

Después de una aprobación editorial real, cambiar `reports/flood/finalized.json` para registrar `reviewed=true` y compilar el informe final sobre los activos existentes, sin entrenamiento:

```powershell
docker compose -f compose.flood.yaml run --rm reports --task flood
```

El compilador controla las 6–10 páginas. Antes de entregar, revisar todas las páginas y subir el PDF y el enlace de esta rama por la plataforma del curso; presentar el 30 de octubre de 2026.
