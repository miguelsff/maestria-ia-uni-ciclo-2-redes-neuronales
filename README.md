# Pregunta 2: segmentación de agua con Sentinel-1

UNI · Unidad de Posgrado — Facultad de Ingeniería Industrial y de Sistemas.

Esta rama publica el material reproducible de la pregunta 2 del examen parcial: U-Net propia PyTorch con VV/VH, seis experimentos manual/manual+débil × BCE/Dice/combinada, test manual oficial y Bolivia reservada. Las corridas científicas están completas; la revisión local comprobó 618 GeoTIFF, integridad de datos y selección de checkpoints por validación.

**Informe con autoría completa:** [PDF de revisión](parcial/delivery/flood/flood_review.pdf), nueve páginas y siete referencias. La aprobación editorial real del grupo y la entrega por la plataforma están pendientes. El estado no certifica una revisión humana que todavía no ocurrió.

Integrantes: Chilón Cruzado Josué; Concepción Tiza Miguel Ángel; Neyra Dianderas Enzo Israel; Tello Guerra Omar Salvador; Alexander Vladimir Paco Cornejo.

## Ejecutar de extremo a extremo

Prerrequisitos: Windows, Docker Desktop con WSL2, GPU NVIDIA compatible con CUDA 12.8 y Sen1Floods11 v1.1 completo. El entorno está fijado en [requirements.txt](parcial/requirements.txt) y [Dockerfile](parcial/Dockerfile). El entrenamiento utiliza los datos locales y no necesita pesos preentrenados.

Desde `parcial/`:

```powershell
.\scripts\run_flood.ps1
```

El comando construye las imágenes, verifica la caché, comprueba CUDA y pruebas, realiza smoke y seis experimentos completos, audita predicciones y genera tablas, figuras, notebook ejecutado y PDF borrador. El estilo NeurIPS oficial se descarga si falta. La revisión y compilación final son pasos editoriales posteriores.

Los originales deben mantener estos prefijos bajo `E:/Kaggle/Sen1Floods11/`:

```text
v1.1/data/flood_events/HandLabeled/S1Hand/
v1.1/data/flood_events/HandLabeled/LabelHand/
v1.1/data/flood_events/WeaklyLabeled/S1Weak/
v1.1/data/flood_events/WeaklyLabeled/S1OtsuLabelWeak/
v1.1/splits/flood_handlabeled/
download_manifest.json
download_receipt.json
```

Fuente: [Sen1Floods11 oficial](https://github.com/cloudtostreet/Sen1Floods11). Para preparar los datos desde `parcial/`, el servicio `downloads` permite red y escritura de datos. El descargador verifica los archivos frente a GCS y genera `download_receipt.json`; la caché requiere además el inventario fijado de los 9.216 archivos usados en este estudio, incluido en esta rama:

```powershell
docker compose build runner
docker compose run --rm downloads download --dataset flood
Copy-Item -LiteralPath 'delivery/flood/evidence/audit/download_manifest.json' -Destination 'E:/Kaggle/Sen1Floods11/download_manifest.json'
```

Si los datos ya están completos, solo comprobar que el inventario fijado exista en su raíz. Ajustar el destino si se cambia `SEN1_HOST_DIR` y `KAGGLE_ROOT` para la descarga. Estos son pasos explícitos de preparación de entradas. El runner científico posterior tiene red desactivada y datos de solo lectura; la caché comprueba tamaños y MD5 del inventario. [FLOOD_RUN.md](parcial/FLOOD_RUN.md) detalla las exclusiones.

Para una reproducción científica independiente, usar una **salida nueva**:

```powershell
$env:SEN1_HOST_DIR = 'E:/Kaggle/Sen1Floods11'
$env:FLOOD_OUTPUT_DIR = 'E:/Kaggle/exam-output-flood-reproduction'
.\scripts\run_flood.ps1
```

La salida predeterminada es `E:/Kaggle/exam-output-flood`. `-Resume` restaura una corrida interrumpida con la misma firma de configuración/datos; sobre corridas completas vuelve a escribir evaluaciones y activos. No usarlo como comprobación de solo lectura. El notebook conserva `RUN_TRAINING=False` y consulta los resultados persistidos.

## Consultar los resultados sin reentrenar

- [Notebook ejecutado](parcial/notebooks/02_flood.ipynb).
- [Métricas de las seis variantes](parcial/delivery/flood/evidence/flood/experiments.csv) y [manifiesto](parcial/delivery/flood/evidence/flood/manifest.json).
- [Evidencia de integridad y validación](parcial/delivery/flood/evidence/audit/).
- [Tablas y figuras](parcial/delivery/flood/evidence/reports/flood/).
- [Guion de exposición](parcial/PRESENTACION_FLOOD.md).

Los 446 chips manuales son el total, no el entrenamiento. Después de nodata: 251 train, 86 val, 88 test y 15 Bolivia; 4.150 débiles válidos fuera de Bolivia. El inventario guardado tiene 4.384 pares débiles frente a 4.385 en el enunciado. La IoU por chip distingue unión vacía de IoU global. La comparación tiene una semilla y 63 frente a 1.101 pasos por época, por lo que no aísla el efecto causal de la supervisión débil. Las causas físicas SAR propuestas se mantienen como hipótesis.

Los datos, checkpoints y GeoTIFF pesados se conservan externamente. Esta rama contiene el paquete compartido necesario para ejecutar la pregunta 2; no declara la entrega final de otras preguntas. La revisión del grupo y el comprobante de plataforma deben registrarse después de ocurrir.
