# Verificar el cierre sin repetir los experimentos

Los originales científicos permanecen en `E:/Kaggle`. Los adaptadores de esta etapa verifican sus hashes, conservan la firma de oclusión y reconstruyen las rutas internas históricas en una copia privada del worktree. No ejecutan entrenamiento, inferencia ni los 200 bootstraps científicos ya terminados.

## Ejecución mínima

Desde la raíz del repositorio, con Python 3.12 y Docker Desktop:

```powershell
$examPython = 'C:/Users/USER/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
& $examPython -X utf8 parcial/compartido/scripts/closure_runtime.py `
  --exam-root parcial --destination parcial/.runtime/cierre `
  --freeze-receipt E:/Kaggle/exam-output-wildlife-v6/receipts/source-freeze.json
New-Item -ItemType Directory -Force parcial/.runtime/derivados/cache/ipython | Out-Null
Copy-Item parcial/compartido/.env.closure.example parcial/compartido/.env.closure
```

Editar las dos rutas de `.env.closure`: `RUNTIME_HOST_DIR` debe apuntar a `parcial/.runtime/cierre` de este worktree, y `CLOSURE_OUTPUT_HOST_DIR` a su `parcial/.runtime/derivados`. Las rutas deben ser absolutas. La imagen local comprobada fue `neural-exam-wildlife:0.2`; su identidad exacta y las versiones verificadas constan en `auditorias/c1/evidence.json`.

```powershell
docker compose --env-file parcial/compartido/.env.closure -f parcial/compartido/compose.closure.yaml config --quiet
docker compose --env-file parcial/compartido/.env.closure -f parcial/compartido/compose.closure.yaml run --rm verify
docker compose --env-file parcial/compartido/.env.closure -f parcial/compartido/compose.closure.yaml run --rm tests
docker compose --env-file parcial/compartido/.env.closure -f parcial/compartido/compose.closure.yaml run --rm notebook-occlusion
```

`verify` comprueba las 44 fuentes, los archivos del padre, las decisiones humanas, la firma y los resultados completos. `tests` utiliza casos sintéticos, incluido un bootstrap de tres fotogramas; no recalcula la muestra científica. `notebook-occlusion` vuelve a ejecutar cuatro celdas sobre las tablas ya calculadas y escribe una copia derivada, conservando el notebook de entrega.

Las fuentes, V6 y la muestra original se montan en solo lectura. El único montaje escribible es la carpeta privada de derivados. No se solicita GPU y la red está desactivada. El verificador rechaza un entorno que difiera de las versiones congeladas de Torch, Torchvision, Ultralytics y Pillow. Si falta la imagen, el Dockerfile original permite construir el entorno; **esta etapa no ha probado una reconstrucción limpia de la imagen desde Internet**. Su instalación requiere acceso de red y paquetes compatibles. La comparación de versiones tampoco demuestra igualdad completa del sistema operativo o del hardware.

El contexto derivado contiene el Dockerfile original y sus fuentes, incluso después de la migración:

```powershell
docker build --target runtime -t neural-exam-closure:local parcial/.runtime/cierre
```

Si se construye esa imagen, configurar `CLOSURE_IMAGE=neural-exam-closure:local` y ejecutar `verify` para comprobar sus versiones antes de usarla. La construcción de imagen no ejecuta entrenamiento ni inferencia; el comando se documenta y queda sin ejecutar en C1.

El Compose usa `parcial/compartido` como contexto de construcción por defecto, conforme al contrato final. En la transición C1 puede apuntarse `CLOSURE_BUILD_CONTEXT` a la copia derivada, donde los archivos de construcción ya existen. En C2 se comprobará el contexto compartido real. Los montajes siguen usando el runtime derivado para conservar `/workspace/notebooks` y `/workspace/reports` sin cambiar fuentes congeladas.

## Rutas después de la migración

El adaptador admite el árbol anterior y el nuevo árbol por pregunta. Copia archivos, nunca mueve los originales. Debe usarse un destino nuevo bajo `parcial/.runtime/`; si ya existe, se rechaza para evitar sobrescrituras.

| Fuente del repositorio | Ruta científica dentro del contenedor |
| --- | --- |
| `compartido/src`, `configs`, `scripts`, `tests` | `/workspace/src`, `/workspace/configs`, `/workspace/scripts`, `/workspace/tests` |
| `compartido/Dockerfile`, `requirements.txt`, `pyproject.toml` o sus ubicaciones anteriores | `/workspace/Dockerfile`, `/workspace/requirements.txt`, `/workspace/pyproject.toml` |
| `pregunta_*/notebooks/*.ipynb` | `/workspace/notebooks/*.ipynb` |
| `pregunta_1_mri/fuentes_informe/` | `/workspace/reports/mri/` |
| `pregunta_2_inundaciones/fuentes_informe/` | `/workspace/reports/flood/` |
| `pregunta_3_fauna/fuentes_informe/` | `/workspace/reports/wildlife/` |
| Recursos comunes en `compartido/plantillas/` | `/workspace/reports/` |
| `compartido/plantillas/notebook_template.ipynb` | `/workspace/tmp/occlusion-accepted-delivery/notebook_template.ipynb` |
| `pregunta_*/entregable/` | `/workspace/deliveries/pregunta_*/` |

El manifiesto del runtime registra cada hash copiado. La verificación vuelve a comprobar los bytes congelados en el destino. Las reglas `.gitattributes` protegen tanto las rutas anteriores como las compartidas y los entregables. Dos fuentes V6 están congeladas con CRLF; el resto de las fuentes afectadas por la conversión de Git usa LF. No se debe aplicar un formateador o `git add --renormalize` a los archivos firmados.

La plantilla heredada continúa en su ruta anterior durante C1 porque el constructor aceptado la usa por defecto. En C2 se comprobará su traslado a `compartido/plantillas/` y el montaje final de los recursos comunes. No editar los cinco helpers firmados para ajustar rutas: utilizar el runtime derivado.

## Alcance de reproducción

| Operación | Insumos necesarios | Estado en C1 |
| --- | --- | --- |
| Verificar decisiones, firma, resultados e intervalos de oclusión | Originales V6 y muestra aceptada; CPU | Ejecutado, sin repetir bootstrap científico |
| Reejecutar `04_occlusion_582.ipynb` | Notebook, `data/`, entorno CPU con pandas | Ejecutado sobre tablas guardadas, cuatro celdas sin errores |
| Consultar MRI e inundaciones | Notebooks ejecutados copiados con sus salidas | Copia exacta verificada; ocho celdas ejecutadas por notebook, sin errores registrados |
| Reejecutar los notebooks MRI/inundaciones en modo lectura | Manifiestos, auditorías, métricas y figuras de los directorios de resultados; entorno científico | Comandos históricos y banderas disponibles; no reejecutados en C1 |
| Recalcular métricas desde predicciones | Predicciones guardadas, manifiestos y revisiones auténticas; CPU | Posible mediante evaluadores originales; no constituye entrenamiento o inferencia nueva |
| Recalcular los intervalos de oclusión de 582 casos | Predicciones V6, revisiones, protocolo, entorno congelado; CPU | Ya completado en los originales; no se invoca `evaluate_occlusion_accepted.py run` en este cierre |
| Reproducción científica completa desde datos | Datasets y reglas Kaggle, metadatos, revisiones, pesos iniciales, GPU y entorno compatible | No ejecutada ni afirmada |

Los notebooks MRI/inundaciones mantienen `RUN_PREPARE=False`, `RUN_TRAINING=False` y `GENERATE_REPORT_ASSETS=False`. Su reejecución de lectura requiere configurar `OUTPUT_DIR` hacia una copia/montaje de los resultados, mantener sus rutas internas y conservar las figuras. Las auditorías contienen evidencia histórica; mostrarla de nuevo no equivale a repetir el experimento. Los datos, cachés y checkpoints voluminosos no se copian al repositorio.

El constructor aceptado conserva su exportación histórica a PDF. La entrega editorial final exige ODT principal y PDF exportado desde ese ODT, y corresponde a las etapas de documentos; este procedimiento no genera informes ni sustituye esa exportación.
