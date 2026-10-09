# Verificar el cierre sin repetir los experimentos

Los originales científicos permanecen en `E:/Kaggle`. Los adaptadores de esta etapa verifican sus hashes, conservan la firma de oclusión y reconstruyen las rutas internas históricas en una copia privada del worktree. No ejecutan entrenamiento, inferencia ni los 200 bootstraps científicos ya terminados.

## Ejecución mínima

Desde la raíz del repositorio, con Python 3.12 y Docker Desktop:

Un comando prepara una copia nueva, configura los montajes absolutos, comprueba el Compose y verifica el cierre:

```powershell
& 'C:/Users/USER/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -X utf8 parcial/compartido/scripts/run_exam.py closure
```

No requiere editar un archivo de entorno. Cada ejecución usa un directorio nuevo en `parcial/.runtime/closure-*/`, escribe `derived/verification.json` y registra los comandos y códigos reales en `derived/command_receipt.json`. Para ejecutar también las pruebas, añadir `--tests` al mismo comando. `--parent`, `--occlusion` e `--image` permiten indicar otras ubicaciones compatibles sin cambiar archivos; sus valores predeterminados apuntan a los originales preservados y a la imagen local comprobada `neural-exam-wildlife:0.2`. Su identidad y versiones constan en `auditorias/c1/evidence.json`.

El archivo `.env.closure.example` y las invocaciones directas de Compose siguen disponibles para uso manual avanzado. El comando recomendado configura el entorno automáticamente y no lee secretos de ejemplos.

`verify` comprueba las 44 fuentes, los archivos del padre, las decisiones humanas, la firma y los resultados completos. `tests` utiliza casos sintéticos, incluido un bootstrap de tres fotogramas; no recalcula la muestra científica. `notebook-occlusion` vuelve a ejecutar cuatro celdas sobre las tablas ya calculadas y escribe una copia derivada, conservando el notebook de entrega.

Las fuentes, V6 y la muestra original se montan en solo lectura. El único montaje escribible es la carpeta privada de derivados. No se solicita GPU y la red está desactivada. El verificador rechaza un entorno que difiera de las versiones congeladas de Torch, Torchvision, Ultralytics y Pillow. Si falta la imagen, el Dockerfile original permite construir el entorno; **esta etapa no ha probado una reconstrucción limpia de la imagen desde Internet**. Su instalación requiere acceso de red y paquetes compatibles. La comparación de versiones tampoco demuestra igualdad completa del sistema operativo o del hardware.

El contexto derivado contiene el Dockerfile original y sus fuentes, incluso después de la migración:

```powershell
# Usar la ruta derived/command_receipt.json mostrada por el launcher.
$receipt = Get-Content -Raw -LiteralPath '<ruta real>/derived/command_receipt.json' | ConvertFrom-Json
docker build --target runtime -t neural-exam-closure:local $receipt.workspace
```

Si se construye esa imagen, ejecutar el launcher con `--image neural-exam-closure:local` para comprobar sus versiones antes de usarla. La construcción de imagen no ejecuta entrenamiento ni inferencia; el comando se documenta y quedó sin ejecutar en C1 y C2/F.

El Compose usa `parcial/compartido` como contexto de construcción por defecto, conforme al contrato final. El launcher configura automáticamente `CLOSURE_BUILD_CONTEXT` con la ruta real del workspace materializado. C2/F comprobó los archivos de construcción compartidos y los tres planes Compose sin construir imágenes. Los montajes usan el runtime derivado para conservar `/workspace/notebooks` y `/workspace/reports` sin cambiar fuentes congeladas.

## Rutas después de la migración

El adaptador admite el árbol anterior y el nuevo árbol por pregunta. Copia archivos, nunca mueve los originales. Debe usarse un destino nuevo bajo `parcial/.runtime/`; si ya existe, se rechaza para evitar sobrescrituras.

| Fuente del repositorio | Ruta científica dentro del contenedor |
| --- | --- |
| `compartido/src`, `configs`, `scripts`, `tests` | `/workspace/src`, `/workspace/configs`, `/workspace/scripts`, `/workspace/tests` |
| `compartido/Dockerfile`, `requirements.txt`, `pyproject.toml` o sus ubicaciones anteriores | `/workspace/Dockerfile`, `/workspace/requirements.txt`, `/workspace/pyproject.toml` |
| `pregunta_*/notebooks/*.ipynb` | `/workspace/notebooks/*.ipynb` |
| `pregunta_1_mri/fuentes_informe/latex_previo_d1/` | `/workspace/reports/mri/` |
| `pregunta_2_inundaciones/fuentes_informe/latex_previo_d1/` | `/workspace/reports/flood/` |
| `pregunta_3_fauna/fuentes_informe/` | `/workspace/reports/wildlife/` |
| Recursos comunes en `compartido/plantillas/informes/` | `/workspace/reports/`, sin el prefijo `informes/` |
| HTML/JS de `compartido/plantillas/reviews/` | `/workspace/reviews/`, sin el prefijo `plantillas/` |
| `compartido/plantillas/notebook_template.ipynb` | `/workspace/tmp/occlusion-accepted-delivery/notebook_template.ipynb` |
| `pregunta_*/entregable/` | `/workspace/deliveries/pregunta_*/` |

Cuando no existe `main.tex` en la raíz de `fuentes_informe`, el adaptador reconstruye el directorio científico desde `latex_previo_d1`, eliminando ese prefijo. Si la raíz contiene `main.tex`, prevalece esa fuente, como ocurre con fauna A. `latex_original` conserva las fuentes documentales de B y no reemplaza los recursos del runner histórico. Los ODT y sus PDF exportados permanecen como entrega editorial final.

El manifiesto del runtime registra cada hash copiado. La verificación vuelve a comprobar los bytes congelados en el destino. Las reglas `.gitattributes` protegen tanto las rutas anteriores como las compartidas y los entregables. Dos fuentes V6 están congeladas con CRLF; el resto de las fuentes afectadas por la conversión de Git usa LF. No se debe aplicar un formateador o `git add --renormalize` a los archivos firmados.

La plantilla firmada está en `compartido/plantillas/notebook_template.ipynb`; el adaptador la copia a la ruta histórica del constructor dentro del runtime. C2/F comprobó sus bytes y los recursos comunes de informes y revisión HTML/JS. Los cinco helpers firmados conservan sus bytes: sus rutas se resuelven mediante el runtime derivado.

## Comprobación C2/F del árbol migrado

La aceptación se ejecutó en un clon limpio con `core.longpaths=true` y `core.autocrlf=true`, creado desde D2 más la corrección del adaptador. No se repararon bytes después del checkout. Pasaron 91 comprobaciones físico/Git sobre **88 archivos únicos**, las 44 fuentes congeladas y los siete helpers, prueba y plantilla conservados de C1. Los recibos C1 siguen siendo antecedentes; los resultados vigentes están en [auditorias/c2/README.md](auditorias/c2/README.md).

El comando `run_exam.py closure --tests` terminó con código cero y 23 pruebas aprobadas. Los tres planes científicos, sus recursos y configuraciones Compose resolvieron sin ejecutar runners. La suite funcional sintética pasó 124 pruebas, omitió tres dependientes de CUDA y excluyó la prueba `slow` que entrena para comprobar reanudación. Pasaron cuatro pruebas de contenido documental en el renderer y el control JavaScript de revisión offline. Los cuatro notebooks finales validaron su esquema, hashes y procedencia; sus 8/8/16/4 celdas de código tienen ejecución registrada y cero errores, sin reejecución científica.

Esta aceptación conserva firma, resultados y 200 réplicas existentes por nivel. No ejecutó entrenamiento, inferencia sobre datos científicos ni bootstrap científico. La reconstrucción limpia de la imagen y la reproducción completa desde entrenamiento siguen sin verificar. La revisión editorial y la edición/exportación final de los ocho documentos corresponden al control documental paralelo de B.

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

## Un comando científico completo por pregunta

Los tres comandos siguientes conservan los runners históricos byte por byte dentro de una nueva copia del runtime, cargan los ejemplos públicos de entorno y redirigen todas las variables de salida a un directorio **nuevo** indicado explícitamente. Sus etapas originales incluyen preparación, comprobaciones del entorno, entrenamiento, evaluación, notebook y artefactos históricos. Requieren Windows, PowerShell, Docker Desktop/WSL, GPU NVIDIA, datos y metadatos Kaggle disponibles, pesos iniciales y revisiones auténticas. Las descargas requieren red. Las rutas de datos predeterminadas siguen siendo las de los Compose originales; pueden sobreescribirse con las variables documentadas por esos Compose.

MRI:

```powershell
& 'C:/Users/USER/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -X utf8 parcial/compartido/scripts/run_exam.py mri --scientific-output E:/Kaggle/reproduccion-mri-nueva --execute-science
```

Inundaciones:

```powershell
& 'C:/Users/USER/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -X utf8 parcial/compartido/scripts/run_exam.py flood --scientific-output E:/Kaggle/reproduccion-flood-nueva --execute-science
```

Fauna V6:

```powershell
& 'C:/Users/USER/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -X utf8 parcial/compartido/scripts/run_exam.py wildlife --scientific-output E:/Kaggle/reproduccion-wildlife-nueva --execute-science
```

El launcher invoca respectivamente `scripts/run_mri.ps1`, `run_flood.ps1` y `run_wildlife.ps1 -Variant v6`, desde el runtime. Fauna toma el padre V5 indicado por su ejemplo de entorno y conserva las puertas científicas del runner original. No inventa revisiones ni cambia decisiones aceptadas. Los runners originales administran cachés/volúmenes Docker y pueden descargar insumos; esta ruta científica tiene efectos adicionales y no comparte la garantía de solo lectura del modo `closure`. Las salidas voluminosas de estos ejemplos permanecen en E:.

Estos comandos científicos **no se ejecutaron en C1 ni C2/F**. Se verificaron únicamente los planes, sus recursos y la preservación de los runners. La reproducción desde entrenamiento sigue **no verificada**. Sin `--execute-science`, o añadiendo `--plan-only`, el launcher prepara y registra el plan sin ejecutar el runner. El directorio científico debe no existir: se rechaza una salida ya existente para impedir reutilizar los originales. El token local de Jupyter se genera en el entorno privado del runtime; no aparece en los recibos ni en Git. La salida editorial final ODT/PDF sigue siendo un proceso separado de los artefactos históricos de los runners.
