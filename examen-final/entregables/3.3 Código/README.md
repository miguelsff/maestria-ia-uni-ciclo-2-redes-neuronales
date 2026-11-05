# PeruFoodNet · Grupo 3

Proyecto del examen final de Redes Neuronales y Aprendizaje Profundo (MIA-203), Maestría en Inteligencia Artificial de la UNI. El objetivo es reconocer platos peruanos en fotografías y estudiar cómo influyen la arquitectura y la cantidad de imágenes de entrenamiento. El proyecto prepara los datos y compara dos arquitecturas mediante entrenamiento y validación controlados.

## Ruta rápida para el profesor

1. Abra `notebooks/02_modelo_final_profesor.ipynb` desde el repositorio o desde el paquete `Equipo3_Codigo.zip`. El cuaderno explica sus rutas editables y permite usar la foto incluida o una propia.
2. Disponga de Python con PyTorch, torchvision, Ultralytics, pandas, Pillow, matplotlib e IPython y del checkpoint final `perufoodnet_best.pt`. El checkpoint es un adjunto separado del ZIP: colóquelo junto al README en `3.3 Código` o en la raíz del ZIP extraído y ejecute todas las celdas. Una ruta indicada en `WEIGHTS_PATH` tiene prioridad; si no se indica, el cuaderno busca primero el adjunto local y después la ruta de la campaña en esta PC (`E:/MIA203-Grupo3/reinicio-20261105-r03/matriz-convergencia/perufoodnet_best.pt`). Se comprobará su integridad automáticamente.
3. Las URL configuradas en el cuaderno apuntan a los adjuntos `Equipo3_Codigo.zip` y `perufoodnet_best.pt` de la versión `v1.0-final`. Para Colab, el cuaderno contempla descargarlos e instalar dependencias. Verificado localmente en Windows con RTX 5090; Colab y las descargas están preparados, pero no se ejecutaron. La demostración no requiere el dataset completo, Git ni MLflow.

La foto incluida `data/ejemplo_aguadito.jpg` procede de `Aguadito/71_Aguadito.jpg` de [PeruFoodNet v3](https://data.mendeley.com/datasets/hxhbbm497d/3), bajo [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Es la primera predicción correcta y no incierta en el orden de `results/test/predicciones.csv`; se usa para ilustrar inferencia. El cuaderno `notebooks/01_exploracion_datos.ipynb` presenta los datos y requiere el dataset completo.

## Datos y licencia

Se utiliza PeruFoodNet, versión 3, publicado en Mendeley Data bajo Creative Commons Atribución 4.0 (CC BY 4.0). La copia local está en `E:\Dataset PeruFoodNet` y contiene 40 carpetas, una por plato, con 100 imágenes cada una. La carpeta determina la etiqueta. El número inicial del nombre identifica la imagen dentro de la clase. Al reutilizar los datos, cite la fuente y respete la licencia. [Fuente y atribución](https://data.mendeley.com/datasets/hxhbbm497d/3) · [Licencia](https://creativecommons.org/licenses/by/4.0/).

## Partición y archivos

Las clases se ordenan de forma determinista por nombre, considerando equivalentes las mayúsculas y minúsculas. El nombre original resuelve los empates. El índice comienza en 0 y corresponde a la posición en `data/classes.json`. Dentro de cada clase, las imágenes se ordenan por su número.

| Números | Partición | Total |
|---|---|---:|
| 01–70 | Entrenamiento (`train`) | 2800 |
| 71–85 | Prueba (`test`) | 600 |
| 86–100 | Validación (`val`) | 600 |

La partición ofrece 2800 imágenes elegibles para entrenamiento; las corridas usan únicamente 01–10 (`subset_10`, 400 imágenes) o 01–40 (`subset_40`, 1600 imágenes). El primero está incluido en el segundo. Las demás imágenes de entrenamiento no participan en estos experimentos. Las marcas 1 y 0 indican pertenencia o ausencia, respectivamente.

- `data/manifest.csv`: una fila por imagen, con ruta relativa al dataset (`relative_path`, separada por `/`), nombre e índice de clase, número, partición y marcas de ambos subconjuntos.
- `data/classes.json`: lista ordenada de los 40 nombres de clase.
- `data/counts.csv`: conteos por clase, partición y subconjunto.
- `src/prepare_data.py`: generación y comprobación de estos archivos.

La división sigue los números de los archivos. Fotografías relacionadas de una misma ración podrían quedar en conjuntos diferentes. Esta dependencia puede elevar las métricas. Los resultados y conclusiones deberán interpretarse para esta partición secuencial y declarar esta limitación.

## Preparación y control

Esta etapa utiliza la biblioteca estándar de Python. Desde la raíz de `examen-final/entregables/3.3 Código`, ejecute el siguiente comando en PowerShell para generar los tres archivos.

```powershell
& 'E:\MIA203-archivo\Grupo3-20261105-014928\ejecucion\venv\Scripts\python.exe' src/prepare_data.py --dataset 'E:\Dataset PeruFoodNet'
```

El siguiente comando comprueba los archivos existentes.

```powershell
& 'E:\MIA203-archivo\Grupo3-20261105-014928\ejecucion\venv\Scripts\python.exe' src/prepare_data.py --dataset 'E:\Dataset PeruFoodNet' --verify
```

El control lee las carpetas y comprueba 40 clases y los números únicos 1–100 por clase. Contrasta todas las filas, rutas, etiquetas, índices, particiones, marcas y conteos con los archivos reales y las reglas anteriores, incluido el anidamiento de subconjuntos. Una salida `Verified` con 4000 imágenes, 2800/600/600 por partición y 400/1600 por subconjunto indica éxito; un mensaje `Error` y código de salida distinto de cero indican que falta un archivo o existe alguna discrepancia. La comprobación valida nombres y estructura, no el contenido visual de las imágenes.

## Entorno y flujo

El entorno local usa Python y CUDA compatibles con la RTX 5090. Los pesos iniciales locales son `resnet18-f37072fd.pth` (ImageNet-1K V1) y `yolo26m-cls.pt`. Para instalar en otro entorno compatible, instale primero PyTorch y torchvision con la variante CUDA adecuada; después ejecute `python -m pip install -r requirements.txt`. Las versiones del entorno comprobado están fijadas en ese archivo.

Orden de lectura: `src/prepare_data.py` prepara y verifica el manifiesto; `datasets.py` carga y transforma imágenes; `models.py` adapta las redes; `evaluate.py` calcula métricas; `train.py` entrena una configuración; `tracking.py` registra evidencias; `experiments.py` coordina corridas; y `predict.py` procesa una imagen. `resources.py` resuelve los recursos locales o la descarga configurada del cuaderno. El cuaderno `notebooks/01_exploracion_datos.ipynb` explica los datos y no entrena.

`configs/default.json` contiene rutas, servidor, lote, tamaño, semilla, épocas y caché. Ambos modelos reciben imágenes de 224 × 224, con volteo horizontal aleatorio solo para entrenar. ResNet usa normalización ImageNet; YOLO usa valores entre 0 y 1 sin normalización adicional. La caché carga imágenes redimensionadas en RAM y aplica el aumento después; desactívela con `"cache": false` si falta memoria. En esta PC, con 1600 imágenes y dos recorridos de carga y transformación, sin caché tomó 21,57 y 21,90 s; con caché tomó 20,10 s de carga inicial y 0,29 s por recorrido, con unos 321 MiB adicionales de RAM. Esta medición no incluye entrenamiento ni representa una comparación completa de tiempo por época.

Los factores comparados son arquitectura (ResNet-18 o YOLO26m), régimen (capa final o red completa), presupuesto (10 o 40 fotos por clase) y tasa inicial (0,0001 o 0,0003). La selección usa validación y se repite con tres semillas. Se controlan el tamaño 224 × 224, lote 16, aumentos, AdamW con decaimiento de pesos 0,01 y entropía cruzada sobre 40 logits. La duración máxima es 200 épocas; la parada temprana ocurre después de 10 épocas consecutivas sin mejora estricta del F1 de validación. Llegar al máximo se registra como límite de presupuesto, sin afirmar convergencia. Se calcula F1 macro con las 40 etiquetas. El mejor checkpoint es el primero con el mayor F1 de validación. El régimen `head` congela los parámetros del extractor y las estadísticas de BatchNorm.

Desde la raíz del proyecto, una prueba corta es:

```powershell
$env:PYTHONIOENCODING = 'utf-8'
& 'E:\MIA203-archivo\Grupo3-20261105-014928\ejecucion\venv\Scripts\python.exe' src/train.py --config RUTA_CONFIG_PROPIA.json --family resnet18 --regime head --budget 40 --lr 0.0001 --epochs 1
```

Cree primero una copia de `configs/default.json` y configure en ella una carpeta `output` propia. Cambie `--family` entre `resnet18` y `yolo26m`, y `--regime` entre `head` y `full`. Para inferencia local, indique un checkpoint y una imagen concretos:

```powershell
& 'E:\MIA203-archivo\Grupo3-20261105-014928\ejecucion\venv\Scripts\python.exe' src/predict.py --image 'E:\ruta\foto.jpg' --checkpoint 'E:\ruta\best.pt' --classes data/classes.json
```

La puntuación es softmax sin calibración; inferior a 0,50 se marca incierta. No equivale a una probabilidad de acierto garantizada.

## MLflow y evidencias

Abra `http://localhost:4567/` y seleccione `PeruFoodNet-Grupo3-20261105-R03`. Cada corrida muestra configuración, métricas por época, entradas de entrenamiento y validación, checkpoint, historial, curvas y modelo nativo. El URI del modelo y el resultado de comparar su salida recuperada con la local se guardan en `model_uri.json`. Las salidas locales se guardan fuera del repositorio en la ruta `output` de la configuración. Consulte la [guía oficial de seguimiento](https://mlflow.org/docs/latest/ml/tracking/) y la [API de PyTorch](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.pytorch.html).

Las comprobaciones cortas usan una época, 40 fotos por clase, tasa 0,0001 y semilla 42. Para ejecutar cada combinación, cambie los argumentos del comando anterior. Un comando individual crea una campaña y su corrida hija; `--parent-id` permite agrupar varias bajo una campaña existente. Los resultados de validación se documentan más abajo. Las evidencias de las pruebas R03 están en `E:/MIA203-Grupo3/reinicio-20261105-r03/verified_index.json`; las métricas de una época sirven para comprobar el flujo, no para seleccionar un ganador.

El notebook acepta `PROJECT_OVERRIDE` para señalar el proyecto en Colab. Si los datos locales faltan, puede usar la URL oficial o un enlace público de Google Drive; estas descargas y Colab están preparados en código, pero no se probaron en esta etapa.

## Campaña y selección

La campaña oficial R03.1 está finalizada. Para consultar su selección use `results/winner.json` y `results/matrix_index.json`; para regenerar sus tablas y figuras en una carpeta aparte, desde la raíz del proyecto:

```powershell
python src/reports.py --results results --output RUTA_DE_REPORTES
```

La regeneración usa los historiales y las predicciones incluidas, sin checkpoints, dataset completo, servidor MLflow ni GPU. La carpeta indicada en `--output` debe ser nueva o estar vacía, y ser distinta de `results/`: la evidencia oficial se conserva sin cambios. El programa comprueba que los identificadores de la evaluación de prueba correspondan al ganador y no cambia `winner.json`.

Para regenerar los recursos del artículo desde esta misma raíz, con `3.1 Artículo técnico en formato IEEE` como carpeta hermana, use una carpeta de salida nueva y separada:

```powershell
python '../3.1 Artículo técnico en formato IEEE/fuente/generate_resources.py' --results 'results' --output 'RUTA_SALIDA_NUEVA'
```

El generador necesita NumPy, pandas y matplotlib. Produce dos tablas LaTeX (`validation_rows.tex` y `ablation_rows.tex`), las curvas y la matriz de confusión en PDF y una copia de la foto de error en `figuras/`. Las demás tablas se componen en `paper.tex`. Si solo dispone de los dos ZIP, extraiga el ZIP de código y `Equipo3_LaTeX.zip` en carpetas distintas; desde la raíz del código extraído, use el generador de la carpeta LaTeX extraída:

```powershell
python 'RUTA_FUENTES_LATEX/generate_resources.py' --results 'results' --output 'RUTA_SALIDA_NUEVA'
```

El generador puede sobrescribir esos cinco nombres si ya existen en la salida; elija una carpeta nueva fuera de `results/` y de las fuentes del artículo.

Para iniciar una campaña nueva, configure las rutas del dataset, pesos iniciales y servidor en una copia de `configs/default.json`, confirme los cambios de código en Git y use un destino propio vacío:

```powershell
python src/experiments.py --config RUTA_CONFIG_PROPIA.json --output RUTA_CAMPANA_NUEVA
```

La campaña guarda su índice, sus pesos y sus reportes de validación bajo ese destino; la selección queda en `RUTA_CAMPANA_NUEVA/results`. No modifica la carpeta `results/` oficial del proyecto. Ejecuta 16 combinaciones con semilla 42, elige la tasa por F1 de validación y repite las ocho seleccionadas con semillas 43 y 44. Un empate de tasa favorece 0,0001. El ganador entre las configuraciones de 40 fotos maximiza la media de las tres semillas; un empate exacto usa la latencia.

Para retomar una campaña interrumpida, use el mismo destino y configuración con el commit de fuente registrado en `matrix_campaign.json`. Por ejemplo, para una campaña creada después del traslado a `examen-final/entregables/3.3 Código`:

```powershell
git worktree add --detach RUTA_TRABAJO COMMIT_FUENTE_REGISTRADO
cd 'RUTA_TRABAJO/examen-final/entregables/3.3 Código'
python src/experiments.py --config RUTA_CONFIG_PROPIA.json --output RUTA_CAMPANA_EXISTENTE
```

La guarda comprueba las dependencias de entrenamiento y la integridad de las corridas con rutas relativas a la raíz del proyecto. Rechaza cambios incompatibles sin repetirlas. La fuente histórica de la matriz oficial es `5c213faf3913bc0ae793a6fd23d270babcd63245`, cuando el proyecto estaba en `examen-final/perufoodnet`; la interfaz de `evaluate.py` también cambió después de entrenar. La versión actual rechaza reabrir esa carpeta aunque el evaluador numérico sea idéntico. Como la campaña oficial está completa, consulte o regenere sus archivos en vez de reanudarla. `seconds` comprende lectura, entrenamiento, evaluaciones por época y registro y recuperación del modelo; termina antes de la última subida de artefactos.

Para evaluar un checkpoint seleccionado, indique las rutas locales. El comando no requiere Git ni MLflow. Si conoce el commit que fijó esa selección, proporciónelo con `--selection-commit`; si no lo proporciona ni figura en la selección, la procedencia se registra como `unknown`.

```powershell
python src/evaluate.py --winner results/winner.json --checkpoint RUTA_AL_CHECKPOINT --manifest data/manifest.csv --classes data/classes.json --dataset RUTA_AL_DATASET --output RUTA_DE_SALIDA --selection-commit 61fe0460c2ff98cdebacc01cec79e9f5118dfbe4
```

El commit mostrado corresponde solo al ganador oficial; para otra selección use su propio commit o deje fuera el argumento.

La ficha de `predict.py` usa `data/ingredients.json`, obtenido de las tablas 1 y 2 del [estudio](https://doi.org/10.1016/j.dib.2025.111604). Conserva los códigos de ambas columnas y presenta su unión como referencia de una receta, no como ingredientes detectados en la fotografía. Una columna sin elementos enumerados no garantiza ausencia; las recetas pueden variar.

## Regenerar curvas

Las curvas usan tres paneles separados para pérdida, exactitud y F1 macro, con marcadores visibles incluso para una época. Este comando solo lee el CSV y escribe la figura, sin entrenar ni modificar métricas:

```powershell
& 'E:\MIA203-archivo\Grupo3-20261105-014928\ejecucion\venv\Scripts\python.exe' src/curves.py --history 'E:\ruta\history.csv' --output 'E:\ruta\curves.png'
```

Añada `--run-id ID` para actualizar el artefacto de esa corrida en MLflow; la etiqueta `curves_revision` identifica la revisión.

## Resultados de validación

La selección se calculó a partir de `results/matrix_index.json` y se conserva en `results/winner.json`. La configuración seleccionada es yolo26m, red completa, 40 fotos por clase y tasa 0.0001; obtuvo F1 macro medio 0.8903 y desviación muestral 0.0076 en tres semillas. Estas cifras se generaron desde el archivo de selección y corresponden a validación.

La tabla `results/resumen_validacion.csv` contiene las ocho medias con sus tasas elegidas; `results/comparacion_misma_tasa.csv` permite comparar regímenes a igual tasa con semilla 42. `results/corridas.csv` y `results/histories/` conservan los identificadores y las curvas numéricas por época. `results/comparacion_validacion.png` muestra media y desviación. El checkpoint oficial de semilla 42 está en la ruta indicada por `winner.json`, fuera de Git; el URI del modelo nativo y el orden de clases también están en ese archivo. El código de entrenamiento se identifica en `training_source_commit`; las interfaces posteriores de evaluación y reportes cambiaron sin modificar el modelo ni repetir el entrenamiento.

Las métricas deben interpretarse considerando que fotografías de una misma ración podrían aparecer en conjuntos distintos.

## Ejecución portable y seguimiento

Desde la raíz del proyecto, prepare un entorno con `python -m pip install -r requirements.txt`. Para verificar los datos use `python src/prepare_data.py --dataset RUTA_AL_DATASET --verify`. Para una corrida individual, use `python src/train.py --config RUTA_CONFIG_PROPIA.json --family resnet18 --regime head --budget 40 --lr 0.0001` después de configurar un destino propio. Para una campaña nueva o interrumpida, use `python src/experiments.py --config RUTA_CONFIG_PROPIA.json --output RUTA_CAMPANA`, como se explica arriba.

Use una carpeta de salida nueva si ya existe `predicciones.csv`; el programa evita sobrescribir una evaluación anterior. El comando de evaluación admite `--tracking-uri http://localhost:4567/` para registrar la corrida, las entradas de prueba y las métricas asociadas al modelo seleccionado. Sin esa opción no se conecta a MLflow. Para una instalación nueva puede iniciarse `mlflow server --host 127.0.0.1 --port 4567`; el almacén de esa instancia es independiente del servidor que guardó esta campaña. En la interfaz, **Metrics** muestra pérdida, exactitud y F1 por época; **Artifacts** reúne configuración, historial, curvas y checkpoint; **Models** muestra el modelo nativo y su esquema.

Ejemplo de evaluación en la PC usada por el grupo, desde la raíz del proyecto:

```powershell
& 'E:\MIA203-archivo\Grupo3-20261105-014928\ejecucion\venv\Scripts\python.exe' src/evaluate.py --winner results/winner.json --checkpoint 'E:\MIA203-Grupo3\reinicio-20261105-r03\matriz-convergencia\perufoodnet_best.pt' --manifest data/manifest.csv --classes data/classes.json --dataset 'E:\Dataset PeruFoodNet' --output 'E:\MIA203-Grupo3\reinicio-20261105-r03\reproduccion-prueba' --tracking-uri http://localhost:4567/ --selection-commit 61fe0460c2ff98cdebacc01cec79e9f5118dfbe4
```

La duración total de evaluación incluye carga del checkpoint, preparación de datos y una pasada completa de inferencia; excluye gráficos, registro y benchmark. La latencia es el tiempo medio de la red para una entrada sintética de lote uno en el dispositivo indicado, después de calentamiento; excluye lectura y preprocesamiento. No equivale al tiempo completo de procesar una fotografía en CPU. Todas las predicciones, incluidas las de puntuación inferior a 0,50, cuentan en las métricas.

## Resultados de prueba

El checkpoint elegido obtuvo exactitud 0.9050 y F1 macro 0.9043 sobre 600 fotografías. Se marcaron 50 predicciones con puntuación inferior a 0,50; todas están incluidas en las métricas. Estas cifras se obtuvieron de `results/test/metricas.json`.

`results/test/predicciones.csv` contiene la ruta, las clases e índices reales y predichos, puntuación, pérdida por imagen, acierto e incertidumbre. Las métricas por clase, matriz completa y pares confundidos se encuentran en la misma carpeta. Los pares suman errores en ambas direcciones; `analisis_errores.json` conserva todos los empates. `ejemplos/error.jpg` y `ejemplo_error.png` muestran el caso documentado, con fuente y licencia CC BY 4.0. Las listas de ingredientes proceden de recetas publicadas; el modelo no detecta ingredientes en los píxeles.
