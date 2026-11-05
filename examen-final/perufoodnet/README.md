# PeruFoodNet · Grupo 3

Proyecto del examen final de Redes Neuronales y Aprendizaje Profundo (MIA-203), Maestría en Inteligencia Artificial de la UNI. El objetivo es reconocer platos peruanos en fotografías y estudiar cómo influyen la arquitectura y la cantidad de imágenes de entrenamiento. El proyecto prepara los datos y compara dos arquitecturas mediante entrenamiento y validación controlados.

## Datos y licencia

Se utiliza PeruFoodNet, versión 3, publicado en Mendeley Data bajo Creative Commons Atribución 4.0 (CC BY 4.0). La copia local está en `E:\Dataset PeruFoodNet` y contiene 40 carpetas, una por plato, con 100 imágenes cada una. La carpeta determina la etiqueta. El número inicial del nombre identifica la imagen dentro de la clase. Al reutilizar los datos, cite la fuente y respete la licencia. [Fuente y atribución](https://data.mendeley.com/datasets/hxhbbm497d/3) · [Licencia](https://creativecommons.org/licenses/by/4.0/).

## Partición y archivos

Las clases se ordenan de forma determinista por nombre, considerando equivalentes las mayúsculas y minúsculas. El nombre original resuelve los empates. El índice comienza en 0 y corresponde a la posición en `data/classes.json`. Dentro de cada clase, las imágenes se ordenan por su número.

| Números | Partición | Total |
|---|---|---:|
| 01–70 | Entrenamiento (`train`) | 2800 |
| 71–85 | Prueba (`test`) | 600 |
| 86–100 | Validación (`val`) | 600 |

Los subconjuntos de entrenamiento usan 01–10 (`subset_10`, 400 imágenes) y 01–40 (`subset_40`, 1600 imágenes). El primero está incluido en el segundo. Las marcas 1 y 0 indican pertenencia o ausencia, respectivamente.

- `data/manifest.csv`: una fila por imagen, con ruta relativa al dataset (`relative_path`, separada por `/`), nombre e índice de clase, número, partición y marcas de ambos subconjuntos.
- `data/classes.json`: lista ordenada de los 40 nombres de clase.
- `data/counts.csv`: conteos por clase, partición y subconjunto.
- `src/prepare_data.py`: generación y comprobación de estos archivos.

La división sigue los números de los archivos. Fotografías relacionadas de una misma ración podrían quedar en conjuntos diferentes. Esta dependencia puede elevar las métricas. Los resultados y conclusiones deberán interpretarse para esta partición secuencial y declarar esta limitación.

## Preparación y control

Esta etapa utiliza la biblioteca estándar de Python. Desde la raíz de `examen-final/perufoodnet`, ejecute el siguiente comando en PowerShell para generar los tres archivos.

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

Cada configuración usa AdamW con decaimiento de pesos 0,01 y entropía cruzada sobre 40 logits. La duración máxima es 200 épocas; la parada temprana ocurre después de 10 épocas consecutivas sin mejora estricta del F1 de validación. Llegar al máximo se registra como límite de presupuesto, sin afirmar convergencia. Se calcula F1 macro con las 40 etiquetas. El mejor checkpoint es el primero con el mayor F1 de validación. El régimen `head` congela los parámetros del extractor y las estadísticas de BatchNorm.

Desde la raíz del proyecto, una prueba corta es:

```powershell
$env:PYTHONIOENCODING = 'utf-8'
& 'E:\MIA203-archivo\Grupo3-20261105-014928\ejecucion\venv\Scripts\python.exe' src/train.py --family resnet18 --regime head --budget 40 --lr 0.0001 --epochs 1
```

Cambie `--family` entre `resnet18` y `yolo26m`, y `--regime` entre `head` y `full`. Para inferencia local, indique un checkpoint y una imagen concretos:

```powershell
& 'E:\MIA203-archivo\Grupo3-20261105-014928\ejecucion\venv\Scripts\python.exe' src/predict.py --image 'E:\ruta\foto.jpg' --checkpoint 'E:\ruta\best.pt' --classes data/classes.json
```

La puntuación es softmax sin calibración; inferior a 0,50 se marca incierta. No equivale a una probabilidad de acierto garantizada.

## MLflow y evidencias

Abra `http://localhost:4567/` y seleccione `PeruFoodNet-Grupo3-20261105-R03`. Cada corrida muestra configuración, métricas por época, entradas de entrenamiento y validación, checkpoint, historial, curvas y modelo nativo. El URI del modelo y el resultado de comparar su salida recuperada con la local se guardan en `model_uri.json`. Las salidas locales se guardan fuera del repositorio en la ruta `output` de la configuración. Consulte la [guía oficial de seguimiento](https://mlflow.org/docs/latest/ml/tracking/) y la [API de PyTorch](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.pytorch.html).

Las comprobaciones cortas usan una época, 40 fotos por clase, tasa 0,0001 y semilla 42. Para ejecutar cada combinación, cambie los argumentos del comando anterior. Un comando individual crea una campaña y su corrida hija; `--parent-id` permite agrupar varias bajo una campaña existente. La matriz queda pendiente de habilitación. Las evidencias de las pruebas R03 están en `E:/MIA203-Grupo3/reinicio-20261105-r03/verified_index.json`; las métricas de una época sirven para comprobar el flujo, no para seleccionar un ganador.

El notebook acepta `PROJECT_OVERRIDE` para señalar el proyecto en Colab. Si los datos locales faltan, puede usar la URL oficial o un enlace público de Google Drive; estas descargas y Colab están preparados en código, pero no se probaron en esta etapa.

## Campaña y selección

Desde la raíz del proyecto, después de fijar en Git los cambios de código, configuración y datos, inicie o reanude la campaña con el mismo comando:

```powershell
$env:PYTHONIOENCODING = 'utf-8'
& 'E:\MIA203-archivo\Grupo3-20261105-014928\ejecucion\venv\Scripts\python.exe' src/experiments.py --config configs/default.json
```

Ejecuta 16 combinaciones con semilla 42 y escoge la tasa por mayor F1 macro de validación; un empate favorece 0,0001. Repite solo las ocho tasas elegidas con semillas 43 y 44. Reanuda corridas finalizadas únicamente cuando la configuración, los archivos y las entradas versionadas son compatibles; cambios en resultados o documentación no fuerzan reentrenamiento. La matriz final usa `E:/MIA203-Grupo3/reinicio-20261105-r03/matriz-convergencia` y registra sus intentos en `matrix_index.json`. La exploración anterior de 15 épocas permanece separada y no participa en la selección. Si la configuración o el código de entrenamiento cambian, el programa rechaza reanudar la carpeta existente y conserva sus archivos. La selección compara medias de tres semillas entre presupuestos de 40 imágenes, usa desviación estándar muestral y resuelve empates exactos por latencia de lote uno en el mismo equipo. El checkpoint oficial es el de semilla 42. `seconds` comprende carga de imágenes, entrenamiento, evaluación por época y registro y recuperación del modelo; termina antes de la última subida de los archivos de la corrida.

La campaña crea `results/corridas.csv`, `comparacion_misma_tasa.csv`, `resumen_validacion.csv`, la figura, historiales y `winner.json`. La comparación a la misma tasa usa semilla 42; el resumen agrupa por la tasa elegida. Para regenerarlos desde un índice completo, indique el ID de la campaña padre:

```powershell
& 'E:\MIA203-archivo\Grupo3-20261105-014928\ejecucion\venv\Scripts\python.exe' src/reports.py --index 'E:\MIA203-Grupo3\reinicio-20261105-r03\matriz-convergencia\matrix_index.json' --parent-id ID_DE_CAMPANA --output 'E:\MIA203-Grupo3\reinicio-20261105-r03\matriz-convergencia'
```

La prueba se reserva hasta que `results/winner.json` esté guardado en Git y la evaluación sea autorizada. Después, el comando es:

```powershell
& 'E:\MIA203-archivo\Grupo3-20261105-014928\ejecucion\venv\Scripts\python.exe' src/evaluate.py --winner results/winner.json --dataset 'E:\Dataset PeruFoodNet' --output 'E:\MIA203-Grupo3\reinicio-20261105-r03\test'
```

La ficha de `predict.py` usa `data/ingredients.json`, obtenido de las tablas 1 y 2 del [estudio](https://doi.org/10.1016/j.dib.2025.111604). Conserva los códigos de ambas columnas y presenta su unión como referencia de una receta, no como ingredientes detectados en la fotografía. Una columna sin elementos enumerados no garantiza ausencia; las recetas pueden variar.

## Regenerar curvas

Las curvas usan tres paneles separados para pérdida, exactitud y F1 macro, con marcadores visibles incluso para una época. Este comando solo lee el CSV y escribe la figura, sin entrenar ni modificar métricas:

```powershell
& 'E:\MIA203-archivo\Grupo3-20261105-014928\ejecucion\venv\Scripts\python.exe' src/curves.py --history 'E:\ruta\history.csv' --output 'E:\ruta\curves.png'
```

Añada `--run-id ID` para actualizar el artefacto de esa corrida en MLflow; la etiqueta `curves_revision` identifica la revisión.
