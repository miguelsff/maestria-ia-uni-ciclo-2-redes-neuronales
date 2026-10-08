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

La pregunta 2 tiene seis experimentos completos, auditoría aprobada de 618 predicciones GeoTIFF y notebook ejecutado sin errores. La autoría incorpora los cinco integrantes y la afiliación proporcionados por el grupo. El [paquete reproducible en GitHub](https://github.com/miguelsff/maestria-ia-uni-ciclo-2-redes-neuronales/tree/codex/flood-delivery) está publicado y verificado desde una copia nueva. Usar [FLOOD_RUN.md](FLOOD_RUN.md) para resultados, configuración congelada y PDF de revisión; queda pendiente la aprobación editorial del grupo y la entrega por la plataforma.

## Reproducción de la pregunta 2 con un comando

Desde `parcial/`, con Docker Desktop, WSL2, GPU NVIDIA compatible con CUDA 12.8 y los datos completos de Sen1Floods11 v1.1:

```powershell
.\scripts\run_flood.ps1
```

El comando construye el entorno, verifica la caché, ejecuta controles, entrena y evalúa las seis variantes y genera tablas, figuras, notebook ejecutado y PDF borrador NeurIPS. Los originales deben conservar su estructura bajo `E:/Kaggle/Sen1Floods11/v1.1`; los resultados se escriben en `E:/Kaggle/exam-output-flood`. La obtención de los datos está documentada en la sección Sen1Floods11 y en [FLOOD_RUN.md](FLOOD_RUN.md).

Para una reproducción independiente, configurar **una salida nueva** antes del comando:

```powershell
$env:SEN1_HOST_DIR = 'E:/Kaggle/Sen1Floods11'
$env:FLOOD_OUTPUT_DIR = 'E:/Kaggle/exam-output-flood-reproduction'
.\scripts\run_flood.ps1
```

`SEN1_HOST_DIR` cambia la ubicación de los originales y `FLOOD_OUTPUT_DIR` la salida. `-Resume` recupera una ejecución interrumpida con la misma firma de configuración/datos; sobre corridas completas vuelve a escribir evaluaciones y activos. Los seis experimentos ya auditados no necesitan volver a entrenarse para completar la entrega. El comando produce un borrador; la revisión editorial y la compilación final son pasos de entrega posteriores.

La pregunta 3 continúa en v6: utiliza YOLOv8s existente y conserva las 90 fotografías aprobadas, incluida la tercera caja corregida según el recuadro rojo del revisor. V6 resuelve la compatibilidad de algunos JPEG mediante copias con píxeles idénticos y mantiene los originales y versiones anteriores. Las notas visuales individuales son opcionales. Continuar con [WILDLIFE_V6_RUN.md](WILDLIFE_V6_RUN.md). Para MRI prevalece MRI_RUN.md y para fauna prevalece la guía v6.

## 1. Prerrequisitos

- Windows con Docker Desktop, backend WSL2 y contenedores Linux; integración de GPU NVIDIA habilitada.
- Driver NVIDIA compatible con CUDA 12.8 y una RTX 5090 reconocida en el equipo.
- Cuenta Kaggle con acceso autorizado a la competencia para los datos MRI.
- Espacio para el ZIP, su extracción y resultados. Mantener los datos en E: evita incorporarlos al repositorio.
- Datos completos y descomprimidos, y pesos descargados explícitamente antes de entrenar.

Hardware consultado durante la planificación: Ryzen 9 9950X3D, 16 núcleos/32 hilos, aproximadamente 61.6 GiB de RAM y RTX 5090 con unos 32 GB de VRAM. Compose limita cada servicio de entrenamiento a **8 CPU, 12 GB de RAM y 2 GB de memoria compartida**; utiliza una GPU. Esos valores son límites, no una medición de consumo. La disponibilidad real de RAM, CUDA y los batches se comprobará en la siguiente etapa.

El servicio de descargas usa 2 CPU/2 GB, y el de informes 1 CPU/2 GB. Los entrenamientos son secuenciales; no iniciar varios servicios de entrenamiento o notebooks activos a la vez. El bloqueo por directorio de salida evita dos experimentos sobre los mismos resultados, pero no reserva toda la GPU frente a otras aplicaciones.

Entorno preparado: Python 3.12, PyTorch 2.10.0, torchvision 0.25.0, CUDA 12.8 y Ultralytics 8.4.21. Las versiones directas están fijadas en requirements.txt; la resolución completa de dependencias y compatibilidad se verificará al construir. La base CUDA no fija la versión del driver del host.

## 2. Descargas necesarias y ubicación

### MRI de rodilla

Los originales completos están en E:/Kaggle/rsna-knee-abnormality-detection y se verificó la correspondencia de sus CSV con data/train.csv y data/train_series.csv. La ejecución MRI seleccionó y auditó 600 series sin volver a descargar el dataset. Los archivos test.csv, test_series.csv, sample_submission.csv y la carpeta test_series no son necesarios para la evaluación local con etiquetas expertas.

Fuente: [RSNA Knee Abnormality Detection — Data](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data).

Estructura esperada:

~~~text
E:\Kaggle\
  rsna-knee-abnormality-detection\
    train_series\
      <StudyInstanceUID>\
        <SeriesInstanceUID>\
          <SOPInstanceUID>.dcm
~~~

También se detecta una carpeta intermedia de extracción. Si hay varias carpetas train_series, configurar RSNA_DATA_DIR explícitamente. No mover, renombrar, abrir como archivo comprimido ni extraer un archivo .crdownload que aún esté en progreso.

La muestra final tiene 434 estudios de entrenamiento, 108 de validación débil y 58 completamente etiquetados por expertos para evaluación. Entrenamiento y parada temprana usan supervisión débil separada por estudio; las doce etiquetas proceden de los CSV reales.

### Sen1Floods11

Fuente: [repositorio y documentación oficial](https://github.com/cloudtostreet/Sen1Floods11).
Se necesita la versión **v1.1**, con estos cinco prefijos del bucket público gs://sen1floods11:

~~~text
v1.1/data/flood_events/HandLabeled/S1Hand/
v1.1/data/flood_events/HandLabeled/LabelHand/
v1.1/data/flood_events/WeaklyLabeled/S1Weak/
v1.1/data/flood_events/WeaklyLabeled/S1OtsuLabelWeak/
v1.1/splits/flood_handlabeled/
~~~

Guardar esa estructura bajo E:\Kaggle\Sen1Floods11\. Las particiones requeridas son flood_train_data.csv, flood_valid_data.csv, flood_test_data.csv y flood_bolivia_data.csv. No se necesitan Sentinel-2 ni checkpoints del repositorio. El script preparado descarga solo SAR, las dos fuentes de etiquetas y particiones. Comprueba tamaño y MD5 publicados por GCS y permite reanudar archivos .part.

Para descarga manual, navegar los prefijos desde el [bucket público](https://console.cloud.google.com/storage/browser/sen1floods11/v1.1), conservando nombres y carpetas. No mezclar archivos de v1.0 y v1.1. Bolivia se excluye tanto de entrenamiento manual como débil.

### CCT20: benchmark oficial

Usar el subconjunto ECCV CCT20 elegido, no la colección completa de 105 GB. Descargar:

1. [eccv_18_all_images_sm.tar.gz — imágenes del benchmark, aproximadamente 6 GB](https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_all_images_sm.tar.gz).
2. [eccv_18_annotations.tar.gz — anotaciones oficiales, aproximadamente 3 MB](https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_annotations.tar.gz).

Extraer ambos bajo E:\Kaggle\CCT20\. Una extracción .tar.gz puede requerir primero descomprimir .gz y luego .tar. Deben quedar las imágenes JPG y eccv_18_annotation_files con train_annotations.json, cis_val_annotations.json, cis_test_annotations.json, trans_val_annotations.json y trans_test_annotations.json.

La [página oficial de LILA](https://lila.science/datasets/caltech-camera-traps) distingue el benchmark de la colección completa. Este benchmark contiene catorce categorías animales; empty y car se excluyen del espacio de especies. Las fotos vacías son fondo, no cajas de una clase «vacío».

La preparación registra el mapeo real, imágenes con cajas incompletas, vacíos, errores y secuencias retiradas de validación por solapamiento con train. Las unidades de bbox se detectan y registran; revisar ese diagnóstico antes de entrenar. bbox_units permite forzar pixels o normalized si una auditoría de los archivos lo requiere.

### Pesos y estilo de informes

El comando explícito de activos descarga ResNet-50 ImageNet, YOLOv8s y un pequeño YOLO26n auxiliar requerido por la comprobación AMP de Ultralytics 8.4.21. El detector del proyecto sigue siendo YOLOv8s.

Los pesos quedan en E:\Kaggle\exam-output\cache\weights\. El entrenamiento no descarga pesos; runner tiene la red deshabilitada. Los notebooks usan modo offline de Ultralytics y los mismos activos locales. Las descargas de activos sin hash publicado verifican la longitud HTTP; no se presentan como verificadas con MD5 del proveedor.

El estilo NeurIPS 2026 se obtiene después mediante download --dataset style desde el [kit oficial](https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip). Sus archivos quedan fuera del repositorio. Los informes son proyectos LaTeX con varios archivos y se compilan mediante el servicio local reports.

## 3. Configurar rutas

Todos los comandos Docker siguientes se ejecutan desde la carpeta parcial/.

~~~powershell
Set-Location D:\trabajos-miguel\MAESTRIA-UNI-IA\maestria-ia-uni-ciclo-2-redes-neuronales\parcial
Copy-Item .env.example .env
New-Item -ItemType Directory -Force E:\Kaggle\exam-output
~~~

Editar .env para indicar KAGGLE_ROOT y OUTPUT_HOST_DIR; reemplazar JUPYTER_TOKEN por un valor propio. La carpeta de datos debe existir. Los notebooks quedan limitados a localhost.

El contenedor lee /datasets y escribe /outputs. runner y notebooks montan datos y código en solo lectura; únicamente el servicio downloads monta los datos con escritura para la descarga explícita de Sen1Floods11 y CCT20. No existe una descarga programada de MRI.

Los parámetros científicos están en configs/default.toml. Las variables RSNA_DATA_DIR, RSNA_METADATA_DIR, SEN1_DATA_DIR, CCT_DATA_DIR, OUTPUT_DIR y PRETRAINED_WEIGHTS_DIR permiten ajustar rutas. Para pasar una variable al contenedor usar -e con una ruta del contenedor:

~~~powershell
docker compose run --rm -e RSNA_DATA_DIR=/datasets/rsna-knee-abnormality-detection runner doctor
~~~

Si se cambia configuración o dataset, usar una salida nueva. Una reanudación con firma distinta se rechaza. Para una salida alternativa en Docker cambiar OUTPUT_HOST_DIR en .env; las rutas internas siguen siendo /outputs.

## 4. Construir y diagnosticar — siguiente etapa

~~~powershell
docker compose build runner reports
docker compose run --rm runner doctor
docker compose run --rm runner doctor --gpu-check
~~~

El primer diagnóstico consulta versiones y carpetas; no valida la descarga completa. La comprobación GPU realiza operaciones CUDA, gradientes y NMS de torchvision. Detener el avance si falla: no se sustituye silenciosamente CUDA por CPU.

Si se decide usar los scripts de descarga preparados:

~~~powershell
docker compose run --rm downloads download --dataset flood
docker compose run --rm downloads download --dataset cct
docker compose run --rm downloads download --dataset weights
docker compose run --rm downloads download --dataset style
~~~

Si los datasets ya se extrajeron manualmente, omitir sus dos comandos de descarga. No hay que ejecutarlos en esta etapa. Interrumpir una descarga conserva .part para reanudación; un error de checksum requiere revisar o retirar exclusivamente ese archivo parcial antes de reintentar.

## 5. Pruebas e integridad — siguiente etapa

~~~powershell
docker compose run --rm --entrypoint python runner -m ruff check --no-cache src scripts tests
docker compose run --rm --entrypoint python runner -m pytest -q -m "not slow" -p no:cacheprovider
docker compose run --rm --entrypoint python runner -m pytest -q -m slow -p no:cacheprovider
docker compose run --rm runner prepare --task all
~~~

Las pruebas usan datos sintéticos pequeños y no descargan datasets. La prueba slow verifica restauración de optimizador, aleatoriedad y orden del loader tras una interrupción en CPU. La preparación real lee DICOM/raster/imágenes, genera cachés y manifiestos, y puede tardar; no entrena.

Revisar manifiestos antes de continuar: particiones independientes, cobertura de hallazgos, series usadas, unidades de cajas, exclusiones y conteos. Se comprueban nombres, correspondencia y decodificación de archivos usados; los sellos de tamaño/fecha detectan cambios para reanudación, pero no son un hash de contenido de todos los datos manuales.

Pruebas pequeñas con datos reales:

~~~powershell
docker compose run --rm runner smoke --task mri
docker compose run --rm runner smoke --task flood
docker compose run --rm runner smoke --task wildlife
~~~

smoke usa una época y datos reducidos; inundaciones ejecuta las seis variantes. Escribe en /outputs/smoke y comparte solo los pesos predescargados cuando corresponden. La preparación de fauna sigue auditando el benchmark completo. Sus métricas no se incorporan a los informes científicos.

## 6. Ejecutar y reanudar — siguiente etapa

Un comando prepara los datos, entrena, evalúa y genera activos de los tres informes, secuencialmente:

~~~powershell
docker compose run --rm runner run --task all
~~~

También se puede ejecutar una pregunta:

~~~powershell
docker compose run --rm runner run --task mri
docker compose run --rm runner run --task flood
docker compose run --rm runner run --task wildlife
~~~

MRI prepara ocho configuraciones y flood seis; fauna ajusta YOLOv8s. Cada entrenamiento conserva checkpoints y metadatos. Una repetición sin --resume sobre resultados existentes se rechaza:

~~~powershell
docker compose run --rm runner run --task all --resume
~~~

Los experimentos ya terminados reutilizan sus mejores pesos y regeneran evaluación. Los interrumpidos continúan desde last.pt. La restauración completa y la equivalencia deberán comprobarse con las pruebas; YOLO mantiene su mecanismo propio de reanudación. Para una prueba CPU explícita:

~~~powershell
docker compose run --rm runner --device cpu smoke --task mri
~~~

## 7. Notebooks

~~~powershell
docker compose --profile notebooks up -d notebooks
~~~

Abrir http://localhost:8890 e introducir el token de .env. Al iniciar se copian los notebooks a /outputs/notebooks solo si no existen; esas copias permiten guardar resultados sin modificar el código montado en solo lectura. Los notebooks de MRI e inundaciones de notebooks/ contienen ejecuciones guardadas y mantienen desactivado el entrenamiento automático.

RUN_PREPARE, RUN_TRAINING y GENERATE_REPORT_ASSETS están desactivados por defecto. Activarlos conscientemente durante la siguiente etapa. No entrenar desde un notebook mientras otro comando utiliza la GPU.

La alternativa nativa requiere Python 3.12 y un entorno separado:

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
$env:KAGGLE_ROOT = 'E:/Kaggle'
$env:OUTPUT_DIR = 'E:/Kaggle/exam-output'
neural-exam doctor --gpu-check
~~~

Esta alternativa evita contenedores, pero no aplica sus límites de CPU/RAM. Compose es la referencia del proyecto. No instalar dependencias ni ejecutar estos pasos durante la preparación.

## 8. Revisión e informes — siguiente etapa

Fauna necesita anotación humana de oclusión:

~~~powershell
docker compose run --rm runner review-occlusion
~~~

Revisar wildlife/occlusion_gallery y completar wildlife/occlusion_review.csv con low, medium o high y evidencia. Usar visibilidad aproximada del cuerpo como criterio: low >75 %, medium 25–75 %, high <25 %; marcar solo casos donde la estimación sea defendible. No usar área de la caja como sustituto de oclusión. Se necesitan al menos 30 objetos por nivel; si la muestra no alcanza, ampliar el CSV con IDs válidos del manifiesto y registrar el criterio de muestreo.

Después, reejecutar wildlife con --resume para recalcular evaluación y tablas. Si un nivel carece de soporte suficiente se informa manual_review_pending, sin inventar resultados. En inundaciones completar qualitative_review.csv con tres tipos de error sustentados por imágenes. En MRI revisar los tres mapas y su plausibilidad anatómica.

~~~powershell
docker compose run --rm runner report --task all
docker compose run --rm reports --task all --draft
~~~

Los borradores quedan en E:\Kaggle\exam-output\reports\*_draft.pdf. El generador usa un baseline fijado previamente para figuras; no selecciona la mejor variante por rendimiento de prueba. Conserva métricas indefinidas como ND.

Completar reports/<pregunta>/discussion.tex y actualizar main.tex con resumen, resultados, conclusiones y enlace real del repositorio. Incorporar nombres/correos de cinco integrantes en reports/preamble.tex. Revisar cifras, interpretaciones, figuras, referencias y 6–10 páginas por pregunta. Las fuentes actuales son borradores metodológicos, no PDF finales ni una garantía de extensión.

Cuando la revisión esté terminada, crear reports/<pregunta>/finalized.json con:

~~~json
{"reviewed": true}
~~~

Compilar sin --draft. El compilador exige esa constancia, activos completos, oclusión resuelta cuando corresponda y 6–10 páginas:

~~~powershell
docker compose run --rm reports --task all
~~~

La revisión visual de los PDF sigue siendo necesaria. La referencia ejemplo.pdf guía la estructura; no se copian sus resultados ni se asumen sus explicaciones causales.

## 9. Salidas y detención

~~~text
E:\Kaggle\exam-output\
  cache\weights\
  mri\                 manifiesto, etiquetas, prevalencia, ocho experimentos
  flood\               manifiesto, seis experimentos, GeoTIFF y errores
  wildlife\            dataset derivado, YOLO, AP, oclusión y actividad
  smoke\               comprobaciones pequeñas, excluidas de informes
  notebooks\           copias editables
  report-assets\       estilo oficial
  reports\             tablas, figuras, borradores y PDF revisados
~~~

Para interrumpir un entrenamiento en primer plano usar Ctrl+C y reanudar posteriormente con --resume. La última época completamente guardada define el punto de reanudación. No eliminar resultados o checkpoints para «desbloquear» una ejecución.

~~~powershell
docker compose stop notebooks
docker compose down
~~~

Los datos y resultados están en carpetas del host y se conservan. Revisar [PREPARATION_STATUS.md](PREPARATION_STATUS.md) antes de iniciar la siguiente etapa.
