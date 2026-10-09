# Pregunta 3: evaluación de CCT20, YOLO y monitoreo ecológico

**Fecha de corte: 4 de octubre de 2026, America/Lima. Veredicto: la pregunta está parcialmente implementada, pero todavía no está contestada con resultados científicos ni lista para entregar.** Hay preparación de datos y evidencia funcional de YOLO; faltan entrenamiento completo, evaluación cis/trans, robustez, interpretación de mapas, demostración ecológica y PDF final. La revisión de etiquetas sigue pendiente en sus 90 filas. No se inició entrenamiento, no se construyeron imágenes Docker, no se modificaron datos, código, notebooks ni informes científicos durante esta evaluación. El único archivo creado es este documento.

## 1. Veredictos separados

| Dimensión | Estado | Conclusión y evidencia decisiva |
|---|---|---|
| Implementación y respuesta al ejercicio | **Parcial** | Existen conversión, detector, evaluación COCO, CAM y módulo ecológico propio. Faltan resultados y su discusión; el entrenamiento contiene cero fotos vacías y la oclusión prevista cubre una muestra de trans_test. |
| Ejecución científica | **Falta** | No existen `E:/Kaggle/exam-output-wildlife/wildlife/yolov8s/run.json`, `training_complete.json`, `evaluation_complete.json` ni `E:/Kaggle/exam-output-wildlife/wildlife/experiments.csv`. |
| Validación | **Parcial** | Se comprobaron hashes, anotaciones, particiones y recálculo de métricas del smoke en CPU. CUDA, AMP y reanudación tienen constancias previas; no se repitieron esas pruebas ni existe validación del experimento final. El compilador cambió respecto del recibo anterior. |
| Notebook | **Parcial** | El fuente tiene 12 celdas de código sin ejecutar; la copia externa tiene las 12 ejecutadas sin errores, con entrenamiento desactivado y mensajes de resultados pendientes. Ambos tienen 24 celdas en total. |
| Informe | **Parcial** | Fuentes metodológicas, seis referencias citadas y texto de uso responsable disponibles. No hay PDF ni activos científicos; autores, resultados y discusión empírica siguen pendientes. No se puede verificar extensión ni composición visual. |
| Entrega al profesor | **Falta** | No hay PDF final. `README.md` está modificado y `parcial/` no está versionado en el checkout auditado. La URL de origin no demuestra que ese código esté publicado ni que se haya entregado en la plataforma. |

**Estados utilizados:** cumple = evidencia directamente comprobada para el requisito; parcial = existe una parte y se identifica la brecha; falta = artefacto o acción ausente; no verificado = no hay comprobación suficiente. Una constancia anterior se identifica como tal y no se convierte en ejecución actual.

## 2. Fuentes y alcance de esta evaluación

Se leyó mediante extracción íntegra de texto el [enunciado de siete páginas](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/examen-parcial.pdf) y el [ejemplo de once páginas](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/ejemplo.pdf). Se inspeccionaron configuración, Compose, fuentes, recibos, manifiestos, CSV y ambas copias del notebook. Las instrucciones AGENTS.md proporcionadas en el chat se aplicaron; no se encontró otro AGENTS.md aplicable en los ancestros comprobados ni debajo de parcial/. Se utilizaron las skills PDF, YOLO datasets/training/inference y diseño de documentos. No se crearon subagentes ni se enviaron mensajes a otros chats.

| Ubicación | Ruta absoluta |
|---|---|
| Proyecto | `D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales` |
| Imágenes reducidas oficiales | `E:/Kaggle/eccv_18_all_images_sm/eccv_18_all_images_sm` |
| Anotaciones oficiales | `E:/Kaggle/CCT20/eccv_18_annotation_files` |
| Resultados auditados | `E:/Kaggle/exam-output-wildlife` |

### Comprobado directamente en esta auditoría

- Los cinco JSON oficiales conservan los SHA-256 del manifiesto. Se reconstruyeron independientemente elegibilidad, exclusiones y cajas normalizadas a partir de ellos: **cero diferencias** con los 47.254 registros elegibles y coincidencia de las 10.610 exclusiones.
- Se recalculó la identidad del manifiesto y se comparó con el recibo previo y con la muestra de revisión. Se comprobaron conjuntos de imágenes, secuencias y cámaras; todos los solapamientos de imágenes y secuencias entre particiones elegibles son cero.
- Las 57.864 entradas de la caché existen y las dimensiones/hashes registrados coinciden con todos los registros del manifiesto. Además se volvieron a calcular los hashes de las **90 fotografías** de la muestra de etiquetas: 90 coincidencias. **No se volvieron a decodificar ni a calcular los hashes de las 57.864 fotografías completas**; esa operación sigue respaldada por la constancia anterior y la caché, no por una nueva ejecución integral.
- Se recalcularon hashes de pesos preentrenados, archivo comprimido de anotaciones, checkpoint del smoke y los 14 archivos fuente registrados previamente. Se analizaron contenido, ejecución y errores de ambos notebooks.
- En un contenedor de la imagen existente, sin GPU ni red y con proyecto/resultados montados en solo lectura, se recalcularon métricas COCO del smoke desde cuatro lotes guardados: 31 imágenes y 9.300 predicciones. El diccionario completo coincide con el guardado. No se hizo inferencia ni entrenamiento nuevo.
- Se leyó el código instalado de Ultralytics 8.4.21: `Metric.fitness()` pondera exclusivamente mAP50:95 con `[0,0,0,1]`; `DetMetrics.fitness` delega allí y el entrenador guarda `best.pt` por ese fitness de validación. La imagen local coincide con el identificador de la constancia anterior.
- Se comprobó ausencia de todos los artefactos científicos finales enumerados en la matriz. No se ejecutó la compilación de un informe inexistente ni se comprobó el contenido del repositorio remoto.

### Constancias anteriores, comprobadas como archivos

[validation-wildlife.json](E:/Kaggle/exam-output-wildlife/receipts/validation-wildlife.json), registrado el 3 de octubre de 2026 a las 19:08:44 UTC, declara 26 pruebas aprobadas, Ruff aprobado, CUDA/AMP, reanudación y notebook ejecutado. El [registro de pruebas](E:/Kaggle/exam-output-wildlife/receipts/pytest-wildlife.txt) contiene `26 passed`; no se repitieron esas pruebas. [doctor_gpu.json](E:/Kaggle/exam-output-wildlife/receipts/doctor_gpu.json) declara RTX 5090, gradientes finitos y NMS esperado; [amp.json](E:/Kaggle/exam-output-wildlife/receipts/amp.json) declara comparación FP32/AMP aprobada. [passed.json](E:/Kaggle/exam-output-wildlife/resume-check/passed.json) declara tres épocas de una prueba sintética, inicio recuperado en época 1 y 183 estados de optimizador; no demuestra equivalencia bit a bit ni entrenamiento científico de CCT20.

No existe `E:/Kaggle/exam-output-wildlife/receipts/installed_packages.json`; sí existe el inventario [environment.txt](E:/Kaggle/exam-output-wildlife/receipts/environment.txt). Se comprobaron versiones actuales en la imagen existente: torch 2.10.0+cu128, torchvision 0.25.0+cu128, ultralytics 8.4.21, numpy 2.2.6, pycocotools 2.0.11 y nbconvert 7.16.6. No se reconstruyó la imagen ni se reprodujo toda la instalación.

## 3. Qué exige realmente el profesor

Las páginas 1 y 7 exigen **un PDF NeurIPS de 6–10 páginas por pregunta**, motivación, trabajos relacionados con al menos cinco referencias, metodología, experimentos, resultados y discusión crítica; grupo de cinco integrantes y código reproducible por GitHub con un único comando de extremo a extremo documentado en README y `requirements.txt` o `environment.yml`. La plataforma del curso debe recibir el PDF y el enlace GitHub. **Entrega y presentación en clase: 30 de octubre de 2026.** La página 7 establece penalización de 10 % por día de retraso.

Las páginas 5–7 añaden los seis entregables de fauna de la matriz siguiente. No exigen exactamente YOLOv8s, 50 épocas, doce mapas, 90 revisiones, 30 objetos por nivel de oclusión, correos de autores, Docker, un notebook o una bandera `finalized.json`. Esos son decisiones o controles del proyecto. No se identificó requisito de envío al leaderboard de Kaggle; el uso de Kaggle indicado en página 2 corresponde a datos MRI y excluye expresamente ese envío.

El ejemplo sirve como referencia de organización: tiene **11 páginas y cuatro autores**, pero no sustituye los límites ni los cinco integrantes del examen. Tampoco debe copiarse su evidencia: su abstract afirma mejora por CLAHE, mientras páginas 8–9 muestran degradación, y página 10 cambia la función de pérdida de la comparación. Es una advertencia editorial para mantener consistencia entre abstract, metodología y resultados propios.

## 4. Matriz requisito → evidencia → estado → pendiente

| Requisito y origen | Evidencia exacta | Estado | Pendiente de cierre |
|---|---|---|---|
| Entrenar y evaluar YOLO, pp. 5–6 | [pipeline.py, run](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/src/neural_exam/wildlife/pipeline.py:344); solo existe entrenamiento en `E:/Kaggle/exam-output-wildlife/smoke/wildlife/yolov8s/` | Parcial | Ejecutar el experimento completo y documentar finalización/parada temprana real. |
| Resolución y augmentations documentados, p. 6, punto 1 | [default.toml](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/configs/default.toml:36), [main.tex](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/reports/wildlife/main.tex), `E:/Kaggle/exam-output-wildlife/smoke/wildlife/yolov8s/args.yaml` | Parcial | Registrar y explicar los argumentos efectivos del entrenamiento científico, incluidos aumentos heredados. |
| Manejo explícito de imágenes vacías como fondo, p. 6, punto 1 | [data.py, conversión](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/src/neural_exam/wildlife/data.py:148); [split_audit.csv](E:/Kaggle/exam-output-wildlife/wildlife/split_audit.csv): **train backgrounds=0** | Parcial | Cerrar la estrategia de negativos antes de entrenar; conservar las particiones reservadas. No basta con código compatible con empty. |
| mAP50, mAP50:95 y AP por especie, p. 6, punto 2 | [metrics.py](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/src/neural_exam/wildlife/metrics.py:13); faltan `E:/Kaggle/exam-output-wildlife/wildlife/{cis_test,trans_test}/metrics.json` y `per_species.csv` | Falta | Evaluar todos los registros elegibles de ambos tests con el mismo checkpoint; publicar soportes y AP indefinidas. |
| Efecto del desbalance, p. 6, punto 2 | [class_balance.csv](E:/Kaggle/exam-output-wildlife/wildlife/class_balance.csv); texto metodológico en main.tex | Parcial | Relacionar AP real con soporte de entrenamiento y error de especies raras. No hay estrategia de mitigación evaluada; el examen de Q3 pide analizar el efecto, no una ablación obligatoria. |
| Partición oficial y cámaras vistas/nuevas, p. 6, punto 3 | [manifest.json](E:/Kaggle/exam-output-wildlife/wildlife/manifest.json), [split_audit.csv](E:/Kaggle/exam-output-wildlife/wildlife/split_audit.csv), cinco JSON oficiales con hash concordante | Cumple en preparación | Mantener cámaras trans fuera de entrenamiento y demostrar esa separación en el experimento final. La depuración de cis_val debe declararse. |
| Brecha de rendimiento entre cámaras y discusión, p. 6, punto 3 | Código en pipeline.py; falta `E:/Kaggle/exam-output-wildlife/wildlife/generalization.json` | Falta | Reportar ambas mAP y brecha cis−trans, considerando categorías con soporte diferente. |
| Evidencia sobre fondo, p. 6, punto 3 | [detection_cam](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/src/neural_exam/wildlife/pipeline.py:133); `E:/Kaggle/exam-output-wildlife/smoke/wildlife/cam_diagnostic.json` es dog en val y scientific_run=false | Parcial | Mapas del checkpoint final en ambos tests y discusión animal/fondo; el smoke no demuestra memorización. |
| Oclusión baja/media/alta y caída de desempeño, p. 6, punto 4 | [review.py](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/src/neural_exam/wildlife/review.py:180), [robustness_metrics](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/src/neural_exam/wildlife/pipeline.py:84); falta `E:/Kaggle/exam-output-wildlife/wildlife/occlusion_review.csv` | Falta | Anotar visualmente, calcular métricas y diferencias entre niveles; justificar cobertura del test frente a la muestra prevista. |
| Tamaño relativo y caída de desempeño, p. 6, punto 4 | Intervalos <1 %, 1–5 %, >5 % y COCO en espacio virtual en metrics.py; falta `E:/Kaggle/exam-output-wildlife/wildlife/robustness.json` | Parcial | Métricas y soportes de los tres grupos en cis_test y trans_test; discusión de la caída y posibles confusores. |
| Módulo ecológico propio, pp. 5–6 | [monitoring.py](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/src/neural_exam/wildlife/monitoring.py:7) | Cumple en implementación | Validar su salida con las predicciones finales y ejemplos comprobados; sus conteos no estiman población. |
| Demostración de secuencia de cámara nueva y tabla especie/periodo, p. 6, punto 5 | [monitoring_demo](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/src/neural_exam/wildlife/pipeline.py:279); faltan `E:/Kaggle/exam-output-wildlife/wildlife/activity.json`, `activity_species_days.csv`, `activity_camera_days.csv`, `activity_sequence.csv` | Falta | Inferencia final, ráfaga completa y cotejo de agregados. Hay metadatos suficientes para la cámara 7, pero no una demostración ejecutada. |
| Media página de uso responsable, pp. 6–7, punto 6 | [discussion.tex](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/reports/wildlife/discussion.tex) | Parcial | Texto cubre FN/FP, validación local y responsabilidad humana; ajustar a aproximadamente media página del PDF y conectarlo con resultados reales. |
| Informe NeurIPS 6–10 páginas y secciones, pp. 1 y 7 | [main.tex](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/reports/wildlife/main.tex), [preamble.tex](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/reports/preamble.tex); no existe `E:/Kaggle/exam-output-wildlife/reports/` | Parcial | Completar narrativa y activos, compilar PDF, contar todas sus páginas y revisar composición visual. |
| Mínimo cinco referencias, p. 1 | main.tex cita `terra,yolo,yolov8,coco,gradcam,adamw`; [references.bib](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/reports/references.bib) contiene las seis claves, 14 entradas globales | Cumple en fuente | Confirmar al menos cinco referencias pertinentes renderizadas, citas resueltas y metadatos correctos en el PDF final; no contar las 14 entradas compartidas como referencias de Q3. |
| Cinco integrantes, pp. 1 y 7 | preamble.tex dice «completar nombres y correos» | Falta | Incorporar cinco nombres reales y los datos editoriales del grupo; no inventarlos. El requisito de correos proviene del compilador del proyecto. |
| Código reproducible y comando único en README, p. 7 | [README](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/README.md), [WILDLIFE_RUN.md](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/WILDLIFE_RUN.md), [requirements.txt](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/requirements.txt), [Compose](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/compose.wildlife.yaml) | Parcial | El comando existe, pero no está demostrado para el experimento final. Llevar el comando aislado de Q3 al README principal de la entrega, precisar prerrequisitos y entradas de revisión. |
| GitHub, PDF y plataforma; presentación 30/10/2026, p. 7 | origin configurado; `git status --short`: ` M README.md`, `?? parcial/` y otros archivos no versionados | Falta / remoto no verificado | Versionar/publicar el alcance final cuando se autorice esa etapa, verificar URL y commit desde GitHub, entregar PDF+enlace en plataforma y preparar presentación del grupo. |
| Notebook completo, control del equipo | [fuente](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/notebooks/03_wildlife.ipynb), [copia ejecutada](E:/Kaggle/exam-output-wildlife/notebooks/03_wildlife.ipynb) | Parcial | Reejecutar para incorporar resultados científicos y verificar su procedencia; un notebook sin errores no acredita entrenamiento. |

## 5. Datos, decisiones científicas y brechas

### Subconjunto oficial y mapeo

La [documentación primaria de LILA](https://lila.science/datasets/caltech-camera-traps) diferencia la colección completa y CCT20 ECCV, proporciona las imágenes reducidas y metadatos cis/trans, y relaciona eccv_train con ubicaciones train/cis y eccv_val con ubicaciones trans. El informe puede defender este subconjunto oficial sin afirmar cobertura de las 21 categorías de la colección completa descrita en el examen. Las 14 categorías se verificaron en los JSON locales; la diferencia debe quedar explícita en metodología y limitaciones. No se exige aumentar artificialmente a 21 salidas.

Mapeo original→YOLO: `1→0 opossum; 3→1 raccoon; 5→2 squirrel; 6→3 bobcat; 7→4 skunk; 8→5 dog; 9→6 coyote; 10→7 rabbit; 11→8 bird; 16→9 cat; 21→10 badger; 34→11 deer; 51→12 fox; 99→13 rodent`. `empty=30` es fondo, `car=33` se excluye del espacio animal. Bird y rodent son categorías agregadas, no especies biológicas únicas.

| Partición | Fotos oficiales | Elegibles | Vacías elegibles | Cámaras | Secuencias elegibles |
|---|---:|---:|---:|---:|---:|
| train | 13.553 | 11.399 | **0** | 10 | 4.733 |
| cis_val (clave interna val) | 3.484 | 2.543 | 995 | 10 | 980 |
| cis_test | 15.827 | 13.073 | 1.180 | 10 | 5.378 |
| trans_val | 1.725 | 1.592 | 61 | 1 | 589 |
| trans_test | 23.275 | 18.647 | 1.778 | 9 | 7.438 |

Train/cis comparten las cámaras `33,38,43,51,61,88,90,108,115,120`; trans_val usa `125` y trans_test `0,7,28,40,46,78,100,105,130`, disjuntas de train. El código selecciona checkpoint con cis_val depurado, no con trans_test. Trans_val se conserva auditado y no se usa para entrenar ni seleccionar actualmente. Se retiraron 270 fotos de cis_val por secuencias compartidas con train; son una depuración añadida, no una nueva partición aleatoria. Las 668 exclusiones de train sin animal/empty corresponden a car, verificado en las anotaciones.

Exclusiones adicionales: cajas animales incompletas `train=1454, val=629, cis_test=1951, trans_val=131, trans_test=3464`; cajas inválidas `32,1,12,2,51`, respectivamente. Imágenes sin animal ni empty explícito `668,41,791,0,1113`. La exclusión completa de imágenes con animales sin caja evita convertir animales en falsos fondos, pero sesga la población evaluada: no equivale a medir todo CCT20.

Las cajas oficiales están en píxeles de las dimensiones originales. La reconversión usa esas dimensiones y el resultado se aplica normalizado a imágenes reducidas: 44.949 registros elegibles son 2048×1494→1024×747, 1.395 son 2048×1536→1024×768 y 910 son 2560×1920→1024×768. Se comprobó correspondencia con caché y cajas; no se cambiaron originales ni se sustituyeron denominadores por las dimensiones reducidas.

Train tiene 11.911 cajas: opossum 2.514, rabbit 2.266, fox **5**, badger **9** y deer **44**, entre las demás categorías registradas. Cis_val no tiene deer ni fox; trans_test no tiene deer. La mAP cis/trans puede promediar 14 y 13 categorías con soporte respectivamente. Además de la métrica propia de cada test, conviene informar la comparación sobre las categorías comunes para separar parcialmente el cambio de composición del cambio de cámara; es un análisis añadido, no un requisito expreso.

### Fondos: cierre necesario antes del entrenamiento

La representación técnica de empty mediante archivos de etiquetas vacíos es correcta, pero los JSON de train no contienen empty. **No se puede afirmar que el entrenamiento usa fotografías vacías.** Los fondos internos de imágenes positivas aportan regiones negativas, pero no sustituyen una cohorte de fotos completamente vacías del sensor.

Plan recomendado: incorporar una cohorte acotada de **empty reales de la colección completa**, pertenecientes únicamente a las diez cámaras de train, excluyendo por imagen y secuencia todos los archivos reservados en cis_val/cis_test/trans_val/trans_test. Usar las etiquetas oficiales de imagen y revisión visual, registrar procedencia, dimensiones, hashes y un manifiesto derivado; no editar las anotaciones oficiales ni mover fotos de validación/prueba. Si se implementa esta ampliación, hacerla en una salida/Compose nuevos y mantener intacta la auditoría actual. El código actual no ingiere esta cohorte adicional: adaptar esa entrada es una tarea futura explícita, no algo ejecutable ya con el comando actual.

La alternativa conserva el benchmark estricto sin negativos adicionales y declara su limitación; requiere menos cambios y mantiene comparabilidad exacta, pero deja el punto de imágenes vacías **parcial** bajo una lectura estricta del examen. Recortes sin animal obtenidos de train pueden servir como aumento complementario, pero no deben presentarse como fotografías empty reales. No se recomienda usar las 995 fotos vacías de cis_val ni vacíos de test para resolver el problema.

### Entrenamiento y procedencia

Configuración actual: YOLOv8s preentrenado, 640 px, batch 16, semilla 42, hasta 50 épocas, paciencia 10, AdamW, lr inicial 0,0003, weight decay 0,0001 y AMP. Mosaic=1, fliplr=0,5, flipud=0 y cierre de mosaic en las últimas diez épocas configuradas. Los argumentos efectivos del smoke también incluyen HSV, traslación y escala heredados: la documentación científica final debe describirlos según los args de su propia ejecución, sin atribuir indiscriminadamente al detector parámetros de otras tareas presentes en el YAML.

`run.json` debe indicar scientific_run=true; `training_complete.json` debe coincidir con el hash real de best.pt; resultados de ambos tests, cachés y evaluation_complete deben usar ese mismo checkpoint. El máximo de mAP50:95 de cis_val es compatible con el código instalado comprobado, pero aún no existe un checkpoint científico seleccionado. Una parada temprana válida no requiere completar exactamente 50 épocas; debe registrarse el historial y la razón de cierre. El manifiesto preparado ya contiene scientific_run=true por no ser smoke: **esa bandera en el manifiesto no demuestra que se haya entrenado**.

La firma de reanudación incluye configuración, datos, versiones y pesos preentrenados. **Los hashes del código se registran aparte y no forman parte de esa firma** en pipeline.py:371. No se debe asumir que cambiar código invalida automáticamente una ejecución o una caché. El plan exige congelar fuentes y comparar hashes antes de reanudar; si cambia lógica, usar una ejecución independiente o documentar explícitamente la regeneración permitida.

### Revisión de etiquetas y oclusión: origen de los controles

La muestra [label_review_sample.json](E:/Kaggle/exam-output-wildlife/wildlife/label_review_sample.json) pertenece al manifiesto correcto; el [CSV](E:/Kaggle/exam-output-wildlife/wildlife/label_review.csv) conserva exactamente sus 90 IDs, todos pending y sin evaluador. La [galería](E:/Kaggle/exam-output-wildlife/wildlife/label_review.html) y ocho hojas de contacto existen. La muestra proviene de cinco positivos por categoría disponible y veinte fondos de train/val; en estos datos los fondos proceden de val. Revisarlos no autoriza trasladarlos a train. El bloqueo previo al entrenamiento lo implementan review.py y pipeline.py, no el profesor; es defendible como control de calidad de la conversión. No se eliminó ni se marcaron revisiones ficticias.

Las anotaciones locales solo contienen campos `bbox,category_id,id,image_id`: no contienen niveles de oclusión. La implementación añade revisión humana de una muestra inicial de hasta 120 objetos de trans_test, extensible, con >75 % visible=baja, 25–75 %=media y <25 %=alta. Exige evaluador y evidencia; casos ambiguos o truncamientos quedan sin nivel. Los **30 objetos por nivel** provienen de default.toml/review.py/pipeline.py y son un mínimo operativo del equipo, no una exigencia del examen ni una garantía estadística. Tamaño no representa oclusión.

El examen pide particionar el conjunto de prueba por oclusión. La muestra de trans_test constituye una aproximación que debe declararse y justificarse; no acredita una partición exhaustiva. Para un cierre literal, estimar niveles en los objetos del test definido para robustez (actualmente trans_test), registrando casos ambiguos. Extender también a cis_test es un análisis añadido; el enunciado no exige expresamente repetir cada estrato en ambos tests. La alternativa de una cohorte muestreada, reproducible y estratificada reduce trabajo de revisión, pero debe mantener visible la cobertura parcial y no presentarse como anotación exhaustiva. Debe evitarse elegir objetos en función del error del modelo. Las regiones crowd ignoradas absorben detecciones de objetos fuera del grupo; son una decisión de evaluación conservadora que cambia la interpretación frente a AP exhaustiva. Informar soportes y diferencias de AP entre niveles, no solo una bandera reviewed.

### Interpretabilidad y demostración ecológica

El único mapa diagnóstico usa dog, imagen `586ae163-23d2-11e8-a6a3-ec086b02610b`, en cis_val de smoke, score 0,05387 y rango de mapa [0,1]. No prueba generalización ni atención final al fondo. La capa 9 del backbone y el score previo a NMS están fijados en configuración/código. Los doce casos finales, seis por test, son un control del proyecto; el examen exige evidencia y discusión sin fijar esa cantidad. Contrastar animal, vegetación/fondo, día/noche, especies raras/frecuentes y aciertos/errores. Los mapas no demuestran causalidad por sí solos.

El módulo ecológico cuenta detecciones, fotos con especie, categorías distintas y tasa de fotos sin detecciones por cámara/día, con confianza 0,25. No cuenta individuos únicos y «vacías» significa predicción de ausencia. Se comprobó que la selección determinista actual podría usar **cámara 7**, 1.853 fotogramas oficiales de trans_test con timestamps válidos, desde 2011-05-07 hasta 2012-05-18. La primera ráfaga completa es `6f06c47d-5567-11e8-b8ef-dca9047ef277`, tres fotogramas a las 11:39:22, 11:39:23 y 11:39:24 del 2011-05-08. IDs: `597515c6-23d2-11e8-a6a3-ec086b02610b`, `5968c329-23d2-11e8-a6a3-ec086b02610b`, `59ef8e67-23d2-11e8-a6a3-ec086b02610b`. Esto prueba viabilidad de metadatos, **no ejecución de la demostración**. No se infiere zona horaria ni ausencia biológica a partir de falta de detección.

## 6. Integridad de los entregables y constancias

| Activo | SHA-256 o comparación actual | Resultado |
|---|---|---|
| Manifiesto, identidad semántica | `c9daeee489eb5d20fbda77b744c34f142b8f8c93cd03ff4d73055ea8350ca94b` | Recalculada; coincide con constancia y muestra. |
| `E:/Kaggle/exam-output-wildlife/wildlife/manifest.json`, bytes | `f1a052568a12967b0aee1893c47039fa3bd5fc8785420d6d9411f77237e77593` | Hash actual; no existía hash de bytes previo en el recibo comparado. |
| `E:/Kaggle/exam-output-wildlife/cache/weights/yolov8s.pt` | `1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a` | Coincide con wildlife-weights.json. |
| `E:/Kaggle/exam-output-wildlife/cache/weights/yolo26n.pt` | `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` | Coincide; auxiliar de AMP, no detector científico. |
| `E:/Kaggle/CCT20/archives/eccv_18_annotations.tar.gz` | `e31d0162d411fb031ba4741758a54fa15cc7257df6f344581f5fda612b2cc974` | Coincide con cct-annotations.json. |
| Cinco JSON oficiales | 5/5 SHA-256 coinciden con annotation_sha256 del manifiesto | Comprobado sobre sus bytes actuales. |
| `E:/Kaggle/exam-output-wildlife/smoke/wildlife/yolov8s/weights/best.pt` | `6df1838ae053e5bcf57855004aa53c9fedc0adfb98b5d7641f33ad06dbfba74f` | Coincide con training_complete del smoke. No es un modelo final. |
| Notebook fuente | `72a06a9ab5bcdffdcd8183692bf2da680808de150b350940c7794238576c8be6` | Coincide con constancia; 0/12 celdas de código ejecutadas. |
| Notebook externo | `e812035923cc628920fcb32fa481aa53e54144c4721fe2198a5ac734dcf60864` | Hash actual; sin hash previo comparable. Fuentes de todas sus celdas idénticas al notebook fuente. |
| 14 archivos del recibo source_sha256 | **13/14 coinciden** | Excepción: scripts/compile_reports.py. |
| Compilador actual | `68ab1acf9f147d964c72b68587cf6d9ba886398be78f454ca33d56c0fa71bf32` | El recibo esperaba `93a6cf263a9b1e215ea438e70740ae1188968850be7d9a92f730b09d00520f94`. Su validación previa no acredita este contenido actual. |
| Imagen local neural-exam-wildlife:0.2 | `sha256:1c680ce7639c7350c45edeaaa6b7e8844c40389fb4818d2e27c0961b64f28fe7` | Coincide con runtime_image del recibo. |
| PDF, checkpoint científico y métricas finales | Ausentes | No hay hashes finales que comparar. |

Los SHA de descargas son constancias locales, no checksums publicados del proveedor: sus recibos tienen provider_md5=null. El smoke tiene mAP50=0,0562904164 y mAP50:95=0,0466833875, 24 objetos de referencia y scientific_run=false; **estas cifras no pueden aparecer como respuesta cuantitativa del examen**. No se usó ningún test para ese recálculo.

## 7. Plan priorizado propio

No se ejecutó ninguno de los pasos futuros que escribe resultados. La cadena principal es **P0 → P1 → P2 → P3/P4 → P5 → P6 → P7**. P3 y P4 pueden organizarse sin un nuevo entrenamiento, pero la GPU debe usarse secuencialmente.

| Prioridad | Trabajo y dependencias | Criterio de cierre y verificación |
|---|---|---|
| **P0: diseño y datos** | Resolver fondos reales y alcance de la robustez antes de usar test. Registrar elección de CCT20 ECCV, 14 categorías, exclusiones, negativos, cohortes y reglas. Si se añaden negativos, adaptar entrada en una etapa posterior autorizada, con manifiesto derivado, salida y volúmenes nuevos. | Ningún ID/secuencia reservado en train; cámaras nuevas disjuntas; dimensiones/cajas correctas; negativos genuinos con revisión y hash. Muestras/identidades congeladas. Mantener intacta la salida auditada. |
| **P1: revisión de etiquetas y entorno** | Revisar las 90 fotos originales y resolver needs_changes; si hay nueva cohorte, preservar esa constancia y revisar además sus negativos bajo nueva identidad. Congelar código/configuración y registrar hashes. Revalidar compilador cambiado. | Todas las filas de la muestra efectiva reviewed con persona real; muestra completa y trazable. No retirar filas para pasar el gate. Entorno, pesos y fuentes compatibles; comprobaciones CPU de conversión, COCO, agregación y controles pasan en una salida temporal. |
| **P2: entrenamiento YOLOv8s** | Depende de P0/P1. Ejecutar solo fauna, sin trabajos GPU simultáneos, con configuración congelada. Usar el comando científico; --resume solo ante una interrupción real compatible. | run.json scientific_run=true, args.yaml, results.csv completo, curvas, razón de cierre y best.pt/last.pt; training_complete con hash real. best.pt procede del máximo de mAP50:95 en cis_val, nunca de test. Una parada temprana documentada es válida. |
| **P3: evaluación cis/trans** | Depende de P2. Inferencia exhaustiva sobre imágenes elegibles reservadas, confianza para AP 0,001, NMS 0,7 y máximo 300. Recalcular métricas desde predicciones en CPU sin repetir entrenamiento. | Ambos metrics.json, predictions.json y per_species.csv; mismo hash de checkpoint; soportes e indefinidas correctos. Generalization/experiments concuerdan. Cobertura de todas las imágenes y lotes, sin omisiones/duplicados. Brecha y AP de especies raras explicadas; análisis de categorías comunes si se usa. |
| **P4: robustez e interpretación** | Congelar protocolo de oclusión en P0; completar revisión y grupos antes de interpretar errores finales. Depende de P2/P3 para métricas/CAM. Revisar cada nivel y ampliar de forma reproducible, sin seleccionar por rendimiento. | Tamaño en ambos tests; oclusión con soportes, evidencia y mínimo interno de 30 por nivel. Para cierre literal, partición del test de robustez definido, con ambiguos registrados; si solo se revisa muestra, conservar estado parcial para cobertura. Caídas de AP calculadas y discutidas. Doce CAM efectivos del checkpoint final, rango finito y lectura visual animal/fondo. El gate interno permanece activo. |
| **P5: demostración ecológica** | Depende de P2; cámara 7 y ráfaga comprobadas como candidatas, volver a verificar selección si cambia el manifiesto. Agregar todos sus fotogramas oficiales y mostrar la secuencia completa. | activity.json y tres CSV, predicciones trazables al checkpoint, cámara ausente de train, secuencia íntegra y fechas válidas. Cotejo manual de unas filas contra predicciones: totales, fotos, categorías, tasa vacía y periodos. Conteos no presentados como individuos ni como registros biológicos confirmados. |
| **P6: notebook e informe** | Depende de P3/P4/P5. Ejecutar notebook para mostrar resultados guardados; completar abstract, resultados, brecha, desbalance, robustez, CAM, demostración, conclusión y uso responsable. Añadir cinco autores reales. Generar activos y compilar borrador. | Notebook 12 celdas ejecutadas sin errores y sin mensajes científicos pendientes; sus cifras y hashes coinciden con resultados. PDF NeurIPS 6–10 páginas, ≥5 referencias resueltas, sin placeholders; media página de uso responsable verificada visualmente. Tablas/figuras legibles y consistentes. Registrar finalized.json solo tras revisión real y compilar final. |
| **P7: publicación y entrega** | Depende de P6 y revisión del grupo. Consolidar README, entorno y archivos exactos, verificar código y resultados accesibles en el commit publicado. La publicación/entrega no forma parte de esta auditoría. | Repositorio remoto contiene los archivos de Q3; comando documentado ejecuta preparación→entrenamiento→evaluación→monitoreo con entradas de revisión especificadas. PDF final y enlace entregados en plataforma; comprobante de entrega; presentación preparada para el 30/10/2026. |

**Controles de cierre adicionales:** `evaluation_complete.json` puede existir con occlusion_status pendiente; su existencia sola no acredita que la pregunta esté completa. El generador de informes comprueba scientific_run y hashes de checkpoint declarados, pero la auditoría final debe comparar también el archivo real, cobertura de predicciones y recálculo. El compilador exige un marcador editorial, pero no inspecciona automáticamente la veracidad de autores ni toda la narrativa. No utilizar esos marcadores como sustitutos de evidencia.

## 8. Comandos concretos de comprobación y continuación

### A. Lectura de estado y hashes; no entrena ni escribe resultados

```powershell
$projectAuditRoot = 'D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales'
$outputAuditRoot = 'E:/Kaggle/exam-output-wildlife'
Set-Location -LiteralPath $projectAuditRoot
git status --short
git remote -v
$auditReceipt = Get-Content -Raw -LiteralPath "$outputAuditRoot/receipts/validation-wildlife.json" | ConvertFrom-Json
$auditReceipt.source_sha256.PSObject.Properties | ForEach-Object {
    $auditFile = Join-Path "$projectAuditRoot/parcial" $_.Name
    $auditActual = (Get-FileHash -LiteralPath $auditFile -Algorithm SHA256).Hash.ToLowerInvariant()
    [pscustomobject]@{File=$auditFile; Match=($auditActual -eq $_.Value); Actual=$auditActual; Expected=$_.Value}
}
Import-Csv -LiteralPath "$outputAuditRoot/wildlife/label_review.csv" | Group-Object status | Select-Object Name,Count
Get-Content -LiteralPath "$outputAuditRoot/wildlife/label_review_status.json"
Get-Content -LiteralPath "$outputAuditRoot/wildlife/split_audit.csv"
Get-Content -LiteralPath "$outputAuditRoot/wildlife/class_balance.csv"
Get-Content -LiteralPath "$outputAuditRoot/smoke/execution_summary.json"
Get-Content -LiteralPath "$outputAuditRoot/smoke/wildlife/cam_diagnostic.json"
```

### B. Anotaciones, identidad, notebooks y ausencia de resultados; solo lectura

```powershell
Set-Location -LiteralPath 'D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales'
@'
import csv, hashlib, json
from pathlib import Path
from itertools import combinations
O = Path('E:/Kaggle/exam-output-wildlife')
A = Path('E:/Kaggle/CCT20/eccv_18_annotation_files')
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
m = read(O/'wildlife/manifest.json')
identity = m.pop('identity')
assert hashlib.sha256(json.dumps(m, sort_keys=True, ensure_ascii=False).encode()).hexdigest() == identity
for split, name in [('train','train'),('val','cis_val'),('cis_test','cis_test'),('trans_val','trans_val'),('trans_test','trans_test')]:
    assert sha(A/f'{name}_annotations.json') == m['annotation_sha256'][split]
for a,b in combinations(m['records'],2):
    for field in ['image_id','sequence_id']:
        assert not {x[field] for x in m['records'][a]} & {x[field] for x in m['records'][b]}
for rel in ['receipts/wildlife-weights.json','receipts/cct-annotations.json']:
    for item in read(O/rel)['files']:
        assert sha(Path(item['file'])) == item['sha256']
smoke = O/'smoke/wildlife/yolov8s'
assert sha(smoke/'weights/best.pt') == read(smoke/'training_complete.json')['sha256']
for p in [Path('parcial/notebooks/03_wildlife.ipynb'), O/'notebooks/03_wildlife.ipynb']:
    cells = [x for x in read(p)['cells'] if x['cell_type']=='code']
    print(p, 'code',len(cells),'executed',sum(x['execution_count'] is not None for x in cells),
          'errors',sum(o['output_type']=='error' for x in cells for o in x.get('outputs',[])), 'sha256',sha(p))
for rel in ['wildlife/yolov8s/training_complete.json','wildlife/yolov8s/evaluation_complete.json',
            'wildlife/cis_test/metrics.json','wildlife/trans_test/metrics.json','wildlife/experiments.csv',
            'wildlife/occlusion_review.csv','wildlife/robustness.json','wildlife/cam_cases.json',
            'wildlife/activity.json','reports/wildlife.pdf']:
    print(rel, (O/rel).exists())
print('hashes y separación comprobados')
'@ | & 'C:/Users/USER/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -
```

### C. Recálculo CPU del smoke desde caché; comprobación ejecutada en esta auditoría

```powershell
@'
import json
from pathlib import Path
from neural_exam.common.io import fingerprint
from neural_exam.wildlife.metrics import coco_evaluate
W=Path('/audit-outputs/smoke/wildlife')
read=lambda p:json.loads(p.read_text())
m=read(W/'manifest.json')
ck=read(W/'yolov8s/training_complete.json')['sha256']
signature=fingerprint({'data':[(x['image_id'],x.get('source_sha256')) for x in m['records']['val']],
                       'imgsz':640,'nms_iou':0.7,'conf':0.001,'max_det':300,'schema':1})
predictions=[]; images=[]
for p in sorted((W/'prediction_cache'/ck/signature).glob('*.json')):
    batch=read(p); predictions.extend(batch['predictions']); images.extend(batch['images'])
assert images==[x['image_id'] for x in m['records']['val']]
score=coco_evaluate(m['records']['val'],predictions,m['names'])
assert score==read(W/'smoke_validation.json')['metrics']
print({'matches':True,'images':len(images),'predictions':len(predictions),'scientific_run':False})
'@ | docker run --rm -i --read-only --network none --cpus 2 --tmpfs /tmp -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/workspace/src -e MPLCONFIGDIR=/tmp/matplotlib --mount 'type=bind,source=D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial,target=/workspace,readonly' --mount 'type=bind,source=E:/Kaggle/exam-output-wildlife,target=/audit-outputs,readonly' --entrypoint python neural-exam-wildlife:0.2 -
```

### D. Pasos futuros; escriben resultados y NO se ejecutaron

Los comandos siguientes son los del flujo **actual**. Si P0 cambia la cohorte, actualizar antes la configuración de salida y el nombre de proyecto Compose para obtener volúmenes independientes; no reutilizar el manifiesto/revisión de otra identidad. Descargar pesos o construir imágenes no es necesario para esta auditoría. Las comprobaciones de recursos y entrenamiento se harán en la etapa de ejecución, secuencialmente.

```powershell
Set-Location -LiteralPath 'D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial'
$wildlifeCompose = @('-f','compose.wildlife.yaml','--env-file','.env.wildlife')
# P1: preparar/abrir galería y completar revisión humana real.
docker compose @wildlifeCompose run --rm runner review-labels
# P2/P3/P5: solo después de P0/P1; prepara, entrena, evalúa y agrega actividad.
docker compose @wildlifeCompose run --rm runner run --task wildlife
# Recuperación si se interrumpe; después reutiliza best.pt sin entrenar de nuevo.
docker compose @wildlifeCompose run --rm runner run --task wildlife --resume
# P4: revisar oclusión; ampliar solo si el soporte por nivel es insuficiente.
docker compose @wildlifeCompose run --rm runner review-occlusion
docker compose @wildlifeCompose run --rm runner review-occlusion --extend 120
# Tras guardar oclusión: regenerar evaluación/activos con el checkpoint terminado.
docker compose @wildlifeCompose run --rm runner run --task wildlife --resume
# P6: mantener RUN_PREPARE/RUN_TRAINING/GENERATE_REPORT_ASSETS=False para lectura.
docker compose @wildlifeCompose run --rm --entrypoint jupyter runner nbconvert --execute --to notebook notebooks/03_wildlife.ipynb --output 03_wildlife --output-dir /outputs/notebooks --ExecutePreprocessor.timeout=180
docker compose @wildlifeCompose run --rm runner report --task wildlife
docker compose @wildlifeCompose run --rm reports --task wildlife --draft
# Tras revisión real y finalized.json; no falsear el marcador.
docker compose @wildlifeCompose run --rm reports --task wildlife
```

Para verificar en la etapa futura la selección y la integridad: leer `E:/Kaggle/exam-output-wildlife/wildlife/yolov8s/results.csv`, identificar máximo de `metrics/mAP50-95(B)`, cotejar args/run, calcular Get-FileHash de best.pt y compararlo con training_complete/evaluation_complete/protocol de ambos metrics.json. Repetir el recálculo CPU de C para cada test usando sus registros y predicciones finales, sin inferencia ni entrenamiento nuevo. La bandera completion no sustituye cobertura de lotes y soportes.

El PDF futuro se comprueba con este comando, una vez que exista, y mediante inspección visual de todas las páginas:

```powershell
& 'C:/Users/USER/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -c "from pypdf import PdfReader; r=PdfReader('E:/Kaggle/exam-output-wildlife/reports/wildlife.pdf'); print('pages=',len(r.pages)); assert 6<=len(r.pages)<=10; t='\n'.join(p.extract_text() or '' for p in r.pages).casefold(); print('pending=', 'pendiente de evidencia' in t); assert 'completar nombres' not in t and 'pendiente de evidencia' not in t"
```

## 9. Archivos a presentar y tareas humanas

### Entrega exigida en la plataforma para esta pregunta

1. **PDF final:** `E:/Kaggle/exam-output-wildlife/reports/wildlife.pdf` (ruta prevista, actualmente inexistente), 6–10 páginas, NeurIPS, cinco autores reales y evidencia científica completa. Si P0 usa una salida nueva, fijar su ruta equivalente antes de empaquetar; no confundir wildlife_draft.pdf con la entrega final.
2. **Enlace del repositorio GitHub:** [miguelsff/maestria-ia-uni-ciclo-2-redes-neuronales](https://github.com/miguelsff/maestria-ia-uni-ciclo-2-redes-neuronales), URL configurada en origin, cuyo contenido de Q3 **no se verificó remoto**. Registrar el commit publicado y una ruta visible a la carpeta de la pregunta.
3. **Presentación del grupo en clase, 30 de octubre de 2026.** No se impone aquí formato de diapositivas que el enunciado no especifica.

### Archivos concretos que debe contener o referenciar la entrega reproducible

Dentro de `D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales`, conservar y publicar en la etapa de entrega:

- `README.md`; `parcial/README.md`; `parcial/WILDLIFE_RUN.md`; `parcial/requirements.txt`; `parcial/pyproject.toml`; `parcial/Dockerfile`; `parcial/compose.wildlife.yaml`; `parcial/.env.wildlife.example`; `parcial/configs/default.toml` y cualquier nueva configuración derivada que P0 introduzca.
- Paquete: `parcial/src/neural_exam/{__init__.py,__main__.py,cli.py,config.py,reporting.py}`; módulo fauna: `parcial/src/neural_exam/wildlife/{__init__.py,data.py,figures.py,metrics.py,monitoring.py,pipeline.py,review.py,runtime.py}`; utilidades compartidas: `parcial/src/neural_exam/common/{__init__.py,cam.py,download.py,figures.py,io.py,locking.py,metrics.py,training.py}`.
- Notebook con evidencia final: `parcial/notebooks/03_wildlife.ipynb` y/o la copia ejecutada `E:/Kaggle/exam-output-wildlife/notebooks/03_wildlife.ipynb` incorporada al paquete entregado, identificando claramente cuál contiene las salidas. Scripts: `parcial/scripts/{compile_reports.py,populate_wildlife_notebook.py,start_notebooks.py}`. Pruebas existentes: `parcial/tests/{test_wildlife_data.py,test_wildlife_workflow.py,test_utilities.py}`.
- Fuentes del informe: `parcial/reports/wildlife/{main.tex,discussion.tex,finalized.json}` (el marcador final aún no existe), `parcial/reports/{preamble.tex,references.bib}`. Tablas y figuras generadas deben acompañar la compilación del PDF; el estilo NeurIPS se obtiene con URL y constancia documentadas.

Los siguientes **resultados científicos**, bajo `E:/Kaggle/exam-output-wildlife`, deben incluirse en el paquete reproducible o referenciarse desde GitHub con acceso y hashes verificables: `wildlife/{manifest.json,split_audit.csv,class_map.csv,exclusions.csv,class_balance.csv,label_review.csv,label_review_sample.json,label_review_status.json,occlusion_review.csv,experiments.csv,generalization.json,robustness.json,cam_cases.json,activity.json,activity_predictions.json,activity_camera_days.csv,activity_species_days.csv,activity_sequence.csv}`; `wildlife/yolov8s/{run.json,args.yaml,results.csv,training_complete.json,evaluation_complete.json,weights/best.pt}`; `wildlife/{cis_test,trans_test}/{metrics.json,per_species.csv,predictions.json}`; `wildlife/{training_curves.png,class_balance.png,cam_cis_test_grid.png,cam_trans_test_grid.png}` y los doce PNG individualizados enumerados en cam_cases.json; `reports/wildlife/{tables.tex,figures.tex,status.json}` y las cuatro figuras exportadas `figures/figure_0.png` a `figure_3.png`. Estos nombres son salidas previstas de la implementación, no una afirmación de existencia actual. Conservar `weights/last.pt` para recuperación; su entrega no está exigida expresamente por el profesor.

Git ignora `*.pt`; los pesos finales necesitan un mecanismo de acceso documentado, por ejemplo un activo de release o almacenamiento del equipo con hash, si se comparten. No versionar datasets masivos ni `.env.wildlife` con su token. Los smoke y resume-check se incluyen únicamente como evidencia funcional identificada; no sustituyen resultados finales. Conservar recibos del entorno y sus hashes en el paquete, y registrar por separado los cambios del compilador actual.

### Intervención humana realmente necesaria

- Revisar visualmente etiquetas y negativos reales, resolver observaciones y firmar solo lo inspeccionado.
- Estimar oclusión con evidencia, distinguir truncamiento de oclusión y cerrar la cobertura acordada de robustez.
- Interpretar CAM y revisar una muestra del reporte ecológico; el software no puede certificar plausibilidad ni identificación biológica por sí solo.
- Proporcionar los cinco nombres reales y datos editoriales; revisar el informe final, su media página responsable y consistencia científica.
- Revisar/publicar la versión de entrega, enviar PDF+GitHub a la plataforma, conservar confirmación y presentar en clase. La auditoría no solicita ni inventa esa confirmación.

## 10. Criterio final de resolución

La pregunta estará resuelta cuando exista un entrenamiento científico trazable seleccionado por cis_val; métricas completas cis/trans, desbalance, tamaño y oclusión con partición del test de robustez definido; evidencia final de fondo; demostración real de cámara nueva; notebook coherente con esas salidas; PDF NeurIPS de 6–10 páginas sin pendientes, cinco autores y al menos cinco referencias; código accesible por GitHub y entrega confirmada en la plataforma. Si la oclusión se limita a una muestra o no se incorporan fotografías empty a train, mantener explícito el cumplimiento parcial correspondiente. **Hoy hay una base funcional sólida y una auditoría de preparación consistente, pero ninguno de esos resultados finales debe darse por realizado.**
