# Pregunta 1: respuesta científica completa; entrega aún pendiente

**Evaluación del 4 de octubre de 2026, America/Lima.** La pregunta de MRI está implementada y contestada, y existen ocho entrenamientos científicos completos. Esta auditoría comprobó en CPU las métricas de 464 predicciones, los checkpoints de las ocho variantes y la integridad de 600 cachés. El informe tiene ocho páginas y siete referencias. Los cinco nombres y la afiliación ya fueron proporcionados por el usuario. Todavía no corresponde declarar la entrega terminada: falta incorporarlos al PDF, cerrar su revisión, resolver la reproducción desde un clon limpio, comprobar la publicación del código y efectuar la entrega/presentación del curso.

La evaluación de los mismos 58 expertos se repitió después de corregir una partición defectuosa. El informe lo divulga; los resultados constituyen una evaluación local sobre una cohorte experta reutilizada, **no una validación externa ciega**. La discusión anatómica solicitada está contestada y reconoce mapas poco selectivos. Una firma de radiólogo no es un requisito del examen.

## Autoría proporcionada por el usuario

**Afiliación:** UNI · Unidad de Posgrado — Facultad de Ingeniería Industrial y de Sistemas.

**Integrantes**, conservando los nombres y el orden proporcionados:

1. Chilón Cruzado Josué.
2. Concepción Tiza Miguel Ángel.
3. Neyra Dianderas Enzo Israel.
4. Tello Guerra Omar Salvador.
5. Alexander Vladimir Paco Cornejo.

Fuente: mensaje del usuario en este chat, posterior a la primera versión de esta evaluación. La identificación de los cinco integrantes deja de ser un dato pendiente. Esta actualización registra la información en la auditoría; su incorporación a las fuentes del informe y al PDF corresponde al cierre editorial posterior. No se proporcionaron correos y no se inventan; el enunciado no los exige expresamente. Proporcionar los nombres no constituye una constancia de revisión editorial o anatómica.

## 1. Alcance y veredictos separados

Se leyeron las siete páginas del enunciado y las once del ejemplo, el código MRI y de entrenamiento/evaluación compartido, los documentos de ejecución y los artefactos vigentes. Se inspeccionaron las ocho imágenes de páginas ya existentes. No se entrenó, no se generaron imágenes, no se recompiló ni se modificaron código, datos, notebooks o informes científicos. Las comprobaciones en contenedores utilizaron CPU, red deshabilitada y montajes de solo lectura. El único archivo creado por esta auditoría es este Markdown.

| Dimensión | Veredicto | Evidencia y límite |
|---|---|---|
| Implementación y respuesta | **Cumple** los siete entregables específicos de la p. 3 | Extractor de doce etiquetas, CNN propia, ResNet-50 ImageNet, matriz de ablación, métricas, prevalencia, mitigación, tres Grad-CAM y discusión de sesgo. Cobertura multilingüe implementada no equivale a eficacia demostrada en cada idioma. |
| Ejecución científica | **Cumple** para la matriz vigente | Ocho `execution.json` científicos, historiales completos, `complete.json`, `best.pt`, `last.pt` y 58 predicciones por variante. Se verificó parada temprana, no solo existencia de archivos. |
| Validación | **Cumple** como comprobación local del trabajo; alcance limitado | Métricas recalculadas exactamente, checkpoints elegidos por pérdida de validación débil y cachés verificadas. No se repitió inferencia desde los pesos ni la decodificación de todos los DICOM. No acredita generalización externa, independencia de personas ni localización clínica. |
| Notebook | **Cumple** como cuaderno ejecutado de consulta | Ocho celdas con contadores 1-8, cero errores guardados, tres copias idénticas. `RUN_TRAINING=False`; no es evidencia de entrenamiento dentro del notebook. El examen no exige expresamente un `.ipynb`. |
| Informe | **Parcial** | PDF NeurIPS de ocho páginas, siete referencias y contenido científico sustancial. Los cinco nombres y la afiliación ya están registrados en esta evaluación; queda sustituir la autoría placeholder del PDF y cerrar su revisión editorial. |
| Entrega al profesor | **Falta** cerrar y acreditar | PDF final inexistente en la ruta prevista; no hay evidencia de publicación de estos cambios ni de envío a la plataforma/presentación. Además, la reproducción desde un clon limpio tiene un problema comprobado de metadatos. |

Estados de la matriz: **cumple** = evidencia suficiente dentro del alcance; **parcial** = existe pero requiere cierre; **falta** = pendiente concreto; **no verificado** = esta auditoría no tiene evidencia suficiente para concluir. Un estado de ejecución completo no garantiza calidad clínica ni entrega administrativa.

## 2. Qué exige el profesor y qué decidió el equipo

Fuente normativa: [examen-parcial.pdf](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/examen-parcial.pdf), pp. 1-3 y 7.

| Origen | Requisito o decisión |
|---|---|
| Examen, p. 1 | Grupo de cinco estudiantes. **Un informe de 6-10 páginas por pregunta**, con motivación, trabajos relacionados, al menos cinco referencias, metodología, experimentos, resultados y discusión crítica. Código reproducible por GitHub y componentes principales propios. |
| Examen, pp. 2-3 | Doce hallazgos; uso del conjunto RSNA. Escala reducida recomendada de 300-600 estudios, una o dos secuencias/planos y 2D/2.5D. Reservar aproximadamente 58 expertos como validación limpia. La reducción de escala es una recomendación, no una obligación de procesar el dataset completo. |
| Examen, p. 2 | Cada estudiante debe tener acceso autorizado a Kaggle y aceptar las reglas; el grupo debe revisar las condiciones de uso fuera de la competencia. **No exige envío al leaderboard**. |
| Examen, p. 3 | Los siete entregables específicos recogidos en la matriz siguiente. Exige discutir plausibilidad anatómica, no obtener mapas necesariamente correctos ni una firma profesional. |
| Examen, p. 7 | PDF en formato NeurIPS y enlace GitHub por la plataforma del curso. Un único comando de extremo a extremo documentado en README, con `requirements.txt` o `environment.yml`. **Entrega y presentación en clase: 30 de octubre de 2026**. Penalización indicada: 10 % por día de retraso. |
| Equipo | 600 estudios, semilla 42, ocho grupos de tres cortes, CNN/ResNet-50, BCE/focal, ponderación/filtro por etiqueta, umbral 0,5, validación débil para seleccionar checkpoint y expertos para evaluar. Son decisiones científicas compatibles con el examen. |
| Implementación | Hashes, cachés, límites Docker, pruebas de reanudación, revisión editorial registrada en `finalized.json`, correos de autores y estado `radiologist_review`. Son controles añadidos. Los correos y la firma de radiólogo no están expresamente exigidos por el enunciado. |

El [ejemplo.pdf](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/ejemplo.pdf) tiene once páginas y cuatro autores (p. 1); sirve como referencia de estructura. No autoriza once páginas ni cuatro integrantes para este examen, ni permite trasladar sus resultados o afirmaciones al proyecto MRI.

## 3. Matriz requisito → evidencia → estado → pendiente

| Requisito y página del examen | Evidencia exacta vigente | Estado | Pendiente o límite |
|---|---|---|---|
| Extractor multilingüe de doce etiquetas, p. 3, punto 1 | [labels.py:324](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/src/neural_exam/mri/labels.py:324), lista `LABELS`, léxicos, negación e incertidumbre; [manifest.json](E:/Kaggle/exam-output-mri/stratified-v2/mri/manifest.json), `records[*].weak`; PDF pp. 2 y 4 | Cumple | Se reprodujeron las 600 extracciones desde el CSV original. El léxico de trece lenguas no demuestra cobertura clínica equivalente; escrituras no son idiomas. |
| Comparación con expertos y acuerdo/ruido, p. 3, punto 1 | [label_agreement.json](E:/Kaggle/exam-output-mri/stratified-v2/mri/label_agreement.json), 24 filas de `per_label`; [pipeline.py:22](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/src/neural_exam/mri/pipeline.py:22); PDF p. 4, tabla 4 | Cumple | Acuerdo y cobertura recalculados sobre los 58 expertos. Son medidas diagnósticas condicionadas a observación, no ruido poblacional ni sensibilidad clínica externa. |
| CNN propia y backbone ImageNet con doce salidas, p. 3, punto 2 | [model.py:10](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/src/neural_exam/mri/model.py:10); [mri_assets.json](E:/Kaggle/exam-output-mri/stratified-v2/receipts/mri_assets.json); PDF p. 2, sección 4 | Cumple | CNN con cuatro bloques y cabeza propia; ResNet-50 cargado desde pesos ImageNet V2 y ajustado. Doce logits con sigmoid independiente en evaluación. Hash del peso local comprobado. |
| Entrenar ambas arquitecturas, p. 3, punto 2 | [experiments.csv](E:/Kaggle/exam-output-mri/stratified-v2/mri/experiments.csv); en cada uno de los ocho directorios de la sección 4: `history.csv`, `complete.json`, `execution.json`, `best.pt`, `last.pt` | Cumple | Todos finalizaron por paciencia 10 dentro de un máximo de 50 épocas. `run.json.status` conserva `running`: metadato desactualizado que no invalida las constancias de cierre verificadas. |
| AUC macro de doce etiquetas y desglose experto, p. 3, punto 3 | [metrics_all_labels.csv](E:/Kaggle/exam-output-mri/stratified-v2/mri/metrics_all_labels.csv), 96 filas; `predictions.csv` y `metrics.json` en los ocho directorios; PDF pp. 3 y 5 | Cumple | Cada etiqueta tiene ambas clases en los 58 expertos. AUC macro completo recalculado; no se sustituyó por un promedio parcial. Reutilización de expertos declarada. |
| Precisión, recall, F1 y umbral justificado, p. 3, punto 3 | [mri.toml](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/configs/mri.toml), `threshold=0.5`; `metrics.json.threshold`; PDF p. 3, protocolo y tabla 2; CSV combinado de 96 filas | Cumple | Umbral fijo explicado como baseline simétrico; no optimizado en expertos. El PDF explica `zero_division=0` y ausencia de calibración. |
| Ablación de al menos dos factores, p. 3, punto 4 | [pipeline.py:60](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/src/neural_exam/mri/pipeline.py:60); [experiments.csv](E:/Kaggle/exam-output-mri/stratified-v2/mri/experiments.csv); PDF pp. 3-4 y curvas pp. 6-7 | Cumple | Se comparan pérdida BCE/focal y tratamiento de ruido ponderación/filtro; además, arquitectura. Una semilla y 58 expertos impiden establecer superioridad general. No hace falta añadir otra ablación de plano. |
| Tres Grad-CAM y plausibilidad anatómica, p. 3, punto 5 | [cam_cases.json](E:/Kaggle/exam-output-mri/stratified-v2/mri/scratch_bce_weighted/cam_cases.json); `cam_0.npz`, `cam_2.npz`, `cam_7.npz` y PNG en ese directorio; [cam_grid.png](E:/Kaggle/exam-output-mri/stratified-v2/reports/mri/figures/cam_grid.png); PDF pp. 4-5 y 8 | Cumple | LCA, menisco medial y derrame. La discusión identifica activación difusa/periférica y fondo, sin afirmar localización demostrada. Revisión especializada adicional no es requisito expreso del curso. |
| Prevalencia de doce hallazgos, p. 3, punto 6 | [prevalence.csv](E:/Kaggle/exam-output-mri/stratified-v2/mri/prevalence.csv), 36 filas por partición/etiqueta; [class_balance.csv](E:/Kaggle/exam-output-mri/stratified-v2/mri/class_balance.csv); PDF p. 3, tabla 3 y p. 4 | Cumple | Conteos y denominadores permiten calcular proporciones; se incluyen ejemplos porcentuales de OA. Mejorar la tabla final con porcentajes explícitos y denominador observado sería útil. No son prevalencias poblacionales. |
| Mitigación implementada y evaluada, p. 3, punto 6 | [loss.py](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/src/neural_exam/mri/loss.py); [pipeline.py:47](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/src/neural_exam/mri/pipeline.py:47), alpha de entrenamiento tras filtro; PDF p. 4 y métricas por variante | Cumple | Focal con balance por etiqueta frente a BCE. Se cuantifica mejora de recall/F1 y costo de precisión; no se exige una mejora uniforme. |
| Media página de sesgo geográfico/lingüístico, p. 3, punto 7 | [discussion.tex](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/reports/mri/discussion.tex), 288 palabras aproximadamente; PDF pp. 5-6, sección 7; tabla 5, p. 4 | Cumple | La extensión acumulada es aproximadamente media página o más, aunque una figura divide el texto. Discute léxico, abstenciones, adquisición, selección y consecuencias de despliegue. No hay evidencia de centros/países fiable para una evaluación geográfica estratificada. |
| Separación por estudio y reserva experta recomendadas, pp. 2-3 | [manifest.json](E:/Kaggle/exam-output-mri/stratified-v2/mri/manifest.json); [data.py:41](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/src/neural_exam/mri/data.py:41); [mri_data.json](E:/Kaggle/exam-output-mri/stratified-v2/audit/mri_data.json); PDF pp. 1-2 | Cumple | Reconstrucción determinista: 434/108/58, 600 IDs únicos. Sin PatientID transversal fiable; no declarar independencia de personas reales ni prueba externa. |
| Motivación, relacionados, método, experimentos, resultados y límites; ≥5 referencias, p. 1 | [mri_draft.pdf](E:/Kaggle/exam-output-mri/stratified-v2/reports/mri_draft.pdf), secciones 1-8 y referencias p. 8; [main.bbl](E:/Kaggle/exam-output-mri/stratified-v2/reports/build/mri/main.bbl), siete entradas | Cumple | El mínimo bibliográfico se supera; cinco fuentes aparecen en motivación/relacionados. La referencia sobre skills es procedimental y no prueba validez experimental. |
| PDF NeurIPS de 6-10 páginas por pregunta, pp. 1 y 7 | [mri_draft.pdf](E:/Kaggle/exam-output-mri/stratified-v2/reports/mri_draft.pdf), ocho páginas; [preamble.tex](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/reports/preamble.tex), `neurips_2026` | Parcial | Formato/extensión cumplen; el PDF aún contiene autoría placeholder. Compilar y revisar el PDF final después del cierre. |
| Grupo de cinco y autoría real, pp. 1 y 7 | Mensaje del usuario y sección «Autoría proporcionada por el usuario» de esta evaluación: cinco nombres y afiliación; PDF auditado p. 1 y `preamble.tex:9` conservan el placeholder | Parcial | Datos de integrantes recibidos. Incorporar los nombres y la afiliación al informe y recompilar el PDF. Los correos no son una condición expresa del examen; no inventarlos ni volver a solicitar los nombres ya aportados. |
| Un comando documentado en README, p. 7 | [README del parcial](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/README.md); [MRI_RUN.md:45](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/MRI_RUN.md:45); [run_mri.ps1](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/scripts/run_mri.ps1) | Parcial | La guía contiene el comando completo, pero el README enlaza esa guía y muestra comandos genéricos con otra configuración. Incluir el comando MRI directamente y resolver la dependencia de CSV excluidos del clon. |
| Especificación de entorno, p. 7 | [requirements.txt](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/requirements.txt); [Dockerfile](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/Dockerfile); [requirements-frozen.txt](E:/Kaggle/exam-output-mri/stratified-v2/audit/requirements-frozen.txt) | Cumple localmente | Versiones directas fijadas y entorno ejecutado disponible. No se reconstruyó la imagen ni se demostró reproducción completa desde un clon limpio en otro equipo. |
| Revisión de reglas/acceso a Kaggle, p. 2 | README enumera acceso autorizado como prerrequisito; no se localizó constancia de revisión de `Rules` o de los cinco accesos personales | No verificado | El grupo debe confirmar cumplimiento y condiciones de reutilización. La presencia del dataset no demuestra esta revisión. No se ha afirmado una licencia no comprobada. |
| Código entregado mediante GitHub, pp. 1 y 7 | `git remote -v` apunta al repositorio del curso; `git status --short`: `README.md` modificado y `parcial/` no versionado; `git ls-files -- parcial` vacío | No verificado | No hay evidencia de que este contenido local esté publicado. Preparar/publicar después y comprobar el commit remoto desde un clon. No inferir el contenido remoto a partir de rutas locales. |
| PDF + enlace por plataforma y presentación, p. 7 | No se encontró comprobante de envío ni presentación; fecha normativa 30/10/2026 | Falta | Entregar por la plataforma y presentar en clase; mantener comprobante real. No se requiere envío Kaggle. |

## 4. Experimentos: evidencia de ejecución científica

Raíz vigente: **`E:/Kaggle/exam-output-mri/stratified-v2/mri/`**. Los nombres siguientes identifican exactamente los ocho subdirectorios; cada uno contiene historial, configuración, cierre, checkpoints, métricas, predicciones y tres casos Grad-CAM.

| Subdirectorio | Épocas completas / checkpoint | AUC macro 12 | F1 macro, umbral 0,5 |
|---|---:|---:|---:|
| `scratch_bce_weighted` | 48 / 38 | 0,5776426221 | 0,3609752021 |
| `scratch_bce_high_confidence` | 23 / 13 | 0,5129699221 | 0,4535273199 |
| `scratch_focal_weighted` | 29 / 19 | 0,5723243038 | 0,4960776237 |
| `scratch_focal_high_confidence` | 23 / 13 | 0,5204093209 | 0,4253469263 |
| `resnet50_bce_weighted` | 12 / 2 | 0,6189079850 | 0,3614485661 |
| `resnet50_bce_high_confidence` | 11 / 1 | 0,6020221328 | 0,4512974661 |
| `resnet50_focal_weighted` | 12 / 2 | 0,6198675938 | 0,5055771048 |
| `resnet50_focal_high_confidence` | 12 / 2 | 0,5877369699 | 0,4863370781 |

Comprobado en esta auditoría: ocho estados `scientific_run=true`; épocas contiguas y pérdidas finitas; `last.pt.epoch` igual al historial y al cierre; paciencia alcanzada; `best.pt.epoch` y pérdida coincidentes con el mínimo de validación débil; tensores de ambos checkpoints finitos; ocho hashes `best.pt` iguales a `execution.json`; 64 comparaciones de hashes de los ocho archivos de entrenamiento compartidos entre las variantes. Las 464 filas de predicción corresponden exactamente a los 58 expertos y sus doce etiquetas, con probabilidades finitas en [0,1]. Las 96 filas del CSV combinado coinciden con sus `metrics.json`.

`resnet50_focal_weighted` tiene el mayor AUC **observado**. No se ha seleccionado un modelo para despliegue ni demostrado superioridad estadística. El baseline cualitativo es `scratch_bce_weighted`, definido por el protocolo; se verifica que el código toma el primer positivo experto de cada hallazgo, no el caso con mejor apariencia o acierto.

**Inconsistencia menor de estado:** los ocho `run.json` todavía dicen `status=running` porque `fit` escribe ese valor y no lo actualiza al terminar. Las pruebas anteriores de cierre y checkpoints confirman finalización. Debe corregirse la representación del estado en una etapa posterior sin alterar la evidencia congelada de estos experimentos.

## 5. Datos, ruido y límites de interpretación

### Partición corregida y evaluación repetida

Se compararon los manifiestos actual y sustituido. Conservan exactamente los mismos 600 estudios y los mismos 58 expertos. Los positivos débiles de Lateral OA pasaron de **0 train / 14 val** a **10 / 4**; Medial OA, de **3 / 44** a **35 / 12**. El algoritmo vigente se reconstruyó desde los CSV originales: usa cuotas de etiquetas débiles, no rendimiento experto. Las doce etiquetas tienen positivos débiles en ambas particiones.

La primera matriz permanece en `E:/Kaggle/exam-output-mri/`, con [protocol_status.json](E:/Kaggle/exam-output-mri/audit/protocol_status.json) marcado `superseded_for_submission`. La explicación consta en [MRI_PLAN.md](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/MRI_PLAN.md), `MRI_RUN.md`, `MRI_STATUS.json` y PDF pp. 1-2. El notebook explica reserva experta, pero conviene incorporar también allí una nota explícita de la evaluación repetida: su texto actual no desarrolla esa historia con la claridad del PDF.

La ausencia de cambios actuales en léxico, umbral e hiperparámetros se respalda con código congelado y documentos. La motivación histórica de todas las decisiones humanas no se puede demostrar solamente con hashes. Con expertos reutilizados no existe una evaluación final completamente nueva; aumentar independencia requeriría nuevos expertos o un diseño posterior independiente, como mejora científica, no como requisito expreso de entrega de esta pregunta.

### PatientID, secuencias y cobertura textual

Se comprobaron 600 `StudyInstanceUID` únicos y se leyó una cabecera DICOM original por serie: sin PatientID vacío ni intersecciones entre particiones en esas cabeceras. La auditoría anterior declara haber revisado todas las 22.954 instancias. **Esta revisión no repitió ese recorrido completo**; tampoco la igualdad de los IDs anonimizados garantiza identidad transversal de personas reales.

Todas las 600 series seleccionadas son sagitales: 588 tienen descriptor sensible a líquido y supresión grasa, y 12 tienen ambos indicadores en 0. La selección/fallback se conserva en `manifest.json`; la reducción a una serie es compatible con la recomendación del examen, pero limita hallazgos que requieran otro plano/secuencia.

Se reextrajeron las etiquetas de los 600 reportes seleccionados y se verificaron contra el manifiesto, junto con las etiquetas expertas contra `train.csv`. También se reprodujeron los conteos del universo de **4.349 reportes no expertos**: 3.939 compatibles con alguna regla. Cobertura por escritura: Latin/mixed 3.488/3.811, Cyrillic 209/217, Greek 239/318, Thai 3/3. La muestra seleccionada tiene Latin/mixed 532, Greek 65 y Cyrillic 3. Estos grupos **no identifican trece idiomas validados ni los 16 centros**; tres reportes tailandeses no sostienen una estimación estable de desempeño.

Los pesos 0,9 para afirmación explícita, 0,2 para ausencia de mención y 0 para abstención son heurísticos. No son probabilidades calibradas. En Lateral OA, el desacuerdo de alta confianza de 0 se apoya en **solo 1/58 estudio**; interpretarlo como extractor perfecto sería incorrecto. Para sinovitis, el filtro conserva en train 140 observaciones y las 140 son positivas: esa ablación no dispone de negativos explícitos para esa etiqueta. El PDF ya reconoce el problema general del filtrado; explicitar este caso fortalecería la discusión.

### Desbalance y mitigación

Conteos comprobados. La tasa de train siguiente usa los 434 estudios como denominador; `prevalence.csv` también permite usar el denominador observado por etiqueta y considerar abstenciones.

| Etiqueta | Train positivos / 434 (%) | Validación débil positivos / 108 | Expertos positivos / 58 |
|---|---:|---:|---:|
| ACL | 143 (32,95) | 42 | 24 |
| MCL | 102 (23,50) | 39 | 9 |
| Medial Meniscus | 158 (36,41) | 47 | 26 |
| Lateral Meniscus | 127 (29,26) | 41 | 23 |
| Medial OA | 35 (8,06) | 12 | 15 |
| Lateral OA | 10 (2,30) | 4 | 11 |
| PF OA | 45 (10,37) | 18 | 21 |
| Effusion | 282 (64,98) | 77 | 35 |
| Synovitis | 140 (32,26) | 41 | 27 |
| Baker's | 145 (33,41) | 43 | 12 |
| Contusion | 129 (29,72) | 39 | 19 |
| Fracture | 124 (28,57) | 42 | 18 |

Los más raros en **esta selección débil** son Lateral OA y Medial OA; no hay que imponer a esta muestra la rareza sugerida en el enunciado para fractura/Baker. ResNet ponderado, BCE → focal: Medial OA recall 0,133 → 0,867, F1 0,222 → 0,473, precisión 0,667 → 0,325. Para Lateral OA mejora recall/F1 pero AUC baja aproximadamente 0,683 → 0,644. Está evaluado el compromiso, no una mejora uniforme. La selección por rareza altera la distribución; no usar estas tasas como estimación poblacional.

### Grad-CAM: requisito contestado, validación clínica ausente

Los 24 NPZ de explicación, tres por variante, tienen imagen/mapa de ocho cortes 224×224 y mapas finitos en [0,1]. En el baseline, `cam_cases.json` registra LCA (grupo 7, p≈0,565), menisco medial (grupo 0, p≈0,661) y derrame (grupo 0, p≈0,810), todos positivos expertos. La figura 4, p. 8, muestra las tres imágenes y superposiciones.

El PDF pp. 4-5 **sí discute** si son plausibles: LCA amplio/periférico sin foco inequívoco; menisco medial difuso sin delimitación selectiva; derrame periférico con fondo. Esta auditoría visual confirma difusión y señal fuera del tejido en las imágenes existentes; no certifica anatomía radiológica. Se explican resolución 14×14, interpolación, normalización por grupo y sesgo del máximo de activación media. No hay ROI expertas ni prueba cuantitativa de localización, ni revisión radiológica registrada.

Una persona del equipo debe comprobar que la descripción final representa lo mostrado. La revisión por especialista sería un control adicional útil si se pretendiera sostener afirmaciones clínicas; **no debe bloquear artificialmente la entrega por exigir una firma que el profesor no pidió**.

## 6. Integridad y qué fue verificado ahora

Fuente: [delivery.json](E:/Kaggle/exam-output-mri/stratified-v2/audit/delivery.json). Los nueve archivos todavía coinciden tanto en tamaño como en SHA-256:

| Archivo exacto | SHA-256 vigente comprobado |
|---|---|
| `E:/Kaggle/exam-output-mri/stratified-v2/reports/mri_draft.pdf` | `29befef7c561bd33c25690d54062288e9a382f666075dcb3606ea4f1b4486b4a` |
| `D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/notebooks/01_mri.ipynb` | `eac75335ebf86ad1b3f2aa4b249557f0f0f6f889206f6e3fd4f008fb1c4f8a1a` |
| `E:/Kaggle/exam-output-mri/stratified-v2/mri/experiments.csv` | `73bfd81fb6680dfd3a66518084320f8bfd62d2510cce9978a2aabd503a1f0adb` |
| `E:/Kaggle/exam-output-mri/stratified-v2/mri/metrics_all_labels.csv` | `8b479e79d6f6306c71c085f8531ae0967bca27637389c625b2e2437005281ea9` |
| `E:/Kaggle/exam-output-mri/stratified-v2/mri/class_balance.csv` | `4482ff0fc2f9960f3412ed34577f9820fd61d79deb075eddca8e0d6a27e8f0de` |
| `E:/Kaggle/exam-output-mri/stratified-v2/audit/mri_results.json` | `5996e8b4fb82ead6c3f4e4b0a8446d05cf740bfa86c5d1d0aee8f2e90702acd5` |
| `E:/Kaggle/exam-output-mri/stratified-v2/audit/mri_data.json` | `de71b23c32cea7f82783060c237b064ab09014489a60faaa05e569106b348a94` |
| `D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/MRI_RUN.md` | `bc4b10978dee2dfc9f0a8159d2e3d6957906b71ffa0902cf335328e0897ef1d1` |
| `D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/configs/mri.toml` | `9f0e83b53658a7189446d5278345cfb6b8c5e8960de529775f43a0b1f648d7e6` |

Las copias del notebook en `E:/Kaggle/exam-output-mri/stratified-v2/notebooks/01_mri.ipynb` y `E:/Kaggle/exam-output-mri/stratified-v2/audit/delivery_source_snapshot/notebooks/01_mri.ipynb` tienen el mismo hash. La configuración coincide byte a byte con `E:/Kaggle/exam-output-mri/stratified-v2/audit/mri-frozen.toml`. Las fuentes actuales de método/discusión coinciden con las compiladas y con la instantánea de entrega.

| Comprobado por esta auditoría | Solo constancia anterior o no repetido |
|---|---|
| Nueve hashes y tamaños; tres notebooks idénticos; ocho celdas sin errores guardados | No se reejecutó el notebook. La ejecución anterior consta en `E:/Kaggle/exam-output-mri/stratified-v2/audit/notebook.log`. |
| Métricas exactas de 464 predicciones, 96 filas combinadas, cierres y checkpoints CPU | No se volvió a inferir con los checkpoints ni a entrenar. Se comprobó coherencia y procedencia registrada, no identidad de nueva inferencia. |
| Hash y contenido de las 600 NPZ exportadas: float32, 8×3×224×224, finitas y en [0,1] | Igualdad con el volumen Docker durante la exportación declarada en `audit/exported_tensor_cache.json`; no se remontó ese volumen para otra comparación. |
| CSV originales y locales coinciden con hashes; extracción y partición reconstruidas; una cabecera por serie | Integridad/decodificación completa de las 22.954 instancias y comprobación de todas las cabeceras declaradas en `audit/mri_data.json`, `audit/mri_source_files.json` y logs. No se repitieron. |
| Lectura de `audit/tests.log`: `32 passed, 14 warnings in 3.30s` | No se volvió a correr la suite. La prueba de reanudación es un fixture controlado, no repetibilidad del experimento MRI entre equipos. |
| PDF de ocho páginas, siete entradas bibliográficas, fuentes iguales y ocho PNG preexistentes inspeccionados | No se recompiló ni se regeneraron páginas. Los PNG son la revisión visual existente; no se calculó una nueva correspondencia píxel a píxel con el PDF. |
| Log LaTeX sin `Overfull`, referencias indefinidas o error; aparecen avisos `Underfull` | La revisión editorial sigue pendiente: saltos de figuras interrumpen párrafos, particularmente pp. 5-7. No se debe describir el log como absolutamente libre de advertencias. |

El PDF pp. 6-7 dice que «las predicciones se recalculan durante la auditoría de métricas». El script [audit_mri_results.py](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/scripts/audit_mri_results.py) **recalcula métricas desde predicciones persistidas**, no ejecuta nuevamente el modelo. Corregir esa frase en el cierre editorial evita exagerar lo validado. Las ramas `\pending` de `main.tex` son fallbacks de compilación; no aparecen como resultados faltantes en el PDF vigente. El placeholder visible es el de autoría.

## 7. Problema comprobado de reproducción desde un clon limpio

[compose.mri.yaml:12](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/compose.mri.yaml:12) fija `RSNA_METADATA_DIR=/workspace/data`. [select_mri_series.py](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/scripts/select_mri_series.py) y `audit_mri.py` leen allí `train.csv` y `train_series.csv`. `.gitignore` excluye `parcial/data/`, y el runner no inicializa esa carpeta desde los originales descargados.

Se simuló la carpeta de metadatos vacía en un contenedor de solo lectura, manteniendo disponibles los CSV originales en `/datasets/rsna`. Resultado real: ambos archivos ausentes en `/workspace/data`, presentes en `/datasets/rsna`, y `FileNotFoundError` para `/workspace/data/train_series.csv`. No se intentó un entrenamiento ni se modificaron las salidas para demostrarlo.

**Solución prevista:** hacer que el Compose MRI lea los CSV de `/datasets/rsna` o añadir una inicialización explícita a partir del dataset autorizado. La primera opción reutiliza directamente el montaje original de solo lectura y evita incluir reportes clínicos en Git; requiere comprobar todos los consumidores del directorio de metadatos. Si se conserva la copia local, el README debe documentar exactamente cómo se obtiene y el comando debe verificarla antes de continuar. No se aplicó ninguna de estas modificaciones en esta etapa de auditoría.

Además, el comando `./scripts/run_mri.ps1` está en `MRI_RUN.md`, mientras los README muestran el flujo genérico. Para satisfacer de forma inequívoca la p. 7, el README debe presentar directamente el comando y la configuración MRI vigente, sus prerrequisitos y qué produce.

## 8. Plan propio priorizado y criterios de cierre

Este plan no autoriza ni ejecuta nuevas modificaciones científicas, publicación o entrenamiento en la presente etapa.

| Prioridad / paso | Acción concreta | Dependencia | Criterio de cierre y comprobación |
|---|---|---|---|
| P0.1: reproducción del código | Resolver el directorio de CSV de un clon limpio; añadir `./scripts/run_mri.ps1` directamente al README, con dataset autorizado, Docker/GPU, rutas y salida vigente. Conservar el código/configuración científica congelados. | Ninguna revisión radiológica. | Un clon sin `parcial/data/` encuentra los CSV desde la descarga autorizada. Comprobación CPU de selección produce los mismos 600 IDs, particiones y series; sin tocar la salida vigente. `requirements.txt`, Compose, runner y guía presentes en el commit de entrega. |
| P0.2: incorporar autoría y revisar el informe | Incorporar al informe los cinco nombres y la afiliación ya proporcionados, conservando su escritura y orden. Revisar con el equipo métodos, resultados y límites. Confirmar la revisión de reglas/accesos exigida en p. 2. | Datos de integrantes disponibles en esta evaluación; revisión humana auténtica aún pendiente. | PDF sin placeholder de autores, con los cinco nombres y la afiliación suministrados. No inventar firmas, correo, revisión ni licencia; no volver a pedir nombres. |
| P0.3: cierre editorial científico | Corregir «predicciones» por «métricas desde predicciones persistidas»; mantener divulgación de expertos reutilizados; explicitar el caso sinovitis del filtro; mejorar denominadores/porcentajes y continuidad de figuras/párrafos. Añadir nota de repetición en notebook y README si se editan en la etapa posterior. | P0.2; mantener resultados auditados. | Los siete puntos de la p. 3 siguen respondidos, ninguna afirmación de validación externa/causal/localización clínica, números consistentes con CSV. Revisión del equipo realizada, sin exigir un radiólogo externo para el requisito académico. |
| P0.4: producir PDF final | Registrar `finalized.json` únicamente después de la revisión real y compilar sin `--draft`. Ese archivo es un control interno. Revisar todas las páginas del nuevo PDF y actualizar hashes de la entrega final, preservando la constancia histórica actual. | P0.2 y P0.3. | `E:/Kaggle/exam-output-mri/stratified-v2/reports/mri.pdf` existe, 6-10 páginas, ≥5 referencias, cinco autores, sin placeholders ni referencias indefinidas; todas las figuras legibles. |
| P0.5: publicación verificable | Preparar la versión de código MRI, entorno, README, notebook y fuentes del informe en GitHub. Comprobar el contenido publicado desde un clon y registrar commit/URL exactos. | P0.1 y cierre de archivos seleccionados; publicación en una etapa autorizada posterior. | El commit remoto contiene los archivos necesarios y su comando documentado. No basta el URL del repositorio ni `git status` local. Mantener dataset, textos clínicos y pesos fuera de Git, conforme a reglas comprobadas. |
| P0.6: entrega y defensa | Subir PDF final y enlace GitHub a la plataforma; preparar la exposición del grupo para el 30/10/2026. | P0.4 y P0.5. | Comprobante real de entrega y presentación en clase. Explicar ruido/abstenciones, AUC de doce etiquetas, selección por validación débil, corrección de split, expertos reutilizados y mapas limitados. |
| P1: claridad de estado | Ajustar el estado terminal de `run.json` en una versión futura del runner y documentar que la finalización actual se demuestra por otros artefactos. | Cambio posterior, sin reescribir constancias congeladas. | Una ejecución de fixture posterior produce estados de cierre coherentes y conserva reanudación. No es motivo para repetir ahora la matriz MRI. |
| P2: mejoras de investigación | Si se busca mayor rigor futuro: incertidumbre estadística pareada, semillas adicionales, expertos independientes, auditoría por idioma/centro y regiones anatómicas anotadas. | Nuevos recursos/datos y protocolo previo a observar resultados. | Son extensiones; no convertirlas en requisitos inventados para aprobar los siete entregables del examen. No se ejecutan en esta auditoría. |

**Tareas humanas realmente necesarias:** revisar y asumir el contenido final, incluida la interpretación limitada de las figuras; confirmar las reglas/accesos Kaggle; decidir la publicación y realizar la entrega/presentación. Los nombres y la afiliación ya están disponibles; incorporarlos al informe es una tarea editorial que no requiere otra consulta de identificación. La revisión editorial es necesaria para cerrar el borrador y el control interno. El examen no exige que un radiólogo firme ni que se obtenga aprobación clínica para este proyecto académico.

## 9. Archivos a presentar y a conservar

### Entrega exigida por la plataforma del curso

| Elemento | Ruta o enlace exacto | Estado |
|---|---|---|
| PDF de pregunta 1, versión final | `E:/Kaggle/exam-output-mri/stratified-v2/reports/mri.pdf` | **Por producir.** El existente es `E:/Kaggle/exam-output-mri/stratified-v2/reports/mri_draft.pdf`; no confundirlos. El nombre `mri.pdf` es una convención de la implementación, no un nombre impuesto por el profesor. |
| Enlace al código reproducible publicado | [Repositorio del curso](https://github.com/miguelsff/maestria-ia-uni-ciclo-2-redes-neuronales), preferentemente enlace al commit/carpeta de entrega | **Publicación de esta versión no verificada.** Debe acompañar al PDF en la plataforma. |

No se exige subir al leaderboard, un notebook separado, checkpoints o todas las evidencias de auditoría a la plataforma. Pueden acompañar la defensa si el docente los solicita.

### Archivos mínimos de la versión de código MRI

Base absoluta: `D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/`. Publicar el paquete fuente completo de `parcial/src/neural_exam/` para conservar sus imports y la CLI; no separar únicamente el notebook. La lista concreta de archivos MRI y soporte a comprobar es:

- `D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/README.md` y `.gitignore`.
- Bajo esa base, `parcial/README.md`, `parcial/MRI_RUN.md`, `parcial/MRI_PLAN.md`, `parcial/MRI_STATUS.json`, `parcial/requirements.txt`, `parcial/pyproject.toml`, `parcial/Dockerfile`, `parcial/.dockerignore`, `parcial/compose.mri.yaml`, `parcial/configs/mri.toml`.
- `parcial/src/neural_exam/__init__.py`, `__main__.py`, `config.py`, `cli.py`, `reporting.py`; todos los módulos MRI: `mri/__init__.py`, `data.py`, `labels.py`, `loss.py`, `model.py`, `pipeline.py`; soporte compartido: `common/__init__.py`, `cam.py`, `download.py`, `figures.py`, `io.py`, `locking.py`, `metrics.py`, `training.py`. Los prefijos corresponden a la base absoluta indicada; incluir también los módulos del paquete que importa la CLI.
- Scripts llamados por el runner: `parcial/scripts/run_mri.ps1`, `select_mri_series.py`, `cache_mri_data.py`, `audit_mri.py`, `download_mri_assets.py`, `verify_mri_runtime.py`, `audit_mri_results.py`, `prepare_mri_report.py`, `compile_reports.py`.
- `parcial/tests/test_mri.py`, `parcial/tests/test_metrics.py`, `parcial/tests/test_utilities.py`; `parcial/notebooks/01_mri.ipynb`; `parcial/reports/preamble.tex`, `parcial/reports/references.bib`, `parcial/reports/mri/main.tex`, `parcial/reports/mri/discussion.tex` y el futuro `parcial/reports/mri/finalized.json` si se utiliza el cierre de la implementación.

Los nombres abreviados de esta lista se resuelven bajo la base/prefijo explícitos; no son rutas a otras salidas. Las tablas/figuras del informe se generan desde experimentos y el kit NeurIPS descargado; la versión publicada debe explicar ese paso. Los resultados locales no pasan automáticamente a GitHub.

Conservar como respaldo local exacto: `E:/Kaggle/exam-output-mri/stratified-v2/audit/delivery.json`, `mri_results.json`, `mri_data.json`, `exported_tensor_cache.json`, `tests.log` y sus instantáneas; `E:/Kaggle/exam-output-mri/stratified-v2/mri/manifest.json`, `experiments.csv`, `metrics_all_labels.csv`, `label_agreement.json`, `class_balance.csv`, `prevalence.csv` y los ocho directorios de experimentos de la sección 4. Mantener también el protocolo sustituido en su salida original para sostener la divulgación. Estas evidencias no son una orden de publicar datos clínicos o todos los pesos.

## 10. Comandos concretos de comprobación y cierre

### A. Hashes y notebook, lectura local

Ejecutado de manera equivalente en esta auditoría; no modifica archivos:

```powershell
$mriOut = 'E:/Kaggle/exam-output-mri/stratified-v2'
$delivery = Get-Content -LiteralPath "$mriOut/audit/delivery.json" -Raw | ConvertFrom-Json
$delivery.artifacts.PSObject.Properties | ForEach-Object {
    $artifact = $_.Value
    $actual = (Get-FileHash -LiteralPath $artifact.path -Algorithm SHA256).Hash.ToLowerInvariant()
    [pscustomobject]@{ artifact=$_.Name; match=($actual -eq $artifact.sha256); bytesMatch=((Get-Item -LiteralPath $artifact.path).Length -eq $artifact.bytes) }
}
Get-FileHash -Algorithm SHA256 -LiteralPath @(
  'D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/notebooks/01_mri.ipynb',
  "$mriOut/notebooks/01_mri.ipynb",
  "$mriOut/audit/delivery_source_snapshot/notebooks/01_mri.ipynb"
)
```

### B. Métricas y checkpoints en CPU, sin reescribir auditorías

El script existente `audit_mri_results.py` escribe `audit/mri_results.json`; **no ejecutarlo contra la salida vigente durante una revisión de solo lectura**. La comprobación independiente utilizó lectura por stdin con montajes de solo lectura. Este comando concreto permite repetir el núcleo de métricas y selección:

```powershell
$mriProject = 'D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial'
$mriOut = 'E:/Kaggle/exam-output-mri/stratified-v2'
@'
import csv, json, hashlib
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
r = Path('/outputs/mri')
def rd(p): return json.loads(p.read_text(encoding='utf-8'))
def rows(p):
    with p.open(encoding='utf-8', newline='') as f: return list(csv.DictReader(f))
m = rd(r/'manifest.json'); labels = m['labels']
gold = {x['study_id']:x for x in m['records'] if x['split']=='gold'}
assert len(gold)==58
for e in rows(r/'experiments.csv'):
    d=r/e['experiment']; s=rd(d/'metrics.json'); x=rd(d/'execution.json')
    assert x['scientific_run'] and x['threshold']==s['threshold']==0.5
    assert hashlib.sha256((d/'best.pt').read_bytes()).hexdigest()==x['selected_checkpoint_sha256']
    history=rows(d/'history.csv'); best=min(history,key=lambda h:float(h['val_loss']))
    b=torch.load(d/'best.pt',map_location='cpu',weights_only=True)
    assert b['epoch']==int(best['epoch']) and b['best_loss']==float(best['val_loss'])
    assert len(history)==rd(d/'complete.json')['epochs_executed']
    p=rows(d/'predictions.csv'); ids=[v['StudyInstanceUID'] for v in p]
    assert len(ids)==58 and set(ids)==set(gold)
    y=np.array([gold[i]['targets'] for i in ids])
    q=np.array([[float(v[k]) for k in labels] for v in p])
    assert np.isfinite(q).all() and ((q>=0)&(q<=1)).all()
    auc=[]; f1s=[]
    for j,k in enumerate(labels):
        a=float(roc_auc_score(y[:,j],q[:,j]))
        pr,re,f1,_=precision_recall_fscore_support(y[:,j],q[:,j]>=0.5,average='binary',zero_division=0)
        z=s['per_label'][j]
        assert z['label']==k
        assert all(float(z[key])==float(v) for key,v in [('auc',a),('precision',pr),('recall',re),('f1',f1)])
        auc.append(a); f1s.append(f1)
    assert float(np.mean(auc))==s['macro_auc']==float(e['macro_auc'])
    assert float(np.mean(f1s))==s['macro_f1']==float(e['macro_f1'])
    print(e['experiment'], 'PASS', len(history), b['epoch'])
'@ | docker run --rm -i --network none --cpus 2 --memory 4g --read-only --tmpfs /tmp:rw,size=128m -e PYTHONDONTWRITEBYTECODE=1 --mount "type=bind,source=$mriOut,target=/outputs,readonly" --entrypoint python neural-exam-mri:0.1 -
```

La auditoría ejecutada fue más amplia: también cargó `last.pt`, verificó tensores, 64 hashes de fuentes, 600 hashes/contenidos de caché, mapas, acuerdo y balance. No ejecutó `fit` ni inferencia. Para comprobar los NPZ exportados sin generarlos:

```powershell
$receipt = Get-Content -LiteralPath "$mriOut/audit/exported_tensor_cache.json" -Raw | ConvertFrom-Json
$bad = @($receipt.files.PSObject.Properties | Where-Object {
    (Get-FileHash -LiteralPath "$mriOut/mri/cache/$($_.Name)" -Algorithm SHA256).Hash.ToLowerInvariant() -ne $_.Value
})
"Mismatches: $($bad.Count)"  # Esperado: 0 de 600.
```

### C. Evidencia de cierre del notebook, PDF y pruebas

```powershell
$nb = Get-Content -LiteralPath "$mriProject/notebooks/01_mri.ipynb" -Raw | ConvertFrom-Json
$code = @($nb.cells | Where-Object cell_type -eq code)
$errors = @($code.outputs | Where-Object output_type -eq error)
[pscustomobject]@{codeCells=$code.Count; errors=$errors.Count; executionCounts=($code.execution_count -join ',')}
Select-String -LiteralPath "$mriProject/notebooks/01_mri.ipynb" -Pattern 'RUN_TRAINING = False'
Get-Content -LiteralPath "$mriOut/audit/tests.log" -Tail 2
Select-String -LiteralPath "$mriOut/reports/build/mri/main.log" -Pattern 'Overfull|Underfull|undefined|LaTeX Error'
git -C 'D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales' status --short
git -C 'D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales' ls-files -- parcial
```

Para contar páginas sin recompilar, con el Python disponible en esta máquina:

```powershell
@'
from pypdf import PdfReader
from pathlib import Path
import re
p=Path('E:/Kaggle/exam-output-mri/stratified-v2/reports/mri_draft.pdf')
print('Pages:',len(PdfReader(p).pages))
b=p.parent/'build/mri/main.bbl'
print('References:',len(re.findall(r'\\bibitem',b.read_text(encoding='utf-8'))))
'@ | & 'C:/Users/USER/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -
```

Lectura visual: `E:/Kaggle/exam-output-mri/stratified-v2/reports/mri-page-1.png` hasta `mri-page-8.png`. Se inspeccionaron las ocho. Para el PDF final nuevo será necesario inspeccionar su propia representación visual, no asumir que estos PNG siguen describiéndolo.

### D. Reproducir el defecto de metadatos sin ejecutar el runner

```powershell
@'
from pathlib import Path
from neural_exam.config import load_settings
s=load_settings('/workspace/configs/mri.toml')
print(s.metadata_root)
for n in ['train.csv','train_series.csv']:
    print(n, (s.metadata_root/n).exists(), Path('/datasets/rsna',n).exists())
from neural_exam.common.io import read_csv
try: read_csv(s.metadata_root/'train_series.csv')
except FileNotFoundError as e: print(e)
'@ | docker run --rm -i --network none --cpus 1 --memory 1g --read-only --tmpfs /tmp:rw,size=64m --tmpfs /workspace/data:rw,size=16m -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/workspace/src -e RSNA_METADATA_DIR=/workspace/data --mount "type=bind,source=$mriProject,target=/workspace,readonly" --mount 'type=bind,source=E:/Kaggle/rsna-knee-abnormality-detection,target=/datasets/rsna,readonly' --entrypoint python neural-exam-mri:0.1 -
```

Resultado comprobado: metadatos locales `False`, originales `True`, error al leer la carpeta vacía. Después de corregir el empaquetado, repetir una comprobación CPU con el entorno corregido y comparar IDs/particiones, sin usar la salida vigente como destino de escritura.

### E. Comandos del plan posterior; no ejecutados ahora

Después de revisión auténtica, autoría completa y registro de cierre:

```powershell
Set-Location 'D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial'
docker compose -f compose.mri.yaml run --rm reports --task mri
```

La compilación escribe `reports/mri.pdf`; requiere `parcial/reports/mri/finalized.json` con revisión verdadera. La revisión anatómica profesional no está implementada como un bloqueo específico en ese compilador.

La suite declarada puede comprobarse después en CPU y espacio temporal, sin cambiar resultados existentes; incluye entrenamiento de un fixture sintético de reanudación, por lo que **no se ejecutó en esta auditoría**:

```powershell
docker run --rm --network none --cpus 2 --memory 4g --read-only --tmpfs /tmp:rw,size=512m -e PYTHONDONTWRITEBYTECODE=1 -e MPLCONFIGDIR=/tmp/mpl -e PYTHONPATH=/workspace/src --mount "type=bind,source=$mriProject,target=/workspace,readonly" --workdir /workspace --entrypoint python neural-exam-mri:0.1 -m pytest tests/test_mri.py tests/test_metrics.py tests/test_utilities.py -q -p no:cacheprovider --basetemp /tmp/pytest
```

Si una etapa posterior necesita comprobar el comando completo de extremo a extremo desde un clon, usar una **salida nueva**, tras corregir los metadatos y con los datos autorizados. Consume recursos de entrenamiento; aquí solo se deja el comando:

```powershell
$env:MRI_OUTPUT_DIR = 'E:/Kaggle/exam-output-mri/reproduction-after-audit'
./scripts/run_mri.ps1
```

No usar ese comando para repetir ahora la matriz vigente ni `-Resume` para recalcular y sobrescribir evaluación durante esta auditoría. El cambio de salida debe acompañarse de una comprobación de los volúmenes de caché compartidos antes de una ejecución posterior.

## 11. Procedencia de la auditoría

Las evidencias científicas provienen de los archivos locales enumerados; las skills de PDF, diseño de documentación, PyTorch y scikit-learn orientaron la revisión. Como referencia procedimental se incluye Kassis, T., Agarwal, V., He, Y., Patel, D. y Brueckner, A. M. (2026), [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://arxiv.org/abs/2609.00065), DOI `10.48550/arXiv.2609.00065`; metadatos y versión vigente consultados el 4/10/2026. Esta referencia no valida los resultados MRI ni sustituye la evaluación del profesor o del equipo.

**Cierre de esta evaluación:** implementación y matriz científica verificadas; notebook de consulta y borrador existentes; cinco nombres y afiliación recibidos; independencia externa y revisión clínica no demostradas. La entrega final sigue pendiente de empaquetado reproducible, incorporación de la autoría al PDF y cierre editorial, publicación comprobada y trámite del curso.

## 12. Implementación posterior autorizada

Después de la auditoría, el usuario solicitó implementar el plan. El registro anterior describe el estado auditado; el cierre técnico actual se documenta en [MRI_DELIVERY.md](../../MRI_DELIVERY.md) y [MRI_VERIFICATION.json](../../MRI_VERIFICATION.json).

Se corrigió la lectura de metadatos desde los originales y se documentó el comando completo en ambos README. El runner rechaza una salida con corridas previas antes de escribir logs si no se solicita reanudación. Las futuras corridas cierran el estado terminal de `run.json`; los metadatos históricos de las ocho corridas permanecieron intactos.

El PDF incorpora cinco autores, porcentajes de prevalencia, la advertencia de sinovitis sin negativos explícitos, expertos reutilizados y la corrección de métricas desde predicciones persistidas. Tablas y figuras se distribuyen sin interrumpir los párrafos. El PDF de nueve páginas y siete referencias se compiló sin `--draft`, se inspeccionaron sus nueve páginas y no contiene referencias indefinidas ni cajas desbordadas. `finalized.json` registra revisión técnica/visual por Codex y mantiene pendiente la aprobación del grupo; no acredita una revisión humana que no se haya realizado.

La salida nueva es `E:/Kaggle/exam-output-mri/submission-2026-10-04`; el PDF también se copió al nombre previsto `E:/Kaggle/exam-output-mri/stratified-v2/reports/mri.pdf`, sin reemplazar el borrador. Se ejecutaron las ocho celdas del notebook con cero errores y entrenamiento desactivado. Los 131 archivos científicos inventariados conservaron sus SHA-256; no se reentrenó la matriz científica. Las versiones históricas del notebook y la guía permanecen en la instantánea editorial original.

La rama `codex/mri-delivery` se publicó con el código en el commit `45efe9d905f4a5a034334de18be3093fcd7d9146`. Un clon real de GitHub pasó las 32 pruebas CPU y produjo exactamente la selección y particiones de los 600 estudios sin `parcial/data/`. Se publicaron fuentes, dependencias, pruebas, guías y notebook sin salidas clínicas. No se publicaron DICOM, textos clínicos, predicciones individuales, pesos, imágenes clínicas ni PDF. El repositorio continúa privado; el acceso del docente queda por confirmar.

P0.1, P0.2, la edición técnica de P0.3, P0.4, P0.5 y P1 están implementados y comprobados. Para P0.6 se preparó [MRI_DEFENSE.md](../../MRI_DEFENSE.md). Siguen pendientes la aprobación editorial del grupo, las obligaciones personales y condiciones de uso Kaggle, el comprobante de la plataforma y la presentación en clase del 30/10/2026. No se conoce el enlace de la plataforma. No se ejecutaron las extensiones de investigación P2.
