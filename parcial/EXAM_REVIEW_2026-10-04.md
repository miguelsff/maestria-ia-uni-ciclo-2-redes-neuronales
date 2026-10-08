# Estado y entrega del examen parcial

Revisión del 4 de octubre de 2026, zona horaria America/Lima. MRI e inundaciones tienen experimentos completos y borradores locales respaldados por evidencias. Fauna todavía no tiene ejecución científica completa. Ninguna pregunta se considera entregada al profesor: faltan autoría, cierre editorial y evidencia de publicación de la versión reproducible en GitHub.

Esta revisión es una fotografía del estado local. Las evaluaciones específicas de los tres chats podrán incorporar hallazgos adicionales. No se reentrenaron modelos ni se ejecutaron pruebas del proyecto durante esta revisión general.

## Estado por pregunta

| Pregunta | Respuesta e implementación | Ejecución | Validación disponible | Entrega final |
|---|---|---|---|---|
| 1. MRI de rodilla | Los siete entregables tienen código, resultados o discusión en el borrador. Falta evaluación específica de cumplimiento. | Ocho variantes completas en `stratified-v2`. | 464 predicciones auditadas; logs de 32 pruebas aprobadas; notebook ejecutado sin errores; PDF de ocho páginas. | Pendiente: cinco autores, cierre editorial, publicación y comprobación de reproducción desde GitHub. La revisión anatómica profesional es un control adicional del proyecto. |
| 2. Inundaciones | Los cinco entregables obligatorios están desarrollados en código y borrador. Sentinel-2 es opcional. | Seis variantes completas. | 618 GeoTIFF auditados; logs de 20 pruebas aprobadas; notebook ejecutado sin errores; PDF de nueve páginas. | Pendiente: cinco autores, cierre editorial, publicación y comprobación de reproducción desde GitHub. |
| 3. Fauna | Pipeline preparado; resultados finales, robustez, demostración y discusión empírica incompletos. | Auditoría, reanudación y smoke disponibles; entrenamiento completo pendiente. | Comprobaciones funcionales; 90 revisiones de etiquetas pendientes; sin revisión de oclusión. | No hay PDF científico. Faltan revisión del dataset, entrenamiento, evaluación, actividad ecológica, notebook de resultados y cierre de entrega. |

"Pruebas aprobadas" describe los logs de las ejecuciones anteriores, no pruebas repetidas hoy. Un notebook ejecutado puede limitarse a leer evidencias o a mostrar smoke; su ejecución no acredita por sí sola el entrenamiento científico.

## Lo que exige el profesor

Fuente de autoridad: [examen-parcial.pdf](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/examen-parcial.pdf), páginas 1 y 7.

1. Grupo de **cinco estudiantes**.
2. **Un informe por pregunta**, PDF en formato NeurIPS de **6 a 10 páginas**. Para las tres preguntas corresponden tres informes.
3. Cada informe debe contener motivación, trabajos relacionados con **al menos cinco referencias**, metodología, experimentos, resultados y discusión crítica de limitaciones.
4. Código reproducible entregado mediante **GitHub**. El examen no exige tres repositorios separados: un repositorio puede contener los tres proyectos si cada ejecución es identificable y reproducible.
5. Un comando de extremo a extremo documentado en README y especificación de entorno en `requirements.txt` o `environment.yml`.
6. PDF y enlace GitHub entregados por la plataforma del curso. **Entrega y presentación en clase: 30 de octubre de 2026**. Penalización indicada: 10 % por día de retraso.

No se requiere envío al leaderboard de Kaggle. Sentinel-2 y un segundo dataset de fauna son extensiones opcionales. Los notebooks, checkpoints, manifiestos, predicciones y constancias de auditoría respaldan la reproducción, aunque el examen no los enumera todos como adjuntos independientes.

[ejemplo.pdf](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/ejemplo.pdf) orienta el estilo del informe. Sus once páginas y cuatro autores no reemplazan los límites ni la composición del grupo exigidos por el examen.

## Pregunta 1: MRI

Salida vigente: `E:/Kaggle/exam-output-mri/stratified-v2`. La primera ejecución en `E:/Kaggle/exam-output-mri` está conservada y fue sustituida para entrega.

| Requisito del examen, pp. 1-3 | Evidencia local | Estado y pendiente |
|---|---|---|
| Extraer doce etiquetas de reportes multilingües; validar acuerdo/ruido frente a expertos. | `src/neural_exam/mri/labels.py`, `mri/label_agreement.json`, tablas y discusión del informe. | Implementado y reportado. Auditar cobertura, abstención y negación; los pesos son heurísticos, no confianza calibrada. |
| CNN propia y transferencia desde ImageNet con doce salidas independientes. | `src/neural_exam/mri/model.py`, ocho carpetas de experimentos con `best.pt`, `last.pt`, historial y predicciones. | Ejecutado. |
| AUC macro de doce etiquetas, AUC por etiqueta, precisión, recall y F1 con umbral justificado. | `mri/experiments.csv`, `metrics_all_labels.csv`, `audit/mri_results.json`. | Reportado. Las doce AUC están definidas en los 58 estudios expertos; el umbral fijo es 0,5. |
| Ablación de al menos dos factores. | Matriz CNN/ResNet x BCE/focal x confianza ponderada/alta. | Ocho variantes completas. |
| Grad-CAM de al menos tres hallazgos y discusión anatómica. | `scratch_bce_weighted/cam_0.png`, `cam_2.png`, `cam_7.png`; explicación crítica en el informe. | Mapas y discusión presentes. Se reconoce respuesta periférica/difusa y no se acredita localización clínica. |
| Prevalencia de doce clases y mitigación evaluada. | `class_balance.csv`, `prevalence.csv`, comparación focal/BCE y discusión por hallazgo. | Presente. Verificar interpretación de precisión/recall y soportes pequeños. |
| Media página de sesgo geográfico/lingüístico. | `reports/mri/discussion.tex` y sección de limitaciones del PDF. | Texto presente. El chat específico comprobará extensión y correspondencia con datos. |

Se seleccionaron 600 estudios: 434 de entrenamiento, 108 de validación débil y 58 de evaluación experta. La primera partición dejó los positivos débiles de Lateral OA solo en validación; se corrigió usando etiquetas débiles y se repitió la matriz. Los mismos 58 expertos se evaluaron en ambas ejecuciones. Esta limitación está divulgada y debe conservarse: no se debe describir la segunda evaluación como una prueba externa nueva y nunca observada.

La nueva auditoría MRI confirmó en CPU las 464 predicciones, la selección de checkpoints y los hashes de las 600 cachés. También identificó una brecha concreta para un clon limpio: `compose.mri.yaml` fija `RSNA_METADATA_DIR=/workspace/data`, pero `parcial/data/` está excluida de Git. El plan debe resolver la lectura o copia documentada de los CSV disponibles junto al dataset y poner el comando MRI directamente en el README. La reproducción desde GitHub todavía no está cerrada.

Evidencias principales:

- [Constancia de entrega local](E:/Kaggle/exam-output-mri/stratified-v2/audit/delivery.json).
- [Auditoría de resultados](E:/Kaggle/exam-output-mri/stratified-v2/audit/mri_results.json).
- [Informe borrador de ocho páginas](E:/Kaggle/exam-output-mri/stratified-v2/reports/mri_draft.pdf).
- [Notebook ejecutado](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/notebooks/01_mri.ipynb).
- [Guía y resultados](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/MRI_RUN.md).

## Pregunta 2: inundaciones

Salida: `E:/Kaggle/exam-output-flood`.

| Requisito del examen, pp. 3-5 | Evidencia local | Estado y pendiente |
|---|---|---|
| U-Net propia PyTorch con VV/VH. | `src/neural_exam/flood/model.py`, configuración congelada y checkpoints. | Implementado y ejecutado. |
| Manual vs manual más débil y comparación de pérdidas. | `flood/experiments.csv`: dos fuentes x BCE/Dice/combinada. | Seis variantes completas. |
| IoU por chip, F1, precisión, recall agua y matriz comentada ID. | Directorios de evaluación, tablas, figuras, `audit/flood-results.json`. | Presente; comprobar en revisión específica la convención de unión vacía. |
| Bolivia fuera de entrenamiento, partición oficial, brecha y dos hipótesis. | Manifiesto, auditoría, brechas y sección de generalización. | Presente. Las hipótesis no se presentan como causas demostradas. |
| Tres tipos de fallo SAR con ejemplos concretos. | `qualitative-statistics.json`, tres casos visuales y discusión. | Hay tres patrones observados. Revisar que la explicación física conserve sus límites. |
| Sentinel-2. | No incluido. | Bono opcional, sin incumplimiento obligatorio. |

La auditoría registra 618 predicciones: 88 chips ID y 15 de Bolivia por seis configuraciones. Nodata queda excluido. La mezcla de fuentes cambia también el número de actualizaciones; las conclusiones no deben atribuir diferencias exclusivamente a la calidad de etiquetas. Los 4.384 pares débiles publicados en el inventario consultado, frente a 4.385 mencionados en el examen, están documentados.

Evidencias principales:

- [Constancia de entrega local](E:/Kaggle/exam-output-flood/audit/delivery-verification.json).
- [Auditoría de resultados](E:/Kaggle/exam-output-flood/audit/flood-results.json).
- [Informe borrador de nueve páginas](E:/Kaggle/exam-output-flood/reports/flood_draft.pdf).
- [Notebook ejecutado](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/notebooks/02_flood.ipynb).
- [Guía y resultados](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/FLOOD_RUN.md).

## Pregunta 3: fauna

Salida: `E:/Kaggle/exam-output-wildlife`.

| Requisito del examen, pp. 5-7 | Evidencia local | Estado y pendiente |
|---|---|---|
| YOLO ajustado, resolución/aumentos documentados y vacíos como fondo. | Pipeline, configuración, smoke, manifiesto y `class_map.csv`. | Preparado. Falta entrenamiento final y revisar la limitación de ausencia de `empty` en train oficial. |
| mAP50 y mAP50:95 por especie y desbalance. | Código de evaluación; no hay resultados finales. | Falta ejecución científica. |
| Cámaras vistas/nuevas, brecha y evidencia sobre fondo. | Particiones auditadas, exclusión de secuencias y CAM diagnóstico de smoke. | Sin métricas ni análisis del modelo final. |
| Oclusión baja/media/alta y tamaño relativo. | Código y procedimiento de revisión. | Falta muestra revisada de oclusión y resultados por grupos. |
| Monitoreo ecológico y demostración con cámara nueva. | Módulo propio preparado. | Falta demostración ejecutada y reporte tabular final. |
| Media página de uso responsable. | `reports/wildlife/discussion.tex`. | Texto general presente; discusión empírica pendiente. |

El CSV `wildlife/label_review.csv` contiene 90 imágenes con estado `pending`. `label_review_status.json` confirma `human_review_pending`; `occlusion_review.csv` todavía no existe. El código impide entrenamiento completo mientras no se cierre esa revisión. Este bloqueo es un control añadido por la implementación, no una cantidad impuesta por el examen.

El benchmark ECCV usado contiene 14 categorías animales; la colección completa descrita en el enunciado es mayor. El chat específico debe comprobar la justificación del subconjunto, el mapeo de categorías y el tratamiento de fotografías vacías sin incorporar validación/prueba al entrenamiento.

El notebook fuente de fauna tiene 12 celdas sin ejecutar. La copia externa tiene 12 ejecutadas sin errores, pero acredita preparación y smoke. No hay directorio de informes con PDF científico ni resultados finales que permitan declarar contestados los entregables cuantitativos.

Evidencias principales:

- [Estado de revisión de etiquetas](E:/Kaggle/exam-output-wildlife/wildlife/label_review_status.json).
- [Muestra pendiente](E:/Kaggle/exam-output-wildlife/wildlife/label_review.csv).
- [Manifiesto](E:/Kaggle/exam-output-wildlife/wildlife/manifest.json).
- [Notebook ejecutado de preparación](E:/Kaggle/exam-output-wildlife/notebooks/03_wildlife.ipynb).
- [Guía para continuar](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/WILDLIFE_RUN.md).

## Comprobaciones realizadas en esta revisión general

- Leídas las siete páginas del examen y la estructura inicial del ejemplo.
- Contados los PDF actuales: MRI ocho páginas; inundaciones nueve; fauna sin PDF científico.
- Verificados los nueve hashes de artefactos referidos en la constancia MRI: todos coinciden.
- Verificados los hashes de PDF, notebook externo, notebook del proyecto y auditoría de resultados de inundaciones: todos coinciden.
- Comprobados `best.pt` y `last.pt` en las ocho variantes MRI y las seis de inundaciones.
- Inspeccionados los logs persistidos: 32 pruebas MRI y 20 de inundaciones aprobadas.
- Comprobados los conteos de ejecución y errores de los notebooks; las copias MRI/inundaciones coinciden byte a byte.
- Leídas las fuentes de los informes y los controles de revisión de fauna. Inspeccionadas visualmente las páginas 1 y 8 renderizadas del PDF MRI; la inspección completa anterior está registrada en sus constancias. La nueva revisión visual completa queda a cargo de cada chat específico.
- Confirmados `requirements.txt`, README y remoto GitHub configurado. `git status` muestra el trabajo del parcial sin versionar en esta copia; no se ha certificado la publicación de la versión evaluada ni su ejecución desde un checkout remoto nuevo.

Estas comprobaciones no sustituyen la auditoría científica independiente, una nueva ejecución del pipeline ni revisiones humanas. No se cambiaron datos, checkpoints o informes científicos.

## Plan de cierre y paquete de presentación

1. Finalizar la evaluación y el plan de cada pregunta en sus chats. Resolver hallazgos concretos antes de volver a ejecutar experimentos que ya tienen evidencias completas.
2. Completar fauna: revisión del dataset, entrenamiento, evaluación cis/trans, análisis de tamaño/oclusión, Grad-CAM, actividad de cámara nueva y notebook científico.
3. Incorporar los cinco integrantes reales, revisión editorial y figuras/tablas definitivas. Conservar las limitaciones de cada protocolo; una revisión anatómica competente puede reforzar MRI, pero el examen exige discusión anatómica y no una certificación clínica.
4. Generar y revisar visualmente los tres PDF finales NeurIPS de 6-10 páginas, con al menos cinco referencias por informe y sin placeholders.
5. Publicar, con autorización de los autores, la versión del código y sus configuraciones. Probar el comando documentado por pregunta desde una copia nueva del repositorio, con datos locales y entorno especificado. GitHub debe permitir identificar exactamente la versión que produjo los entregables.
6. Entregar los tres PDF y enlace GitHub en la plataforma del curso y preparar la presentación del 30 de octubre. El enunciado no especifica duración ni exige diapositivas; confirmar esas condiciones por el canal del curso si se necesitan.

En GitHub deben quedar código, README, dependencias, configuraciones, comandos, notebooks y las evidencias pequeñas necesarias para seguir las cifras. Mantener imágenes originales, grandes cachés y checkpoints voluminosos fuera de Git; documentar su ubicación y obtención/reproducción. La política de uso del dataset debe respetarse al decidir qué ejemplos publicar.

## Evaluaciones y planes por chat

| Pregunta | Chat creado | Documento de evaluación y plan |
|---|---|---|
| MRI | `01a109cf-6501-7413-8fb7-9c243dba5e1c` | `parcial/audits/revision-2026-10-04/01_mri.md` |
| Inundaciones | `01a109cf-6b01-74f1-9fed-f58d7ee5f945` | `parcial/audits/revision-2026-10-04/02_flood.md` |
| Fauna | `01a109cf-6e91-74c2-9509-4b0e0b830dc2` | `parcial/audits/revision-2026-10-04/03_wildlife.md` |

Los tres chats iniciaron su propia auditoría. Sus documentos pueden no existir todavía al consultar esta fotografía inicial.

## Herramientas de revisión

Se aplicaron los skills locales de PDF, PyTorch, scikit-learn y diseño de documentación. La referencia solicitada por el skill de evaluación se registra aquí como herramienta de trabajo, no como evidencia experimental: Kassis, T., Agarwal, V., He, Y., Patel, D., y Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. [arXiv:2609.00065](https://doi.org/10.48550/arXiv.2609.00065), versión vigente consultada el 4 de octubre de 2026.
