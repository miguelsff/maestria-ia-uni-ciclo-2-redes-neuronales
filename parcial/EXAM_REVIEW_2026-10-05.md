# Estado de las tres preguntas del examen parcial

Revisión del 5 de octubre de 2026, America/Lima. **MRI e inundaciones están resueltas y validadas técnicamente, con autoría y código publicado. Fauna sigue pendiente de entrenamiento científico. No hay constancia de entrega de ninguna pregunta en la plataforma del curso.**

Este documento actualiza la fotografía del 4 de octubre. Las auditorías históricas conservan su valor como evidencia, pero sus pendientes de autores, publicación y ruta de CSV deben leerse junto con las constancias actuales.

## Estado actual

| Etapa | Pregunta 1: MRI | Pregunta 2: inundaciones | Pregunta 3: fauna v2 |
|---|---|---|---|
| Respuesta e implementación | Completa técnicamente; siete entregables desarrollados. | Completa técnicamente; cinco entregables obligatorios desarrollados. | Pipeline preparado; entregables científicos incompletos. |
| Entrenamiento científico | Ocho variantes terminadas y preservadas. | Seis variantes terminadas y preservadas. | No iniciado. |
| Evaluación científica | 464 predicciones auditadas; métricas de doce etiquetas. | 618 GeoTIFF auditados; ID y Bolivia. | Sin métricas finales cis/trans, por especie, tamaño u oclusión. |
| Pruebas y controles | 32 pruebas CPU desde clon limpio, según logs y constancia. | 20 pruebas originales; 17 CPU aprobadas y tres CUDA omitidas en la comprobación del paquete publicado. | 32 pruebas enfocadas; CUDA/AMP/NMS, smoke y reanudación verificados. Una prueba lenta excluida y reanudación GPU comprobada por separado. |
| Notebook | Ocho celdas ejecutadas, cero errores; copia local de entrega. | Ocho celdas ejecutadas, cero errores; salida científica original. | Doce celdas ejecutadas, cero errores, pero muestran preparación y pendientes. |
| Informe | PDF técnico de nueve páginas, siete referencias y cinco autores. | PDF de revisión de nueve páginas, siete referencias y cinco autores. | Sin PDF científico. |
| GitHub | Rama `codex/mri-delivery` publicada; comprobada desde clon limpio. | Rama `codex/flood-delivery` publicada; comprobada desde clon limpio. | Preparación v2 sin publicación de una entrega científica. |
| Aprobación del grupo | Pendiente. | Pendiente. | Pendiente tras producir resultados e informe. |
| Plataforma y presentación | Sin constancia de envío; presentación pendiente. | Sin constancia de envío; presentación pendiente. | Pendientes. |

Una prueba funcional o un notebook ejecutado no acreditan entrenamiento científico. No se volvió a entrenar ni se repitieron pruebas del proyecto durante esta revisión general.

## Pregunta 1: MRI

El problema de reproducción detectado el 4 de octubre quedó corregido: Compose lee los CSV desde `/datasets/rsna`, junto a los originales. El comando está documentado en README. La constancia registra selección idéntica de 600 estudios desde un clon sin `parcial/data/` y 32 pruebas CPU aprobadas. No se repitieron las ocho variantes desde ese clon; se conservaron las corridas científicas y se comprobaron empaquetado, selección y pruebas.

El PDF vigente tiene nueve páginas, cinco autores reales, siete referencias, resultados, ablaciones, desbalance, ruido textual, discusión de sesgo y tres Grad-CAM. La revisión anatómica profesional no es un requisito expreso del examen. La discusión reconoce activación periférica/difusa y no presenta los mapas como localización clínica demostrada.

Los mismos 58 estudios expertos se evaluaron antes y después de corregir la partición débil. Esa limitación sigue declarada: el segundo resultado no constituye una prueba externa nueva nunca observada.

Evidencias actuales:

- [PDF técnico para aprobación del grupo](E:/Kaggle/exam-output-mri/submission-2026-10-04/reports/mri.pdf).
- [Notebook ejecutado de entrega](E:/Kaggle/exam-output-mri/submission-2026-10-04/notebooks/01_mri.ipynb).
- [Constancia de entrega técnica](E:/Kaggle/exam-output-mri/submission-2026-10-04/audit/final_delivery.json).
- [Guía de entrega](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/MRI_DELIVERY.md).
- [Código publicado](https://github.com/miguelsff/maestria-ia-uni-ciclo-2-redes-neuronales/tree/codex/mri-delivery).

Falta aprobación editorial real del grupo, confirmar las condiciones de acceso/uso de Kaggle exigidas por el enunciado, comprobar que el docente pueda acceder al repositorio, entregar PDF y enlace por la plataforma y realizar la exposición. El guion de defensa ya está preparado.

## Pregunta 2: inundaciones

Las seis configuraciones, la evaluación ID/Bolivia, la matriz de confusión, los tres patrones de fallo SAR y las dos hipótesis de generalización están contestados. Bolivia permanece fuera de entrenamiento y selección. El informe mantiene los límites de una semilla, quince chips externos y presupuestos de actualizaciones distintos al añadir débiles.

La autoría y afiliación ya están completas. El PDF vigente de revisión tiene nueve páginas y siete referencias. El paquete publicado fue comprobado desde una copia nueva, con hashes, Compose, importaciones y pruebas CPU. No se volvió a entrenar la matriz completa para publicar.

Evidencias actuales:

- [PDF de revisión con autoría](E:/Kaggle/exam-delivery-flood/review-2026-10-04/reports/flood_draft.pdf).
- [Notebook científico ejecutado](E:/Kaggle/exam-output-flood/notebooks/02_flood.ipynb).
- [Auditoría científica de resultados](E:/Kaggle/exam-output-flood/audit/flood-results.json).
- [Verificación de revisión y paquete](E:/Kaggle/exam-delivery-flood/review-2026-10-04/audit/review-verification.json).
- [Verificación de la publicación vigente](E:/Kaggle/exam-delivery-flood/review-2026-10-04/audit/publication-head-verification.json).
- [Código publicado](https://github.com/miguelsff/maestria-ia-uni-ciclo-2-redes-neuronales/tree/codex/flood-delivery).

Falta aprobación editorial real del grupo, compilar la versión final después de esa aprobación, entregar PDF y enlace en la plataforma y presentar el trabajo. Sentinel-2 es bono opcional y no bloquea el cumplimiento obligatorio.

## Pregunta 3: fauna v2

La preparación actual está en `E:/Kaggle/exam-output-wildlife-v2`; la anterior se conserva. Se añadieron 1.000 fotografías vacías oficiales de ocho cámaras adicionales, sin trasladar imágenes de las particiones reservadas a entrenamiento. La metodología documenta la ampliación, las 14 categorías animales del benchmark y sus limitaciones; no la presenta como réplica estricta del train ECCV original.

El conjunto y el entorno están preparados y las verificaciones funcionales pasaron. El bloqueo real ocurre antes de cargar el detector: **las 90 filas de revisión de etiquetas siguen pendientes**. El modelo científico todavía no se entrenó.

La galería de oclusión contiene **17.620 objetos**, todos sin nivel, evaluador ni evidencia registrada. Según el protocolo v2 se exige cobertura completa y al menos 30 objetos por nivel. Esa cantidad y cobertura son decisiones del protocolo implementado: el examen exige evaluar oclusión baja/media/alta, pero no impone esos números. La revisión de oclusión no es el bloqueo inmediato del entrenamiento; permanece necesaria para cerrar el análisis y el informe de este protocolo.

Faltan entrenamiento YOLOv8s, evaluación cis/trans y por especie, brecha, robustez por tamaño y oclusión, Grad-CAM del modelo final, demostración ecológica con cámara nueva, notebook de resultados, PDF NeurIPS de 6-10 páginas, publicación, aprobación y entrega.

Evidencias actuales:

- [Estado de fauna v2](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/WILDLIFE_STATUS.json).
- [Guía actual para continuar](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/WILDLIFE_V2_RUN.md).
- [Constancia de verificaciones funcionales](E:/Kaggle/exam-output-wildlife-v2/receipts/final-verification.json).
- [Constancia del bloqueo científico](E:/Kaggle/exam-output-wildlife-v2/receipts/scientific-gate.json).
- [Galería de las 90 etiquetas](http://127.0.0.1:8894/label_review.html).
- [Galería de oclusión](http://127.0.0.1:8894/occlusion_review.html).

Las dos galerías respondieron HTTP 200 durante esta revisión. Ese control no constituye una aprobación visual de sus imágenes ni garantiza que el servidor siga activo posteriormente.

## Comprobaciones de esta revisión

- Se relevaron los estados actuales y las guías de entrega; las tres auditorías específicas del 4 de octubre permanecen como antecedentes.
- Los seis artefactos referidos en la constancia actual de MRI existen y conservan sus hashes.
- Ambos PDF actuales conservan sus hashes registrados, tienen nueve páginas, siete referencias y los cinco nombres; no mantienen el placeholder de autoría.
- Se inspeccionaron las nueve páginas MRI mediante sus renderizados guardados. Se volvió a renderizar el PDF actual de inundaciones con la imagen local de reportes y se inspeccionaron sus nueve páginas; no se observaron recortes ni solapamientos en la vista de conjunto.
- Los notebooks locales de MRI e inundaciones tienen ocho celdas ejecutadas sin errores. El notebook de fauna tiene doce, con preparación y resultados científicos pendientes.
- El notebook y la auditoría científica original de inundaciones conservan sus hashes históricos.
- La consulta directa a GitHub confirmó las ramas: MRI `d185c4861481c27983fa01377b3b18723b210449`; inundaciones `3343729cf1de412938bf60f17e13667d217d491e`. Coinciden con las constancias de publicación.
- Se contaron las revisiones de fauna: 90 etiquetas pendientes y 17.620 objetos sin nivel de oclusión, sin evaluador ni evidencia. No hay PDF ni entrenamiento científico final.

La comprobación de ramas no reemplaza una reproducción científica completa desde GitHub. Las constancias distinguen pruebas de empaquetado/CPU de entrenamientos completos. Los conteos de pruebas aprobadas proceden de logs anteriores, que no se volvieron a ejecutar hoy.

## Cierre del examen

La prioridad técnica es completar fauna. En paralelo, el grupo puede aprobar y cerrar los dos informes ya preparados. Los cinco autores ya están configurados; no falta inventar correos ni conseguir una firma radiológica.

Según [el examen, páginas 1 y 7](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/examen-parcial.pdf), deben presentarse tres PDF NeurIPS de 6-10 páginas, al menos cinco referencias pertinentes por informe y los enlaces al código reproducible en GitHub, con comando de extremo a extremo y entorno documentados. El envío es por la plataforma del curso y la presentación en clase está fijada para el **30 de octubre de 2026**.

No se ha registrado aprobación editorial real, comprobante de envío a la plataforma ni presentación. Las ramas de MRI e inundaciones están publicadas por separado; se deben proporcionar enlaces que identifiquen la versión de cada pregunta y verificar acceso del profesor. La publicación de fauna sigue pendiente.
