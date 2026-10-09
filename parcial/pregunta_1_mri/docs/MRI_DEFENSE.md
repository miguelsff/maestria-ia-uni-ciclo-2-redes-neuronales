# Defensa de la pregunta 1: MRI de rodilla

Guion para la exposición del grupo el 30 de octubre de 2026. La duración de diez minutos es una propuesta de ensayo, no una condición del enunciado. El grupo debe confirmar el reparto y revisar el PDF antes de presentarlo.

## Recorrido propuesto

| Tiempo | Tema | Evidencia para mostrar |
|---|---|---|
| 0–2 min | Problema, muestra y particiones | PDF pp. 1–2: doce hallazgos, 434 train, 108 validación débil y 58 expertos. |
| 2–4 min | Extractor multilingüe y ruido | PDF pp. 2 y 4: negaciones, abstenciones, pesos 0,9/0,2 y acuerdo condicionado por cobertura. |
| 4–6 min | Diseño y selección | PDF pp. 2–3: CNN 2.5D propia, ResNet-50 ImageNet, ocho variantes y selección por pérdida de validación débil. |
| 6–8 min | Métricas, ablaciones y desbalance | PDF pp. 3–5 y 7: AUC de doce etiquetas, F1 al umbral 0,5, efectos pareados y prevalencia. |
| 8–10 min | Grad-CAM, sesgo y alcance | PDF pp. 5–6 y 8: tres hallazgos, mapas difusos/periféricos y límites de la evaluación. |

## Explicaciones que deben quedar claras

La etiqueta no mencionada recibe supervisión negativa débil, no una negación verificada. La incertidumbre y las contradicciones producen abstención. El léxico contiene vocabulario de trece idiomas, pero no demuestra la misma calidad de interpretación en todos ellos. El grupo de escritura tampoco identifica idioma, centro o geografía.

El filtro actúa por etiqueta y conserva los 434 estudios de entrenamiento y 108 de validación. Para sinovitis deja 140 etiquetas observadas en entrenamiento, todas positivas. Una mayor confianza textual puede reducir la diversidad de supervisión y eliminar negativos explícitos.

La partición inicial dejó Lateral OA sin positivos débiles en entrenamiento. La corrección utilizó únicamente etiquetas débiles y repitió la matriz. Los mismos 58 expertos se consultaron en ambas versiones, por lo que esta evaluación no constituye una nueva prueba externa nunca consultada. El léxico, el umbral y los hiperparámetros permanecieron fijados.

El baseline CNN/BCE/ponderación obtiene AUC macro 0,5776 y F1 macro 0,3610. El mayor AUC observado es 0,6199 para ResNet/focal/ponderación, con F1 0,5056. Estas cifras describen la comparación y no seleccionan un modelo para despliegue. Los doce AUC están definidos porque cada etiqueta experta tiene ambas clases.

Focal puede elevar recall y reducir precisión. En Medial OA, ResNet con ponderación pasa de recall 0,133 a 0,867, mientras la precisión baja de 0,667 a 0,325. En Lateral OA mejora F1 al umbral fijo, pero el AUC baja de 0,683 a 0,644. La mitigación no mejora uniformemente todas las métricas.

Los mapas de LCA, menisco medial y derrame pertenecen a positivos expertos con probabilidad superior a 0,5. El acierto de clasificación no valida la localización. Los mapas muestran señal difusa o periférica y no hay regiones anotadas ni evaluación profesional de localización. La interpolación de 14×14 a 224×224 no agrega precisión anatómica.

## Preguntas de defensa

**¿Por qué 0,5?** Es un baseline simétrico fijado antes de evaluar; no se optimizó con los expertos. Las probabilidades no están calibradas y este umbral no constituye una recomendación clínica.

**¿Cómo se eligió cada checkpoint?** Por pérdida mínima de validación débil. No se ajustaron léxico, umbral ni hiperparámetros según los expertos. La repetición de la evaluación sigue siendo una limitación divulgada.

**¿Se puede garantizar independencia de pacientes?** Se separaron StudyInstanceUID y no se encontraron PatientID DICOM compartidos entre particiones. La anonimización no demuestra que personas reales distintas no tengan varios estudios.

**¿El acuerdo filtrado de Lateral OA es perfecto?** El desacuerdo observado es cero sobre un único caso comparable de 58. Ese soporte no demuestra fiabilidad general.

**¿Cómo se verificaron las métricas?** Se recalcularon desde las 464 predicciones persistidas y se comprobaron los checkpoints contra las curvas. La auditoría no repitió inferencia.

**¿Qué haría falta para uso clínico?** Datos expertos independientes, evaluación externa por idioma/centro, incertidumbre estadística, calibración y evaluación de localización con regiones anotadas y profesionales. Este ejercicio no acredita esas condiciones.

## Ensayo y entrega

Cada integrante debe poder explicar las limitaciones aunque exponga otra sección. Preparar el PDF y el enlace al commit publicado, comprobar el acceso del docente al repositorio privado y conservar el comprobante de la plataforma. Las obligaciones personales de Kaggle y la revisión de condiciones de uso deben confirmarlas los integrantes; este guion no certifica su cumplimiento.
