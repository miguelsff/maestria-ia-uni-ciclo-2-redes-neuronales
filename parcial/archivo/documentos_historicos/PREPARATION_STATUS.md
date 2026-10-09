# Estado de preparación

## Actualización MRI: 3 de octubre de 2026

La pregunta 1 completó ocho experimentos científicos sobre 600 estudios, auditoría de 464 predicciones, notebook ejecutado y PDF borrador de ocho páginas. [MRI_RUN.md](../../pregunta_1_mri/docs/MRI_RUN.md) y MRI_STATUS.json registran el estado vigente, la corrección de la partición y la primera ejecución preservada. Faltan autoría real, revisión anatómica, revisión editorial y publicación del código. Las notas de preparación anteriores se conservan como historial.

## Actualización de inundaciones: cierre editorial del 4 de octubre de 2026

Los cinco nombres reales y la afiliación fueron proporcionados por el usuario e incorporados en la plantilla común. Se documentó directamente en README el comando completo de inundaciones y una salida nueva para reproducciones independientes. El PDF de revisión con autoría completa se compila en `E:/Kaggle/exam-delivery-flood/review-2026-10-04`, conservando los experimentos y el borrador original. La aprobación editorial del grupo, la entrega por la plataforma y la presentación continúan pendientes. El [material reproducible](https://github.com/miguelsff/maestria-ia-uni-ciclo-2-redes-neuronales/tree/codex/flood-delivery) está publicado en `codex/flood-delivery`, commit `3343729cf1de412938bf60f17e13667d217d491e`; 168 archivos coinciden con sus hashes desde una copia recuperada de GitHub. Compose e importaciones pasaron desde esa copia. La revisión funcional en CPU pasó 17 pruebas y omitió tres casos que requieren GPU. Las constancias nuevas se conservan en `E:/Kaggle/exam-delivery-flood/review-2026-10-04/audit/`.

### Ejecución científica del 3 de octubre de 2026 (registro preservado)

La pregunta 2 usa [FLOOD_RUN.md](../../pregunta_2_inundaciones/docs/FLOOD_RUN.md), configuración propia y salida `E:/Kaggle/exam-output-flood`. CUDA, 20 pruebas y smoke de las seis variantes pasaron. Se verificaron tamaño y MD5 de 9.216 archivos y se creó una caché Linux en solo lectura. Su manifiesto conserva exactamente la identidad de la preparación sobre originales.

Los conteos válidos son 251 manuales de entrenamiento, 86 de validación, 88 de prueba, 15 de Bolivia y 4.150 débiles. Se excluyeron 16 chips totalmente inválidos. Los seis experimentos científicos terminaron: IoU por chip entre 0,3251 y 0,4100 en ID y entre 0,2690 y 0,4790 en Bolivia. La supervisión débil mejora esa IoU en cero de tres pérdidas en ID y dos de tres en Bolivia; la diferencia de pasos y una semilla impiden atribuirlo únicamente a las etiquetas.

La auditoría final pasó para las 618 predicciones (88 ID + 15 Bolivia por variante): geometría original, nodata, probabilidades, conteos, métricas y checkpoint de mínima pérdida de validación. Se generaron tres casos visuales con estadísticas verificadas, tres patrones observados y dos hipótesis explícitamente no causales. El notebook tiene ocho celdas de código ejecutadas sin errores y entrenamiento desactivado; se conserva en el proyecto y en la salida externa. El PDF se compiló y sus nueve páginas se renderizaron e inspeccionaron; se corrigieron etiquetas superpuestas y un acento en modo matemático. La compilación final no presenta caracteres faltantes, referencias indefinidas ni cajas desbordadas.

Evidencia final en `E:/Kaggle/exam-output-flood/audit/flood-results.json` y `delivery-verification.json`; notebook en `notebooks/02_flood.ipynb` y PDF en `reports/flood_draft.pdf` dentro de la salida externa. La autoría real de cinco integrantes y la revisión editorial del grupo siguen pendientes; no se marca aprobación humana ni entrega final.

No se modificaron los originales, ni se iniciaron experimentos de MRI o fauna desde esta ejecución.

## Actualización de fauna: 3 de octubre de 2026

La pregunta 3 ya tiene implementación ejecutable, entorno construido y validación funcional. La guía vigente es [WILDLIFE_RUN.md](../../pregunta_3_fauna/docs/WILDLIFE_RUN.md). El estado del 2 de octubre que aparece abajo se conserva como registro de la preparación inicial.

- Compose independiente: montajes de fauna, originales en solo lectura, volúmenes Linux para enlaces y límites de recursos verificados.
- Anotaciones oficiales, YOLOv8s, auxiliar AMP y estilo NeurIPS descargados explícitamente, con recibos SHA-256 locales.
- CUDA, gradientes, NMS y AMP comprobados en RTX 5090 con PyTorch 2.10.0+cu128, torchvision 0.25.0+cu128 y Ultralytics 8.4.21.
- Las 57.864 imágenes referenciadas se decodificaron por completo y tienen SHA-256; no falta ninguna y no hay archivos extra en la raíz configurada.
- Inventario real, exclusiones, mapa de 14 clases y solapamientos de todas las particiones verificados. Después de retirar 270 fotos de cis_val no hay secuencias compartidas.
- Pruebas de fauna y utilidades ejecutadas; reanudación YOLO con interrupción, época siguiente y 183 estados de optimizador recuperados.
- Smoke real completado: 56 fotos de train, 31 de cis_val, una época; sin predicciones en tests. Grad-CAM produjo un mapa finito en cis_val.
- Notebook de fauna: 12 celdas de código ejecutadas sin errores en la copia externa de resultados. Entrenamiento completo desactivado.
- Galería de 90 etiquetas, ocho hojas de contacto y muestra persistida para revisión humana. Revisar las hojas permite comprobar visualmente el escalado sin convertir esa inspección del asistente en aprobación humana.

**Pendiente:** aprobación humana de etiquetas antes del entrenamiento científico; después, evaluación completa y revisión de oclusión con al menos 30 objetos por nivel. También faltan discusión empírica, cinco autores reales y PDF final de 6–10 páginas. Train oficial no contiene empty; no se importan fondos de validación o test. MRI e inundaciones no se ejecutaron ni procesaron en esta etapa.

| Unidad de fauna | Evidencia | Límite de reversión |
|---|---|---|
| Entorno y activos | Compose independiente, recibos de dependencias/descarga/GPU. | Cambiar únicamente configuración y código de fauna; conservar originales y activos descargados. |
| Auditoría y entrenamiento | Manifiesto, CSV, pruebas, resume-check y smoke externos. | Preservar checkpoints y revisión humana; no reutilizar otra configuración en la misma salida y volumen. |
| Entregables | Notebook ejecutado de auditoría, fuentes y generador de informes. | Mantener revisiones pendientes explícitas; no sustituir resultados científicos por smoke. |

No se hicieron commits, publicaciones ni cambios en los archivos originales de datos.

## Registro inicial

Fecha: 2 de octubre de 2026. Alcance: implementar y revisar archivos estáticamente, sin ejecutar experimentos.

## Preparado

- Paquete Python con tres proyectos y utilidades de configuración, entrenamiento, métricas, checkpoints, Grad-CAM, figuras y activos de informes.
- MRI: selección de hasta 600 estudios, reserva de expertos, reglas multilingües con negación/abstención, CNN 2.5D propia, ResNet-50 y ocho configuraciones.
- Inundaciones: U-Net propia VV/VH, exclusión de nodata, normalización de entrenamiento manual, seis configuraciones y Bolivia independiente.
- Fauna: conversión del benchmark oficial CCT20, vacíos como fondo, exclusiones auditadas, secuencias de validación depuradas, YOLOv8s, métricas COCO, oclusión manual y actividad ecológica.
- Compose con límites solicitados, GPU para entrenamiento y servicios separados de notebooks, descargas e informes.
- Comandos de diagnóstico, preparación, smoke, ejecución, reanudación, descarga explícita, revisión de oclusión y generación de tablas.
- Tres notebooks sin ejecutar, cinco archivos de pruebas y tres fuentes de informes NeurIPS con al menos cinco referencias citadas por informe.
- README general y guía local con rutas, descargas necesarias, pasos y gestión de resultados.

## Revisión estática realizada

- Análisis de sintaxis de los archivos Python, sin importarlos ni ejecutar sus funciones.
- Lectura estructural de pyproject.toml y configs/default.toml.
- Análisis JSON y sintaxis de celdas de los tres notebooks; execution_count permanece null y outputs vacío.
- Comprobación de imports internos y referencias bibliográficas; cada informe cita siete fuentes existentes en references.bib.
- Revisión y formato con Ruff 0.16.1 ya disponible en la caché local. La versión fijada para el entorno es 0.11.13 y su comprobación queda pendiente dentro de la imagen.
- Resolución estática de Compose con .env.example y todos sus perfiles; se comprobaron CPU, RAM, memoria compartida, montajes y reservas GPU. Esto no construyó ni arrancó contenedores.
- Revisión de correspondencia entre comandos de documentación y las interfaces preparadas.

La validación estática comprueba estructura y coherencia; no demuestra corrección numérica, rendimiento ni compatibilidad binaria.

## Pendiente de verificar y ejecutar

1. Finalización del ZIP MRI y extracción completa; disponibilidad de Sen1Floods11 y CCT20 con sus anotaciones y particiones.
2. Descarga explícita de pesos y estilo oficial, sin modificar el archivo MRI que está en progreso.
3. Construcción de Docker, resolución de dependencias y comprobación CUDA/NMS/AMP sobre la RTX 5090.
4. Ruff del entorno, pytest sintético y prueba de reanudación; ninguna prueba funcional se ha ejecutado en esta etapa.
5. Integridad real de DICOM/raster/imágenes, decodificadores comprimidos, cobertura por hallazgo, cajas normalizadas, exclusiones y particiones.
6. Pruebas pequeñas, ajuste de recursos si aparece falta de memoria y entrenamientos completos secuenciales.
7. Ejecución de notebooks y evaluación con checkpoints reales; las métricas actuales no existen.
8. Oclusión visual low/medium/high con soporte suficiente; revisión de fallos SAR y plausibilidad anatómica de Grad-CAM.
9. Discusión sustentada en resultados, nombres/correos de cinco integrantes y enlace real del repositorio.
10. Generación de tablas/figuras, compilación y revisión visual de tres PDF de 6–10 páginas.

No se presenta esta preparación como una entrega experimental terminada. No se han iniciado descargas de datasets, ejecutado pytest o notebooks, construido imágenes del proyecto, levantado sus servicios, entrenado modelos ni compilado informes.

## Unidades de trabajo y límites de reversión

| Unidad | Evidencia preparada | Límite de reversión |
|---|---|---|
| Entorno y comandos | Dockerfile, compose.yaml, configuración, requirements, CLI y documentación. | Revertir únicamente esos archivos; preservar datos y PDFs originales. |
| Implementaciones | Módulos MRI, flood y wildlife, utilidades y pruebas relacionadas. | Revertir el módulo junto con sus pruebas y documentación; no eliminar datasets ni checkpoints ajenos. |
| Entregables | Notebooks sin salidas, fuentes NeurIPS, generador de activos y guía. | Revertir fuentes nuevas; conservar material de referencia y resultados futuros fuera de Git. |

No se crearon commits ni se publicó el repositorio. La siguiente etapa debe verificar el entorno y los datos antes de producir resultados científicos.
