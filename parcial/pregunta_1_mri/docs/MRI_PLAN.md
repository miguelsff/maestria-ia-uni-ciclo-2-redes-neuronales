# Pregunta 1: plan previo a la ejecución

Registro histórico del plan previo: ejecutar y validar exclusivamente MRI en local, conservar los cambios existentes y producir un borrador NeurIPS. La restricción de publicación y commits correspondía a esa etapa. El usuario autorizó posteriormente implementar el plan de cierre de la auditoría; [MRI_DELIVERY.md](MRI_DELIVERY.md) describe esa entrega y su publicación.

## Protocolo fijado

1. Auditar 4.407 estudios, correspondencia con carpetas y descriptores; seleccionar 600 estudios con semilla 42. Reservar los 58 expertos para evaluación final; distribuir los 542 débiles entre entrenamiento y validación. Comprobar identificadores DICOM y hashes de todas las instancias seleccionadas, decodificación, geometría y cobertura textual. Documentar la limitación de los identificadores de paciente anonimizados.
2. Reutilizar la imagen compatible verificada (Python 3.12.3, PyTorch 2.10.0+cu128, torchvision 0.25.0+cu128). Configuración y Compose exclusivos; originales en solo lectura y salidas en E:/Kaggle/exam-output-mri. Límites: 8 CPU, 12 GB RAM y 2 GB shm. Auditar consumo y no iniciar entrenamiento concurrente en GPU.
3. Descargar únicamente pesos oficiales ResNet-50 ImageNet V2 y estilo NeurIPS 2026, con procedencia y SHA-256. Validar CUDA/gradientes, decodificadores, pruebas de pérdidas, separación, métricas, Grad-CAM y reanudación. Ejecutar smoke separado; no interpretarlo como resultado científico.
4. Ejecutar secuencialmente CNN propia/ResNet-50 × BCE/focal × ponderación/filtro de confianza. AdamW, lr 0,0003, weight decay 0,0001, batch 4, ocho stacks de tres cortes 224×224, hasta 50 épocas y paciencia 10. Elegir checkpoint por pérdida de validación débil; umbral 0,5 fijado. Congelar configuración, código y hashes antes de la matriz. Cualquier corrección del protocolo se registra antes de resultados.
5. Evaluar los 58 expertos, conservar métricas indefinidas, auditar predicciones/checkpoints y mostrar Grad-CAM de LCA, menisco medial y derrame. Ejecutar el notebook con entrenamiento desactivado. Completar método, resultados, sesgo, limitaciones y referencias verificadas; compilar y revisar todas las páginas del PDF.
6. Actualizar estado y guía MRI con comandos para ejecutar, reanudar y detener, rutas de evidencia y pendientes reales. Autores, correos y revisión del grupo no se inventan.

## Estado inicial verificado

- Código, notebook e informe preparados sin resultados MRI.
- Originales presentes en E:/Kaggle/rsna-knee-abnormality-detection; disponibilidad no equivale a integridad.
- RTX 5090 de 32 GB visible; sin proceso de entrenamiento detectado al comienzo. Docker dispone de aproximadamente 30 GiB de memoria. La RAM libre del host es menor que el límite del servicio y se vigilará el consumo efectivo.
- Pregunta 2 completa científicamente y pregunta 3 pendiente de revisión humana; no se modifican ni ejecutan.

## Unidad de trabajo y reversión

Verificación: auditoría de datos, pruebas relevantes, smoke y comparación de estados tras reanudación; resultados exactos se registrarán en MRI_RUN.md. La reversión se limita a archivos MRI nuevos y cambios en parcial/src/neural_exam/mri/, parcial/tests/test_mri.py, parcial/notebooks/01_mri.ipynb, parcial/reports/mri/ y enlaces de estado. Las salidas son independientes; no se alteran originales ni resultados de otras preguntas.

## Corrección previa a resultados

La frecuencia para alpha de focal loss en alta confianza debe excluir las etiquetas rechazadas por el filtro. La implementación preparada contaba también los negativos de confianza 0,2. Se corrige antes de entrenar y se registra alpha en run.json. La estrategia ponderada conserva su cálculo original. No se modifica el extractor ni se consultan resultados expertos para esta decisión.

La lectura de archivos pequeños sobre el montaje Windows mostró espera dominante en p9_client_rpc. Se introduce una copia derivada únicamente de las 600 series seleccionadas en el volumen neural-exam-mri_dicom_cache. La copia se transmite desde Windows, con hashes SHA-256 de origen, y se valida byte a byte dentro de Docker. Los originales siguen montados en solo lectura; no se descarga ni se extrae otra vez el archivo original del dataset. Esta optimización no altera selección, imágenes, etiquetas ni parámetros.

La inspección de normalización detectó promoción a float64 por los escalares NumPy de los percentiles. Se fuerza salida float32 y se cambia la versión de la identidad de caché para evitar reutilizar tensores incompatibles con las convoluciones. Se añade una prueba DICOM sintética que verifica tipo, geometría y rango; la corrección precede a smoke y a todo entrenamiento científico.

La comprobación de batch completo detectó que las etiquetas débiles sin abstenciones se convertían en tensores enteros. BCE requiere objetivos flotantes; se fuerza float32 en MRIDataset y se cubre el caso real con una prueba. No se cambia el significado de ninguna etiqueta.

## Corrección de partición tras la primera matriz

La revisión posterior detectó que la asignación por rareza enviaba todos los positivos débiles de Lateral OA a validación (0 train/14 val), y casi todos los de Medial OA (3 train/44 val). Faltó comprobar este requisito antes de entrenar. La primera matriz, sus predicciones y notebook se conservan en la raíz original y quedan identificados como protocolo sustituido para la entrega.

La corrección utiliza solo las etiquetas débiles de los mismos 542 estudios: minimiza desviación respecto a cuotas aproximadas de 20 % de positivos por etiqueta en validación, impidiendo retirar el último positivo de entrenamiento. Se verifica soporte positivo de las doce etiquetas en ambas particiones antes de cualquier nuevo entrenamiento. No se cambia la selección de los 600 estudios, los 58 expertos, extractor, umbral, pérdidas, seed ni hiperparámetros. No se utiliza el rendimiento experto para decidir cuotas o modelos. La segunda matriz se registra por separado en E:/Kaggle/exam-output-mri/stratified-v2. La evaluación debe reconocer que el conjunto experto también se utilizó en la ejecución inicial; no se presenta como un nuevo conjunto externo nunca consultado.
