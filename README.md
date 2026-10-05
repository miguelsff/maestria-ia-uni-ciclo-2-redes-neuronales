# Redes Neuronales y Aprendizaje Profundo — UNI

Esta rama contiene la entrega de código reproducible de la pregunta 1: clasificación de doce hallazgos MRI de rodilla con supervisión débil multilingüe. El paquete compartido se conserva completo para resolver sus imports; las corridas y entregas de otras preguntas no forman parte de esta rama.

Ocho experimentos sobre 600 estudios, 434 de entrenamiento, 108 de validación débil y 58 expertos. La evaluación conserva 464 predicciones auditadas. El PDF local tiene nueve páginas, cinco autores y siete referencias. La aprobación editorial del grupo y la entrega por la plataforma siguen pendientes.

## Reproducción

Con Python 3.12, Docker Desktop/WSL2, GPU NVIDIA compatible con CUDA 12.8 y los datos RSNA autorizados, desde `parcial/`:

```powershell
.\scripts\run_mri.ps1
```

`RSNA_HOST_DIR` señala la carpeta con `train.csv`, `train_series.csv` y `train_series/` (por defecto `E:/Kaggle/rsna-knee-abnormality-detection`). El runner lee metadatos desde los originales montados en solo lectura, sin requerir `parcial/data/`. Para una reproducción independiente, configurar una salida nueva mediante `MRI_OUTPUT_DIR`; el valor predeterminado corresponde a la ejecución local histórica y no debe reutilizarse para una corrida nueva.

El comando construye el entorno, verifica series/caché y CUDA, ejecuta pruebas y una corrida pequeña, entrena/evalúa ocho variantes, audita resultados y genera notebook y PDF borrador. [Prerrequisitos y comando completo](parcial/README.md), [historia experimental](parcial/MRI_RUN.md), [entrega y evidencia](parcial/MRI_DELIVERY.md), [guion de defensa](parcial/MRI_DEFENSE.md).

## Alcance de la evidencia

El paquete sin metadatos locales reprodujo exactamente la selección de los 600 estudios y sus particiones. Las 32 pruebas CPU pasaron, incluida la reanudación sintética. No se repitió la matriz científica durante el cierre. El código histórico de las corridas se conserva localmente; la versión actual añade el cierre del estado de entrenamiento sin cambiar la optimización.

Los mismos 58 expertos se consultaron en la ejecución inicial y la partición corregida. No constituyen una nueva evaluación externa nunca consultada. El umbral es 0,5, el filtro se aplica por etiqueta y los mapas Grad-CAM no tienen validación clínica. Consultar el informe y las limitaciones antes de interpretar las cifras.

El notebook publicado tiene salidas vacías para excluir imágenes clínicas. Su copia local ejecutada conserva las ocho celdas sin errores. Datos, textos clínicos, predicciones individuales, pesos, figuras clínicas y PDF permanecen fuera de Git. Las fuentes del informe y las dependencias fijadas permiten generar los entregables desde salidas auditadas. El repositorio es privado y el grupo debe comprobar el acceso del docente.
