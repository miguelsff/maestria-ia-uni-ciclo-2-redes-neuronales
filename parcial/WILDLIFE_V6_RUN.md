# Pregunta 3: ajuste de YOLOv8s y compatibilidad JPEG

V6 usa el detector YOLOv8s existente de Ultralytics y conserva las 90 fotografías aprobadas por Miguel, incluidas las tres cajas de ave corregidas en V5. No requiere notas individuales en las fotografías aprobadas. La decisión de utilizar YOLO existente está documentada como instrucción del usuario, sin atribuir una confirmación al docente.

El intento V5 se interrumpió durante la lectura inicial: algunos JPEG tienen espacios después del marcador final. Ultralytics intenta reescribir esos archivos y los enlaces a originales de solo lectura provocaban que los descartara. V5 completó cero épocas; sus fuentes, registros y datos se conservan.

V6 genera copias de trabajo únicamente para ese caso: elimina espacios, tabulaciones y saltos de línea después de un marcador JPEG final existente. No vuelve a comprimir la imagen. Comprueba igualdad exacta de los píxeles decodificados y conserva hashes del original, la copia y los píxeles. Un marcador ausente o un contenido final distinto provoca un error. Los originales permanecen en solo lectura; el registro está en `wildlife/jpeg_materialization.json`.

Antes de entrenar, el comprobador de la versión instalada de Ultralytics verifica todas las imágenes y etiquetas de las cinco particiones. Rechaza imágenes omitidas, etiquetas ausentes y cajas eliminadas. La evidencia está en `receipts/yolo-loader-verification.json`. La migración de revisiones exige igualdad de fotografía y cajas y queda ligada al manifiesto padre y al hash del CSV humano en `wildlife/review_migration.json`.

Configurar una salida nueva con `.env.wildlife-v6.example`; conservar el token privado en `.env.wildlife-v6`. Desde `parcial/`:

```powershell
.\scripts\run_wildlife.ps1 -Variant v6
```

La salida actual es `E:/Kaggle/exam-output-wildlife-v6`; su padre preservado es V5. La [galería](http://127.0.0.1:8901/label_review.html) conserva las decisiones auténticas del CSV. El registro de correcciones V6 reaplica las cajas aprobadas sobre la anotación oficial sin introducir cambios de etiquetas frente a V5. Las particiones, secuencias y fotogramas de monitoreo se contrastan con el padre.

El protocolo conserva 14 clases, resolución 640, batch 16, semilla 42, AdamW con tasa inicial 3e-4 y weight decay 1e-4, hasta 50 épocas y paciencia 10. `cis_val` selecciona el checkpoint; `cis_test` y `trans_test` permanecen reservados. La ejecución calcula AP por especie, comparación entre cámaras conocidas y nuevas, tamaño relativo e intervalos con 200 remuestreos de secuencias; genera doce Grad-CAM y la demostración de monitoreo.

La preparación actual comprobó igualdad de los 48.254 registros, etiquetas y metadatos frente a V5 y conservó las 90 revisiones. Ultralytics aceptó las 48.254 imágenes sin perder fotografías ni cajas. Las 1.000 copias JPEG corresponden exclusivamente a los fondos añadidos. Se congelaron 44 archivos experimentales. Las pruebas de conversión, revisión, carga real, reanudación y entrega pasaron: 52 pruebas. Esa verificación técnica no sustituye los resultados de entrenamiento. La oclusión humana y la aprobación editorial siguen pendientes y se declaran en el borrador; no se generan juicios humanos automáticamente. Consultar `WILDLIFE_STATUS.json` y los registros de `receipts/` para el estado real.

Para reanudar una interrupción con checkpoint y las mismas fuentes congeladas:

```powershell
.\scripts\run_wildlife.ps1 -Variant v6 -Resume
```

Si cambian las fuentes experimentales, conservar esta salida y preparar otra independiente. Las preguntas 1 y 2 no forman parte de esta ejecución.

## Recuperación de los intervalos tras el límite de memoria

El entrenamiento V6 terminó con 39 épocas; `training_complete.json` conserva el hash del modelo seleccionado. Ambas evaluaciones de prueba y los resultados por tamaño se guardaron. El 5 de octubre de 2026 a las 22:24 UTC, el registro del kernel de Docker confirmó que el proceso de cálculo de intervalos superó el límite de 12 GiB del contenedor. Los intervalos parciales de esa ejecución no se habían guardado y no se declaran como resultados terminados.

Para continuar exclusivamente la evaluación y los entregables con ese modelo:

```powershell
.\scripts\recover_wildlife.ps1
```

La recuperación verifica las 44 fuentes congeladas, las versiones, el manifiesto, el checkpoint y la procedencia de las predicciones. Ejecuta cada grupo de intervalos en un proceso nuevo y guarda los grupos terminados en `wildlife/bootstrap_cache/`. Usa la función original sin cambiar la semilla 42, los 200 remuestreos de secuencias ni las métricas. Tres pruebas comprueban la igualdad exacta frente al cálculo original, la persistencia de resultados y el rechazo de un checkpoint modificado.

Después de los intervalos, continúa la demostración de monitoreo, los doce Grad-CAM, el notebook y el PDF borrador. No repite el entrenamiento ni completa revisiones humanas. Los registros son `receipts/recovery-evaluation-*.log` y `receipts/recovery-report-*.log`; el estado del contenedor queda preservado en `receipts/recovery-run-state-*.json` cuando termina el proceso. Si se interrumpe, el mismo comando reutiliza únicamente grupos con la identidad de entradas verificada.

## Entrega automática terminada

La recuperación terminó sin otro fallo de memoria: ocho grupos de intervalos con 200 réplicas cada uno, doce casos Grad-CAM, demostración de la cámara 7 con 1.853 fotografías y notebook con doce celdas ejecutadas sin errores. El PDF borrador tiene diez páginas; se revisaron sus diez renderizados. La discusión utiliza las métricas e intervalos reales. Tres mapas son planos y algunos rótulos de horario no coinciden con la apariencia de iluminación; esas limitaciones quedan explícitas, conservando el método y los casos congelados.

Los entregables están en `E:/Kaggle/exam-output-wildlife-v6/reports/wildlife_draft.pdf` y `E:/Kaggle/exam-output-wildlife-v6/notebooks/03_wildlife.ipynb`. La constancia de comprobación está en `tmp/pdfs/wildlife_v6_review/completion_validation.json`. Las tablas de resultados y actividad copiadas al informe se ligan al checkpoint verificado; si se generan resultados nuevos, deben actualizarse esas tablas y la discusión antes de recompilar.

Las 90 aprobaciones de etiquetas permanecen intactas. La oclusión registra cero niveles humanos de 17.620 objetos y la aprobación editorial del equipo continúa pendiente. La revisión completa es una decisión del protocolo actual, no una exigencia atribuida al docente. Una muestra de revisión necesitaría un protocolo separado, con cobertura y límites declarados; no modifica esta ejecución V6 congelada ni requiere otro entrenamiento. No se creó una aprobación final automática.
