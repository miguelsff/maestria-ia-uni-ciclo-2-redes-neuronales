# Pregunta 3: preparación v3 y revisión necesaria

**Registro histórico.** El revisor confirmó globalmente las 89 fotografías correctas y decidió que no requieren notas individuales. El flujo actual y la nueva propuesta para la tercera ave están en [WILDLIFE_V4_RUN.md](WILDLIFE_V4_RUN.md). Las exigencias de notas descritas abajo corresponden al protocolo anterior.

V3 implementa la corrección propuesta de etiquetas, la migración de revisiones auténticas, la reanudación con paciencia y la evaluación con incertidumbre. **El entrenamiento científico permanece pendiente de revisión humana.** No hay métricas finales ni PDF científico. La salida v2 se conserva íntegra; v3 escribe en `E:/Kaggle/exam-output-wildlife-v3`.

## Continuar desde la galería

Abrir la [galería v3 de etiquetas](http://127.0.0.1:8895/label_review.html). Se conservan las 89 decisiones de Miguel sobre fotografías cuyas cajas y categorías no cambiaron. La foto `596bdb10-23d2-11e8-a6a3-ec086b02610b` vuelve a `pending`: se propone ampliar la caja del ave de la izquierda y añadir la segunda ave visible a la derecha. La propuesta es del agente y requiere una nueva inspección humana; no constituye una corrección aprobada. El comentario original se preserva en `wildlife/parent_label_review.csv`.

Las 90 filas requieren evaluador y evidencia visual escrita. Las 89 aprobaciones existentes carecen de esa evidencia y no se rellena automáticamente. El revisor debe describir lo observado, corregir cualquier estado que considere necesario y exportar el CSV. Guardarlo como `E:/Kaggle/exam-output-wildlife-v3/wildlife/label_review.csv`. Las filas modificadas reciben el nombre ingresado por el revisor; las demás conservan su autoría.

La galería se reconstruye desde el CSV real al repetir el comando. No exportar el HTML histórico de v2 sobre la nueva salida: su contenido guardado precede a las revisiones actuales. Una fila `needs_changes` requiere resolver el problema con otra identidad y salida si cambia las etiquetas congeladas.

## Resolver el requisito de arquitectura

El [enunciado](../../compartido/referencias/examen-parcial.pdf) exige componentes principales de arquitectura implementados por estudiantes; la pregunta 3 pide ajuste fino de YOLOv8/YOLOv9. La implementación actual utiliza el detector completo de Ultralytics. Los módulos propios de datos, métricas, Grad-CAM y monitoreo no son componentes principales de la red. El material de clase muestra una implementación didáctica de YOLO y un flujo de ajuste fino, pero eso no acredita una excepción del docente para esta entrega.

Antes del entrenamiento definitivo debe existir una decisión humana documentada en `E:/Kaggle/exam-output-wildlife-v3/receipts/architecture-review.json`. El control acepta `reviewed: true`, un `reviewer` real, `evidence` verificable y una `decision`:

| Decisión | Evidencia necesaria | Consecuencia |
|---|---|---|
| `pretrained_detector_permitted` | Aclaración real del docente que permita este ajuste fino | Mantener el protocolo YOLOv8s preparado |
| `student_components_verified` | Componentes principales propios, archivos y verificaciones inspeccionados por el revisor | Revisar el protocolo y usar otra salida si cambia el código experimental |

No se creó esa constancia ni se atribuyó una aprobación al docente. El informe y el notebook declaran el requisito pendiente.

## Un comando para preparar o continuar

Desde `parcial/`, con Docker Desktop/WSL2 y los originales disponibles, copiar `.env.wildlife-v3.example` a `.env.wildlife-v3` y definir un token privado. El ejemplo reutiliza la cohorte original de fondos en solo lectura y señala la salida v2 como `PARENT_OUTPUT_HOST_DIR`. Para una reproducción independiente, elegir una **salida nueva** y puertos disponibles.

```powershell
.\scripts\run_wildlife.ps1 -PrepareOnly
```

El comando prepara automáticamente los metadatos y pesos de v2 cuando la salida está vacía, reconstruye el dataset derivado en un volumen independiente, migra únicamente revisiones de objetivos idénticos, actualiza las galerías, audita contra el manifiesto padre y congela las fuentes. Si una salida ya está congelada, comprueba sus fuentes **antes** de modificarla. El código actual rechaza reutilizar v2 por esa comprobación; su snapshot histórico sigue en `receipts/source-freeze/`.

Después de completar las dos revisiones previas:

```powershell
.\scripts\run_wildlife.ps1
```

Comprueba ambos controles antes de cargar pesos, comprueba disponibilidad de GPU y CUDA/AMP, entrena, evalúa, genera mapas y actividad, ejecuta el notebook y compila un borrador. `-Resume` recupera una interrupción compatible o reutiliza un entrenamiento terminado. Una interrupción exige que `resume_state.json` coincida con el hash de `last.pt`; recupera también mejor época, mejor fitness, paciencia y proximidad a la parada. No se garantiza equivalencia bit a bit.

## Protocolo conservado y análisis nuevos

| Tema | Regla |
|---|---|
| Detector | YOLOv8s, 14 categorías animales, semilla 42, 640, batch 16, hasta 50 épocas, paciencia 10 |
| Optimización | AdamW, tasa inicial 0,0003, weight decay 0,0001; checkpoint elegido por mAP50:95 en cis_val depurado |
| Corrección propuesta | Una fotografía de train; se verifica SHA-256 y cajas originales antes de aplicar el registro |
| Particiones | IDs, secuencias, cámaras, holdouts y fotogramas de monitoreo idénticos a v2 |
| Generalización | Brecha global y comparación sobre categorías con soporte en ambos tests; AP sin referencia permanece indefinida |
| Incertidumbre | 200 remuestreos de secuencias completas, percentiles del 95 %, modelo fijo; global, tamaño y oclusión aprobada |
| Oclusión | 17.620 objetos de trans_test, revisión completa y mínimo 30 por nivel; visibilidad incierta o truncamiento como `ambiguous` |

La [galería de oclusión](http://127.0.0.1:8895/occlusion_review.html) requiere evidencia y evaluador por objeto; sigue pendiente. No impide entrenar una vez revisadas etiquetas y arquitectura, pero impide cerrar el análisis de oclusión y el PDF final. Baja: >75 % visible; media: 25–75 %; alta: <25 %. El tamaño no mide oclusión.

La ampliación conserva 1.000 fondos de ocho cámaras adicionales sin positivos de esas cámaras, y fotos de resolución original frente a positivos reducidos. Es entrenamiento ampliado, con posibles sesgos de cámara y resolución; no es una reproducción estricta del train ECCV. Los intervalos son condicionales a las cámaras observadas y una semilla. No miden incertidumbre entre entrenamientos ni regiones nuevas. El tiempo y consumo del remuestreo científico completo permanecen por medir.

## Verificación y límites de entrega

Las pruebas y constancias actuales están en [WILDLIFE_STATUS.json](../evidencias/WILDLIFE_STATUS.json) y `receipts/` de v3. La prueba de reanudación utiliza imágenes sintéticas, completa tres épocas y comprueba época, 183 estados de optimizador y paciencia. No reemplaza el entrenamiento científico. El notebook tiene entrenamiento desactivado y muestra únicamente evidencia persistida.

| Unidad de trabajo | Evidencia | Límite de reversión |
|---|---|---|
| Correcciones y revisión | Pruebas de cajas restringidas a train, migración sin inventar autoría y control de evidencia; preparación real y auditoría contra v2 | Desactivar el protocolo v3 y conservar su salida; v2 y su snapshot no se reescriben |
| Reanudación e incertidumbre | Pruebas de hash/paciencia y bootstrap; interrupción GPU sintética real. Incertidumbre científica pendiente de checkpoint | Retirar callbacks y remuestreo en otra salida, sin modificar una corrida congelada |
| Notebook e informe | Ejecución de evidencia disponible; compilador probado para rechazar ausencia de activos científicos | Revertir únicamente la documentación de fauna y su notebook; no alterar informes de preguntas 1/2 |

Tras entrenamiento, completar la revisión de oclusión, inspeccionar doce Grad-CAM y la secuencia ecológica, y redactar discusión sustentada en cifras. El cierre requiere PDF NeurIPS de 6–10 páginas, al menos cinco referencias, media página de uso responsable, cinco autores reales y revisión editorial. Crear `reports/wildlife/finalized.json` únicamente después de esa revisión. Publicación, entrega por plataforma y presentación siguen pendientes. El enunciado fija la entrega y presentación para el **30 de octubre de 2026**.
