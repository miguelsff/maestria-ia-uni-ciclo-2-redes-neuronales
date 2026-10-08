# Pregunta 3: ejecución v2 y revisión pendiente

**Registro histórico de preparación.** Las revisiones humanas posteriores a este documento se verificaron directamente en el CSV. El flujo actual, con v2 preservada, está en [WILDLIFE_V3_RUN.md](WILDLIFE_V3_RUN.md). El comando actual selecciona v3 por defecto y rechaza modificar la salida v2 congelada con fuentes diferentes.

La preparación v2 incorpora 1.000 fotografías vacías reales al entrenamiento y conserva las particiones reservadas. **El entrenamiento científico todavía no comenzó:** las 90 fotografías de la muestra de etiquetas requieren revisión humana. La galería de oclusión contiene los 17.620 objetos elegibles de trans_test, todos pendientes. Esta guía registra ejecución técnica; no acredita revisión humana ni entrega del examen.

Salida nueva: `E:/Kaggle/exam-output-wildlife-v2`. La salida anterior `E:/Kaggle/exam-output-wildlife` y la [auditoría histórica](audits/revision-2026-10-04/03_wildlife.md) se conservan. Los pasos y las constancias actuales están en [WILDLIFE_STATUS.json](WILDLIFE_STATUS.json) y en `receipts/` de la salida nueva.

## Acción necesaria antes del entrenamiento

Abrir la [galería de etiquetas](http://127.0.0.1:8894/label_review.html). Inspeccionar las 90 fotografías, verificar categoría y caja o ausencia de animales, registrar observación visual y nombre del evaluador. Marcar cada fila como correcta solo después de esa inspección. El botón de exportación descarga `label_review.csv`; guardarlo con ese nombre en `E:/Kaggle/exam-output-wildlife-v2/wildlife/`, reemplazando únicamente el CSV revisado. Las filas que no se modifican conservan su autoría anterior.

Una fila `needs_changes` impide el entrenamiento. Resolver el error en una cohorte nueva si afecta datos o conversión; conservar la evidencia original. No retirar filas ni reutilizar el CSV de otra identidad para superar el control. La muestra v2 tiene cinco fotos por categoría disponible y veinte fondos de entrenamiento, distribuidos entre las ocho cámaras adicionales. La revisión original permanece pendiente y archivada en su salida; no se presenta como aprobación de la muestra nueva.

La [galería de oclusión](http://127.0.0.1:8894/occlusion_review.html) puede revisarse antes de observar errores del modelo. Asignar visibilidad corporal: baja >75 %, media 25–75 %, alta <25 %. Usar `ambiguous` para visibilidad incierta o truncamiento; no convertir tamaño en oclusión. Registrar evidencia y evaluador en cada objeto, exportar `occlusion_review.csv` y guardarlo en la misma carpeta. El análisis v2 exige cobertura completa y al menos 30 objetos por nivel; conserva estado pendiente mientras falte una revisión. Los casos ambiguos se cuentan explícitamente y se ignoran como regiones crowd en la comparación por nivel.

Ambas galerías muestran 24 tarjetas por página, permiten ampliar imágenes y navegar a pendientes. El servidor solo lee archivos y escucha en el puerto local 8894. Se comprobaron el acceso HTTP, enlaces, sintaxis y paginación/exportación en fixtures sintéticos en memoria. La prueba conservó autoría previa y solo asignó el evaluador nuevo a filas modificadas; no alteró ninguna aprobación real. La automatización del navegador integrado no pudo abrir una pestaña durante esta ejecución, por lo que no se declara verificación visual completa de su interfaz.

## Reproducción con un comando

Desde `parcial/`, con Docker Desktop/WSL2, GPU NVIDIA, los originales CCT20 y `.env.wildlife-v2` configurado:

```powershell
.\scripts\run_wildlife.ps1
```

El comando verifica o descarga la cohorte acotada de fondos y los pesos locales, prepara y audita los datos, abre el servidor de revisión, comprueba CUDA/AMP y congela las fuentes experimentales. Se detiene si las etiquetas carecen de revisión humana real. Tras esa revisión, el mismo comando entrena YOLOv8s, evalúa cis/trans, calcula tamaños y oclusión, genera doce CAM, ejecuta el módulo ecológico, exporta activos y ejecuta el notebook. La oclusión pendiente queda indicada en los activos e impide cerrar el informe final.

Para preparar datos y galerías sin iniciar entrenamiento:

```powershell
.\scripts\run_wildlife.ps1 -PrepareOnly
```

Para recuperar una interrupción compatible o regenerar evaluaciones con el modelo ya completo:

```powershell
.\scripts\run_wildlife.ps1 -Resume
```

El protocolo está en `configs/wildlife.toml`: YOLOv8s, 14 categorías, semilla 42, resolución 640, batch 16, AdamW, tasa inicial 0,0003, weight decay 0,0001, hasta 50 épocas y paciencia 10. best.pt se selecciona por mAP50:95 de cis_val depurado. La firma incluye fuentes, configuración, datos, versiones y pesos. Cambiar código experimental requiere otra salida; los archivos editoriales son independientes. Evitar trabajos GPU simultáneos.

`.env.wildlife-v2` es local e ignorado por Git. Para otra reproducción, copiar su ejemplo, elegir una salida y carpeta de fondos nuevas, definir un token privado y usar otro nombre de proyecto Compose si se ejecutan ambas preparaciones en paralelo. La fuente montada bajo `/workspace` y los originales se leen en solo lectura; los enlaces de fotografías residen en volúmenes Linux.

## Cohorte fijada y desviación del plan inicial

La propuesta inicial buscaba negativos independientes en las diez cámaras train de ECCV. Los metadatos completos contienen 2.130 fotos vacías de esas cámaras, pero todas estaban reservadas por imagen y secuencia. Se fijó una ampliación explícita con ocho cámaras del train oficial de la colección completa, fuera de las veinte cámaras ECCV: `106, 23, 26, 57, 64, 70, 94, 96`. Cada cámara aporta 125 secuencias distintas y una fotografía por secuencia. La selección utiliza metadatos y semilla, sin inspeccionar rendimiento del modelo.

Solo se admiten fotografías anotadas exclusivamente como empty. Se excluyen todos los IDs, secuencias y cámaras ECCV originales, incluidos los registros descartados de evaluación. Se verificaron dimensiones, orientación EXIF, decodificación completa, SHA-256 y MD5 del proveedor en las descargas. La entrada adicional incluye metadatos originales, recibos por archivo y `inputs/backgrounds/cohort.json`; la preparación vuelve a reconstruir la selección y comprueba hashes.

Fuentes: [colección oficial LILA](https://lila.science/datasets/caltech-camera-traps), [metadatos completos](https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/labels/caltech_camera_traps.json.zip) y [particiones oficiales de la colección completa](https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/CaltechCameraTrapsSplits_v0.json).

| Partición | Fotografías elegibles | Fondos | Cámaras |
|---|---:|---:|---:|
| train ampliado | 12.399 | 1.000 | 18 |
| cis_val depurado | 2.543 | 995 | 10 |
| cis_test | 13.073 | 1.180 | 10 |
| trans_val | 1.592 | 61 | 1 |
| trans_test | 18.647 | 1.778 | 9 |

Todos los registros originales elegibles y los fotogramas de monitoreo coinciden exactamente con el manifiesto anterior. No hay imágenes ni secuencias compartidas entre particiones. La integridad abarca 58.864 fotografías: 57.864 originales y 1.000 adicionales. Identidad del manifiesto v2: `84c57988de08a0464de385c5fb623ada3741d7adfdb6832f4755305961fdbc39`.

La ampliación aporta exclusivamente negativos en las ocho cámaras adicionales y utiliza fotografías de resolución original, mientras que los positivos ECCV son reducidos. Aunque el detector redimensiona ambos grupos a 640, estas diferencias pueden asociar procedencia con vacío. El experimento debe declararse como entrenamiento ampliado; no equivale a reproducir estrictamente el train ECCV. Una comparación con/sin negativos necesita otra ejecución controlada. La oclusión completa cubre el test elegible definido, no los animales sin caja excluidos del benchmark efectivo.

## Verificaciones y evidencia

Pasaron 32 pruebas enfocadas y los controles de estilo. Cubren conversión, procedencia, independencia de particiones, evaluación COCO, controles de revisión, cobertura completa de oclusión, reportes y bloqueo del compilador sin activos científicos. CUDA, gradientes, NMS y AMP pasaron en RTX 5090. La reanudación sintética recuperó época y 183 estados de optimizador y completó tres épocas; no demuestra equivalencia bit a bit. La corrida pequeña terminó con `scientific_run=false` y sin usar tests; sus cifras no se incorporan al informe científico.

El notebook `E:/Kaggle/exam-output-wildlife-v2/notebooks/03_wildlife.ipynb` muestra evidencia guardada, con entrenamiento desactivado. Sus doce celdas de código se ejecutaron sin errores; conserva mensajes pendientes de resultados científicos. El notebook fuente y la metodología del informe ya describen la ampliación y el protocolo completo de oclusión.

Constancias principales bajo la salida nueva: `receipts/preflight.json`, `receipts/source-freeze.json`, `receipts/doctor_gpu.json`, `receipts/amp.json`, `receipts/focused-tests.json`, `receipts/lint.json`, `receipts/scientific-gate.json`, `receipts/final-verification.json`, `resume-check/passed.json` y `smoke/execution_summary.json`. Se verificó el comando completo en modo `-PrepareOnly`, se congelaron 24 archivos experimentales y se comprobó el bloqueo real antes de cargar el detector. El inventario instalado está en `receipts/environment.txt` y la identidad de imagen Docker en `receipts/runtime_image_id.txt`. El snapshot previo de fuentes está en `baseline-source/`; el manifiesto original preservado está en `baseline-manifest.json`.

## Informe, cierre y entrega

Tras completar entrenamiento y oclusión, volver a ejecutar con `-Resume` para generar los activos actualizados y el notebook. Completar la discusión con cifras reales, revisar doce CAM y verificar la demostración de cámara nueva y su ráfaga íntegra. El preámbulo compartido ya contiene los cinco autores utilizados en las otras preguntas; confirmar la autoría con el grupo antes del cierre editorial.

```powershell
$wildlifeV2Compose = @('--env-file', '.env.wildlife-v2', '-f', 'compose.wildlife.yaml', '-f', 'compose.wildlife.v2.yaml')
docker compose @wildlifeV2Compose run --rm reports --task wildlife --draft
```

Actualmente no hay activos científicos ni PDF v2. El compilador rechaza esa ausencia incluso en modo borrador. Su compatibilidad con el informe final y el diseño del PDF se verificarán cuando existan resultados. El cierre requiere NeurIPS de 6–10 páginas, al menos cinco referencias, discusión crítica y media página de uso responsable; conservar las revisiones reales. Registrar `reports/wildlife/finalized.json` únicamente después de revisión editorial y compilar sin `--draft`.

La publicación del repositorio, entrega por plataforma y presentación permanecen pendientes. No se registró ninguna de esas acciones en esta ejecución. Para volver a consultar la preparación original, utilizar su salida y snapshot; no restaurar archivos sobre esta cohorte ni retirar sus controles de revisión.
