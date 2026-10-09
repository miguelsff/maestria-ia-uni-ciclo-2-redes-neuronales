# Evaluación de la pregunta 2: Sen1Floods11 e inundaciones

**Estado vigente:** el plan técnico de cierre fue ejecutado después de la auditoría. La sección 11 registra autoría completa, PDF revisado visualmente y publicación verificada. Las secciones anteriores conservan el diagnóstico histórico; la aprobación editorial del grupo y la entrega por la plataforma siguen pendientes.

Fecha de revisión: **4 de octubre de 2026, America/Lima**. Alcance: auditoría y planificación de la pregunta 2. Se leyeron las siete páginas del examen y las once del ejemplo. No se entrenaron modelos, no se modificaron datos, código, notebooks ni informes científicos y no se publicó nada. Este documento es la única entrega nueva de esta revisión.

**Veredicto:** la pregunta está implementada y contestada científicamente; las seis corridas están completas y sus 618 predicciones vuelven a pasar una auditoría en CPU. El notebook contiene resultados ejecutados y el borrador NeurIPS tiene nueve páginas y siete referencias. **La entrega al profesor sigue pendiente:** falta incorporar la autoría real al informe, completar la revisión editorial, producir el PDF final y acreditar la publicación de la versión auditada en GitHub. Los cinco integrantes y la afiliación ya fueron proporcionados por el usuario y se registran abajo. Conviene además hacer explícito el comando completo en el README y corregir instrucciones que todavía describen la fase de preparación.

### Datos del grupo proporcionados por el usuario

**Afiliación:** UNI · Unidad de Posgrado — Facultad de Ingeniería Industrial y de Sistemas.

| Orden proporcionado | Integrante |
|---|---|
| 1 | Chilón Cruzado Josué |
| 2 | Concepción Tiza Miguel Ángel |
| 3 | Neyra Dianderas Enzo Israel |
| 4 | Tello Guerra Omar Salvador |
| 5 | Alexander Vladimir Paco Cornejo |

Esta actualización resuelve la falta de información sobre los cinco nombres y la afiliación. Los correos no fueron proporcionados y no se inventan; no son un requisito expreso del examen. Se conserva el orden y la escritura enviados por el usuario. El informe fuente y el PDF siguen pendientes de incorporar estos datos: esta etapa mantiene el alcance de auditoría y solo actualiza este documento.

## 1. Veredictos separados

| Dimensión | Estado | Fundamento y límite |
|---|---|---|
| Implementación y respuesta | **Cumple** | U-Net explícita en PyTorch con VV/VH, seis ablaciones, evaluación ID/Bolivia, dos hipótesis y tres patrones de fallo con ejemplos. La lectura del código verifica la implementación visible; no certifica por sí sola la autoría de su desarrollo. |
| Ejecución científica | **Cumple** | Seis directorios con `scientific_run=true`, historial completo, `complete.json`, checkpoints y evaluación de 88 chips ID + 15 Bolivia por variante. No son salidas de `smoke/`. |
| Validación local | **Cumple** | Recalculados conteos/métricas de los 618 GeoTIFF, selección del checkpoint por mínima pérdida de validación, integridad de datos y normalización manual de entrenamiento. Las 20 pruebas funcionales y CUDA/AMP están documentadas por logs anteriores; no se repitieron esas pruebas en esta revisión. |
| Notebook | **Cumple** | 17 celdas totales, ocho de código con ejecuciones 1–8, sin salidas de error; las copias de proyecto y salida son idénticas. `RUN_TRAINING=False`: demuestra lectura y presentación de resultados persistidos, no entrenamiento dentro del notebook. |
| Informe | **Parcial** | PDF de nueve páginas, contenido requerido, siete referencias y figuras legibles. La página 1 conserva el placeholder de autores; el estado editorial es pendiente y no existe `flood.pdf` final. |
| Entrega al profesor | **Falta** | No hay evidencia de entrega por la plataforma ni de presentación. `parcial/` está sin versionar en este checkout; la publicación de la versión auditada no está acreditada. No se inspeccionó el contenido remoto de GitHub. |

Estados de la matriz: **cumple** = evidencia suficiente dentro del alcance; **parcial** = existe pero requiere cierre; **falta** = ausencia comprobada; **no verificado** = no se comprobó directamente. “Cumple” no implica aprobación del profesor ni revisión humana ficticia.

## 2. Qué exige el profesor

Fuente normativa: [examen-parcial.pdf](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/examen-parcial.pdf), pp. 1 y 7; requisitos específicos en pp. 3–5.

- **Un informe PDF NeurIPS de 6–10 páginas por pregunta**, con motivación, trabajos relacionados y al menos cinco referencias, metodología, experimentos, resultados y discusión crítica de limitaciones.
- Grupo de **cinco estudiantes**. La autoría real debe representar al grupo; el enunciado no exige expresamente correos electrónicos, aunque la plantilla local los solicita.
- Código reproducible entregado por **GitHub**, especificación `requirements.txt` o `environment.yml` y **un único comando de extremo a extremo documentado en README**.
- PDF y enlace GitHub por la plataforma del curso; entrega y presentación en clase el **30 de octubre de 2026**. La p. 7 indica penalización del 10 % por día de retraso.
- No se exige envío al leaderboard de Kaggle. Sentinel-2 es **bono opcional**, p. 5; su ausencia no es un incumplimiento.

[ejemplo.pdf](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/ejemplo.pdf) tiene once páginas y cuatro autores en su portada. Orienta la estructura y el uso de tablas/figuras; **no sustituye las exigencias de 6–10 páginas y cinco integrantes del examen**. Tampoco impone sus pérdidas, umbral, datasets o explicaciones causales al estudio SAR.

## 3. Evidencia exacta y matriz de cumplimiento

Los identificadores siguientes resuelven rutas absolutas; en la matriz se indican además archivos, claves o páginas concretas.

| ID | Ubicación absoluta y contenido |
|---|---|
| E1 | [model.py](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/src/neural_exam/flood/model.py:26): `UNet`, encoder/decoder y saltos explícitos. |
| E2 | [data.py](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/src/neural_exam/flood/data.py:33): `read_pair`, `build_records`, `prepare`, `FloodDataset`; [loss.py](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/src/neural_exam/flood/loss.py:6): máscaras y pérdidas. |
| E3 | [flood.toml](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/configs/flood.toml); [pipeline.py](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/src/neural_exam/flood/pipeline.py:131); [training.py](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/src/neural_exam/common/training.py:86); [metrics.py](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/src/neural_exam/common/metrics.py:53). |
| E4 | [manifest.json](E:/Kaggle/exam-output-flood/flood/manifest.json): `records`, `empty_chips`, `counts_before_empty_exclusion`, `split_sha256`, `mean`, `std`, `identity`; [coverage.csv](E:/Kaggle/exam-output-flood/flood/coverage.csv). |
| E5 | [experiments.csv](E:/Kaggle/exam-output-flood/flood/experiments.csv). Sus seis directorios hermanos son `gold_bce`, `gold_dice`, `gold_combined`, `gold_weak_bce`, `gold_weak_dice`, `gold_weak_combined`; cada uno contiene `execution.json`, `run.json`, `complete.json`, `history.csv`, `best.pt`, `last.pt`, y en `test/` y `bolivia/`, `metrics.json`, `per_chip.csv` y mapas `*_probability.tif`. |
| E6 | [flood-results.json](E:/Kaggle/exam-output-flood/audit/flood-results.json), [delivery-verification.json](E:/Kaggle/exam-output-flood/audit/delivery-verification.json), [data-integrity.json](E:/Kaggle/exam-output-flood/audit/data-integrity.json), [final-source-sha256.json](E:/Kaggle/exam-output-flood/audit/final-source-sha256.json). |
| E7 | [02_flood.ipynb](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/notebooks/02_flood.ipynb), idéntico a [copia de salida](E:/Kaggle/exam-output-flood/notebooks/02_flood.ipynb). |
| E8 | [flood_draft.pdf](E:/Kaggle/exam-output-flood/reports/flood_draft.pdf); [main.tex](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/reports/flood/main.tex), [discussion.tex](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/reports/flood/discussion.tex), [flood_references.bib](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/reports/flood/flood_references.bib), [preamble.tex](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/reports/preamble.tex). |
| E9 | [qualitative_cases.csv](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/reports/flood/qualitative_cases.csv), [qualitative-statistics.json](E:/Kaggle/exam-output-flood/audit/qualitative-statistics.json), [event_metadata.geojson](E:/Kaggle/exam-output-flood/audit/event_metadata.geojson), [remote-weak-inventory.json](E:/Kaggle/exam-output-flood/audit/remote-weak-inventory.json). |
| E10 | [FLOOD_RUN.md](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/FLOOD_RUN.md:24), [README.md del parcial](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/README.md), [run_flood.ps1](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/scripts/run_flood.ps1), [compose.flood.yaml](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/compose.flood.yaml), [requirements.txt](D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/requirements.txt). |

| Requisito y página del examen | Evidencia exacta | Estado | Pendiente |
|---|---|---|---|
| U-Net propia PyTorch con VV/VH, p. 4, punto 1 | E1 `UNet`; E4 `band_order=[VV,VH]`; E8, pp. 2–3 | Cumple | Completar atribución real del trabajo del grupo. |
| Manual frente a manual+débil y segundo factor, pp. 4–5, punto 2 | E3 `supervision` × `losses`; E5 seis corridas; E8 tablas 2–6 | Cumple | Mantener explícito que también cambia el número de actualizaciones. |
| Particiones oficiales; Bolivia fuera de entrenamiento, pp. 4–5 | E2 `build_records`; E4 `records` y hashes CSV; E5 `training_chips=251/4401`, `validation_chips=86` | Cumple | Ningún entrenamiento adicional necesario. |
| No usar los 446 manuales como entrenamiento incluyendo test/Bolivia, pp. 4–5 | E4 original 252/89/90/15; efectivo 251/86/88/15; E3 `training_records=records['train']` | Cumple | Explicar que 446 es el total manual, no el tamaño de train. |
| IoU promedio por chip y omisión/comisión, p. 4 | E5 `mean_chip_iou`, `reference_mean_chip_iou`, `pooled_pixels`; E8 pp. 3–4, tablas 2–5 | Cumple | Conservar visibles las dos convenciones y sus denominadores. |
| IoU, F1, precisión y recall agua ID, p. 5, punto 3 | E5 `test/metrics.json`, `test/per_chip.csv`; E8 tabla 3 | Cumple | No intercambiar IoU por chip con IoU global. |
| Matriz de confusión ID comentada, p. 5, punto 3 | E8 p. 4, comentario; p. 6, figura 3; E5 `gold_bce/test/metrics.json` | Cumple | Conservar denominadores: FP/no agua no equivale a comisión. |
| Evaluación solo manual Bolivia y brecha, p. 5, punto 4 | E4 quince registros Bolivia; E5 `bolivia/`; E8 tabla 2 | Cumple | Mantener el signo ID−Bolivia; no convertir una mejora externa en caída. |
| Dos hipótesis sobre generalización, p. 5, punto 4 | E8 p. 8, H1 apariencia/prevalencia y H2 adquisición/sesgo débil; E9 metadatos | Cumple | Hipótesis pendientes de contraste, no hechos causales probados. |
| Tres tipos de error SAR con ejemplos visuales, p. 5, punto 5 | E8 pp. 6–8, figuras 4–6; E9 tres casos y estadísticas | Cumple | Revisión editorial del grupo; no afirmar sombra, doble rebote o cobertura confirmados. |
| Sentinel-2, p. 5, punto 6, bono | E8 p. 2 declara exclusión | Cumple respecto al alcance obligatorio | No se reclama el bono. |
| Motivación, relacionados, metodología, experimentos, resultados, discusión crítica, p. 1 | E8 secciones 1–8 | Cumple | Revisión editorial final. |
| Mínimo cinco referencias, p. 1 | E8 p. 9 enumera [1]–[7]; siete entradas en bibliografía propia | Cumple | Verificar que sigan siendo siete después de edición. |
| PDF NeurIPS, 6–10 páginas por pregunta, pp. 1 y 7 | E8 nueve páginas; `preamble.tex` usa `neurips_2026` en modo preprint | Cumple en borrador | Volver a comprobar extensión y visualización del PDF final. |
| Cinco integrantes, pp. 1 y 7 | Cinco nombres y afiliación proporcionados por el usuario, registrados al inicio; E8 p. 1 todavía dice “completar nombres y correos” | Parcial | Incorporar los nombres y la afiliación al informe. Correos únicamente si se proporcionan y se conservan en la plantilla. |
| Entorno especificado, p. 7 | E10 versiones fijadas y Dockerfile; E5 entorno ejecutado | Cumple localmente | Publicar archivos y comprobar disponibilidad desde un checkout limpio. |
| Comando único completo en README, p. 7 | E10 guía vinculada contiene `scripts/run_flood.ps1`; README literal ofrece el runner genérico y otros pasos | Parcial | Escribir directamente el comando flood, prerrequisitos y salida en README; aclarar que produce borrador. |
| Código reproducible GitHub, pp. 1 y 7 | `git status --short`: `M README.md`, `?? parcial/`; origin configurado en GitHub | No verificado en remoto | Publicar la versión cerrada y verificar enlace/contenido accesible. La configuración de origin no demuestra publicación. |
| PDF y enlace por plataforma; presentación 30/10/2026, p. 7 | No existe PDF final ni comprobante de plataforma/presentación | Falta | Entrega y presentación reales. |

## 4. Comprobaciones nuevas y constancias anteriores

### Comprobado directamente en esta revisión

1. **Integridad de datos:** 9.216 archivos, **7.709.124.682 bytes**, sin fallos de tamaño/MD5 frente al inventario guardado en `E:/Kaggle/Sen1Floods11/download_manifest.json`. Se cotejaron además los SHA-256 de **9.180 imágenes/máscaras usadas** contra E4. La huella del inventario coincide con E6. Esto verifica el inventario publicado previamente guardado, no una nueva consulta en vivo al bucket.
2. **Particiones:** hashes actuales de los cuatro CSV coinciden; los diez cruces entre train/val/test/Bolivia/weak tienen cero IDs compartidos. Train, val, test y weak contienen cero IDs Bolivia; Bolivia contiene quince. Los 224 débiles Bolivia se excluyeron ya del inventario descargado; por eso `manifest.excluded.bolivia=0` se refiere al inventario local y no niega esa exclusión anterior.
3. **Normalización:** recalculada en CPU con los 251 manuales train y 57.017.597 píxeles válidos: media `[-10.455064161886204, -17.325308892562283]`, desviación `[4.06613688848822, 4.791758381768977]`. Coincide con E4; no depende de val/test/Bolivia ni de débiles.
4. **Resultados científicos:** reejecutado el auditor existente con CPU, montajes de solo lectura y la escritura de constancias sustituida por impresión en memoria. Pasaron los **618 GeoTIFF**: CRS/transformación, forma, `float32`, nodata −1, probabilidades válidas, conteos por chip, métricas globales y dos promedios de IoU. Verificó que `best.pt` corresponde a la mínima `val_loss`, su pérdida coincide y los pesos son finitos. No se realizó inferencia ni entrenamiento nuevo.
5. **Casos cualitativos:** recalculadas medianas VV/VH por TP/TN/FP/FN y concentración de errores en borde de los tres chips. Coinciden con E9. Los paneles muestran los tres patrones descritos; no permiten identificar causas físicas particulares.
6. **Entregables:** PDF, ambas copias del notebook, constancia de resultados, seis `best.pt` y nueve PNG de páginas coinciden con las huellas anteriores. El PDF tiene nueve páginas; se revisaron las imágenes existentes y se renderizó directamente la página 8 en memoria para despejar una aparente anomalía de la vista previa: los títulos de Bolivia están completos. No se guardaron imágenes nuevas. No se detectó un defecto visual que impida la lectura del borrador.
7. **Fuente:** **39 de 42 archivos** coinciden con `final-source-sha256.json`. Cambiaron `README.md`, `parcial/README.md` y `parcial/PREPARATION_STATUS.md`. No cambió el código de entrenamiento, la configuración flood ni sus fuentes de informe recogidas por esa constancia. El snapshot completo de fuente requiere actualizarse al cerrar la documentación.
8. **Comando:** el parser de PowerShell no reportó errores y Compose acepta la configuración. Se inspeccionaron sus etapas y los logs persistidos. No se repitió el flujo completo desde cero ni se acreditó reproducibilidad desde GitHub.

### Evidencia anterior inspeccionada, sin repetir su ejecución

`E:/Kaggle/exam-output-flood/audit/tests.log` termina en **20 passed, 34 warnings in 3.75s**. `doctor.log`, `smoke.log`, `training-attempt-1.log`, `notebook.log`, `report-build.log`, `final-lint.log` y `final-format.log` son constancias anteriores del entorno, controles y generación de entregables. La equivalencia caché/originales consta en `audit/cache-equivalence.json`; no se volvió a copiar ni auditar el volumen Linux. La selección previa del baseline `gold_bce` se declara en el protocolo y es fija en el código; esta revisión no reconstruye retrospectivamente la fecha de esa decisión.

No existe `D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/reports/flood/finalized.json`; tampoco `E:/Kaggle/exam-output-flood/reports/flood.pdf`. El campo `qualitative_examples_reviewed=3` de los activos registra revisión técnica, **no certifica aprobación humana del grupo**.

### Huellas relevantes que siguen coincidiendo

| Archivo | SHA-256 |
|---|---|
| `E:/Kaggle/exam-output-flood/reports/flood_draft.pdf` | `00c86d3b767cb5ed35f03325cf92d9e906cea3b59e637ab77f044eb2825eeeb0` |
| Ambos `02_flood.ipynb` de E7 | `af449a14783ca7a546565f7abf00f007ad42a125dd7d6e5bd9c04cf22ba0a5b0` |
| `E:/Kaggle/exam-output-flood/audit/flood-results.json` | `9c942ed9637d6cce98991b97577ac1e8af7f3613f476ee31dfc7efd12c60b8df` |
| `E:/Kaggle/Sen1Floods11/download_manifest.json` | `059f22c2de000885da9c1f4550a2b888d8fd60f4cc1dea4a4a72a225f4dae32e` |

Las seis huellas de checkpoints y las nueve de páginas están íntegramente en E6. Las huellas actuales de los documentos que cambiaron son `997c5d112ba5a4c5687b66ff87ea987433150bf5e84205348b7f8aedc07fa2d9` (PREPARATION_STATUS), `af76b2945d5838bfdb0527d176322803da37a0df567ca7022d0e74d426e761da` (README del parcial) y `8ccedb90d92c666e5031a848de274c1fd86adb7077275f0a2b9298a26c1c68a6` (README raíz), al momento del cotejo.

## 5. Qué resultados están sostenidos por experimentos completos

| Experimento | Épocas ejecutadas | Mejor época validación | Pasos/época | IoU chip ID | IoU chip Bolivia | ID−Bolivia |
|---|---:|---:|---:|---:|---:|---:|
| gold_bce | 50 | 47 | 63 | 0,3939 | 0,4555 | −0,0616 |
| gold_dice | 41 | 31 | 63 | 0,4100 | 0,4463 | −0,0364 |
| gold_combined | 50 | 49 | 63 | 0,4068 | 0,4057 | +0,0010 |
| gold_weak_bce | 38 | 28 | 1.101 | 0,3251 | 0,2690 | +0,0560 |
| gold_weak_dice | 11 | 1 | 1.101 | 0,3487 | 0,4790 | −0,1302 |
| gold_weak_combined | 22 | 12 | 1.101 | 0,3593 | 0,4511 | −0,0918 |

Llegar a once épocas con el mejor checkpoint en la primera no significa una prueba corta: `gold_weak_dice` completó la parada con paciencia diez, sobre 4.401 chips por época. Los máximos de 50 épocas y las paradas tempranas están respaldados por `history.csv`, `complete.json` y checkpoints.

Para el baseline `gold_bce`, ID tiene TP=1.884.956, FP=322.880, FN=630.052 y TN=17.606.071. F1=0,7982, precisión=0,8538, recall=0,7495, omisión=0,2505 y comisión=0,1462. IoU **global**=0,6642; IoU **promedio por chip**=0,3939. La matriz comentada de la p. 4 y figura de la p. 6 muestran ambos errores y el predominio de no agua.

La IoU principal excluye chips con unión vacía; el número excluido ID varía por predicción: **2/0/0/4/1/2** en el orden de la tabla. La columna de referencia usa epsilon `1e-7` y asigna uno a unión vacía; incluye los 88 chips válidos. En Bolivia las seis variantes tienen quince IoU definidas. La principal, por tanto, cambia su conjunto efectivo de chips entre modelos; la tabla de referencia permite una comparación sobre el mismo conjunto. El informe no debe ocultar esa diferencia.

Datos efectivos: 251 train, 86 val, 88 test, 15 Bolivia y 4.150 weak. Los 16 chips sin píxeles válidos son uno train, tres val, dos test y diez weak. La exclusión de `Ghana_83483` y `Paraguay_34417` explica que test oficial de 90 termine en 88 evaluables. No se inventa una puntuación para chips totalmente nodata.

El inventario remoto **guardado el 3/10/2026** tiene 4.384 débiles S1, frente a 4.385 en el enunciado: 224 Bolivia y 4.160 admisibles locales, diez inválidos. `weak_max=4385` es un techo; no rellena el faltante ni convierte todo el conjunto manual en entrenamiento.

## 6. Decisiones científicas, controles y límites

**Exigencias del examen:** dos factores de ablación, entradas VV/VH, test manual ID, Bolivia excluida, métricas, dos hipótesis y tres ejemplos de error. No exige una semilla específica, un valor mínimo de IoU ni demostrar causalmente las hipótesis.

**Decisiones del equipo:** U-Net con base 32 y GroupNorm; clipping −50 a 10 dB; normalización solo manual train; peso débil 0,3 por chip; BCE/Dice/suma; AdamW; semilla 42; batch cuatro; 50 épocas máximas; paciencia diez; umbral fijo 0,5; checkpoint mínimo de pérdida de validación y baseline cualitativo fijo. Son decisiones documentadas, no mandatos del profesor.

**Controles añadidos por la implementación:** hashes y firmas de configuración/datos, rechazo de reanudaciones incompatibles, caché verificada, límites de recursos, logs, auditor de predicciones y archivo `finalized.json` antes de compilar el PDF final. Este último es un control editorial del proyecto, no una constancia exigida expresamente por el examen. `--draft` evita ese control y no certifica entrega final.

Límites que deben mantenerse en el cierre:

- Una sola semilla y 15 chips de un evento externo no permiten afirmar significancia o generalización a países en general. CSV oficiales no garantizan independencia espacial o por evento en ID.
- Manual y mixto usan **63 y 1.101 actualizaciones por época**; cambian también épocas ejecutadas y el peso nominal de las fuentes. El efecto de débiles está confundido con presupuesto de entrenamiento. Igualar pasos y repetir semillas fortalecería la inferencia, a mayor costo computacional; no es un requisito pendiente para cumplir esta entrega descriptiva.
- BCE/Dice/combinada tienen escalas de validación distintas. Cada corrida se selecciona dentro de su propia pérdida; sus mínimos no son una clasificación común de calidad.
- Los casos son intencionales y no estiman frecuencia poblacional de fallos. Se observa ambigüedad de baja retrodispersión (`Pakistan_70625`), omisión de agua con señal más alta (`Pakistan_849790`) y fragmentación/bordes (`Bolivia_379434`). El último concentra 9.844/12.116 errores, 81,25 %, en la banda morfológica definida. La correspondencia geográfica verificada no demuestra ausencia de todo error de anotación.
- H1 plantea cambio de apariencia/prevalencia; H2 plantea adquisición y sesgo débil. Los metadatos Bolivia respaldan órbita descendente relativa 156 y fecha 15/2/2018, no ángulo de incidencia por chip. Sombra, doble rebote, suelo húmedo o vegetación inundada siguen siendo explicaciones posibles; VV/VH no las identifican por sí solos.
- La clase objetivo mide agua, sin separar agua permanente de inundación nueva. No se midieron profundidad, daño ni capacidad operativa de emergencia.

## 7. Comandos concretos de comprobación

Todos los comandos siguientes son de inspección. El auditor habitual escribe una constancia: el envoltorio mostrado **suprime esa escritura** y usa salidas/datos de solo lectura, sin GPU. Imprime el resultado nuevo; no sustituye los recibos anteriores.

```powershell
Set-Location 'D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales'
git status --short
git remote -v
Get-FileHash -Algorithm SHA256 -LiteralPath 'E:/Kaggle/exam-output-flood/reports/flood_draft.pdf'
Get-FileHash -Algorithm SHA256 -LiteralPath 'parcial/notebooks/02_flood.ipynb'
Get-FileHash -Algorithm SHA256 -LiteralPath 'E:/Kaggle/exam-output-flood/notebooks/02_flood.ipynb'
docker compose -f parcial/compose.flood.yaml config --quiet
```

```powershell
@'
import runpy, sys
import neural_exam.common.io as io
io.write_json = lambda path, value: print('READ_ONLY_RECEIPT', value)
sys.argv = ['scripts/audit_flood_results.py', '--config', 'configs/flood.toml']
runpy.run_path('scripts/audit_flood_results.py', run_name='__main__')
'@ | docker run --rm -i --read-only --network none --cpus 2 --memory 2g --tmpfs /tmp `
  -e PYTHONDONTWRITEBYTECODE=1 -e MPLCONFIGDIR=/tmp/matplotlib -e OMP_NUM_THREADS=1 `
  -e SEN1_DATA_DIR=/datasets/Sen1Floods11 -e OUTPUT_DIR=/outputs `
  --mount 'type=bind,source=D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial,target=/workspace,readonly' `
  --mount 'type=bind,source=E:/Kaggle/Sen1Floods11,target=/datasets/Sen1Floods11,readonly' `
  --mount 'type=bind,source=E:/Kaggle/exam-output-flood,target=/outputs,readonly' `
  --entrypoint python neural-exam-flood:0.1 -
```

Criterio: salida `status=passed`, `complete_matrix=True`, `geotiffs_verified=618`, doce evaluaciones y salida del proceso cero. La imagen local usada tiene ID `sha256:d72c270987d25e96f58f1913c746e8618fcad885f429ec3d6f09a0a4ac327046`.

Comprobación corta de hashes de fuente, notebook y extensión:

```powershell
@'
import json, hashlib
from pathlib import Path
from pypdf import PdfReader
root = Path.cwd()
out = Path('E:/Kaggle/exam-output-flood')
sha = lambda p: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
receipt = json.loads((out/'audit/final-source-sha256.json').read_text())
print('SOURCE_MISMATCHES', [x['path'] for x in receipt['files']
      if not (root/x['path']).is_file() or sha(root/x['path']) != x['sha256']])
nb = json.loads((root/'parcial/notebooks/02_flood.ipynb').read_text(encoding='utf-8'))
code = [c for c in nb['cells'] if c['cell_type']=='code']
print('CODE_CELLS', len(code), 'COUNTS', [c['execution_count'] for c in code],
      'ERRORS', sum(o['output_type']=='error' for c in code for o in c['outputs']))
print('COPIES_MATCH', sha(root/'parcial/notebooks/02_flood.ipynb') == sha(out/'notebooks/02_flood.ipynb'))
print('PDF_PAGES', len(PdfReader(out/'reports/flood_draft.pdf').pages))
'@ | & 'C:/Users/USER/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -
```

El comando científico completo documentado en la guía es `D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/scripts/run_flood.ps1`. Se ejecuta desde `parcial/` como `./scripts/run_flood.ps1`, con Docker Desktop, WSL2, NVIDIA y datos completos. Construye, verifica caché, comprueba entorno/pruebas, realiza smoke y seis corridas, audita, genera activos/notebook y compila **borrador**. No lo ejecutar ahora sobre la salida auditada. `-Resume` reutiliza checkpoints completos pero vuelve a evaluar y escribir activos; no es una comprobación de solo lectura. Para una reproducción completa futura usar una **salida nueva** mediante `FLOOD_OUTPUT_DIR`.

## 8. Plan propio de cierre, sin repetir experimentos completos

| Prioridad | Paso y dependencias | Criterio de cierre y verificación |
|---|---|---|
| P0 | Incorporar en `parcial/reports/preamble.tex` los cinco integrantes y la afiliación registrados al inicio. La información necesaria sobre nombres y afiliación ya está disponible. | Cinco nombres con la escritura y el orden proporcionados; afiliación indicada; ningún placeholder. Correos solo si se proporcionan y se conservan en plantilla. No inventar responsabilidades. |
| P0 | Revisión editorial del grupo sobre el borrador E8, cifras y limitaciones. Depende de autoría y lectura del estudio. | Grupo confirma contenido, tres patrones, dos hipótesis, denominadores y comparación con pasos distintos. Registrar `finalized.json` solo después de revisión real. No se necesita una anotación SAR causal nueva para cerrar el requisito. |
| P1 | Ajustar README del parcial y resumen raíz para describir el estado actual. Depende de acordar la entrega. | Comando literal `./scripts/run_flood.ps1`, datos/prerrequisitos, variables de rutas, salida nueva y alcance del borrador. Corregir frase “originales ... sin ejecutar” que contradice el notebook actual y actualizar referencias a etapas futuras. |
| P1 | Compilar el PDF final desde activos existentes, sin lanzar el runner científico. Depende de P0 y `finalized.json` real. | Desde `parcial/`: `docker compose -f compose.flood.yaml run --rm reports --task flood`. Debe producir `E:/Kaggle/exam-output-flood/reports/flood.pdf`, 6–10 páginas, al menos cinco referencias, autoría completa y lectura visual de todas las páginas. |
| P1 | Actualizar la evidencia de cierre y conservar el borrador y constancias históricas. Depende de ediciones/compilación. | Hashes nuevos del PDF final y fuente/documentación; diferencias explicadas. Notebook y métricas siguen rastreables a E5. No reemplazar una constancia histórica como si hubiera cubierto los cambios posteriores. |
| P1 | Preparar/publicar código y material reproducible en GitHub, en una etapa expresamente autorizada para ello. Depende de documentación cerrada. | Checkout limpio contiene paquete, configuración, scripts, pruebas, notebook e informes fuente; enlace accesible y versión identificable. Los tres documentos cambiados quedan incluidos. No tomar `origin` como evidencia de contenido remoto. |
| P2 | Verificar desde checkout publicado el entorno y el comando. Depende de publicación. | Como mínimo resolver Compose, importaciones, configuración y acceso a datos/activos. Si se exige reproducción científica independiente, usar `FLOOD_OUTPUT_DIR` nuevo y ejecutar el comando completo una vez; no es necesario repetirlo en esta auditoría. Registrar límites de hardware/versiones. |
| P0 para la fecha de entrega | Subir PDF final y enlace GitHub a la plataforma, preparar exposición y presentar el 30/10/2026. Depende de PDF/publicación. | Comprobante real de entrega, enlace verificado y exposición del grupo. Ningún comprobante o revisión se fabrica. |

Un estudio adicional con múltiples semillas y pasos igualados sería una mejora científica posterior. El costo es otro entrenamiento; el beneficio es separar mejor el efecto de supervisión del presupuesto. No se convierte esa extensión en obligación del profesor ni en pendiente de la entrega actual.

## 9. Archivos y material exactos de entrega

**Por la plataforma del curso, para esta pregunta:**

1. PDF final previsto: `E:/Kaggle/exam-output-flood/reports/flood.pdf` (**todavía no existe**). El disponible `E:/Kaggle/exam-output-flood/reports/flood_draft.pdf` es material de revisión, no debe presentarse como final mientras conserve el placeholder.
2. Enlace del repositorio configurado: `https://github.com/miguelsff/maestria-ia-uni-ciclo-2-redes-neuronales`, una vez publicada y verificada la versión auditada. No es un archivo adicional ni acredita por sí solo publicación.

El profesor no pide este Markdown, los GeoTIFF, checkpoints ni un ZIP de datos como cargas independientes en la plataforma. El notebook apoya la reproducibilidad; no reemplaza el PDF.

**Conjunto local que debe estar disponible en el repositorio reproducible** (base absoluta `D:/trabajos-miguel/MAESTRIA-UNI-IA/maestria-ia-uni-ciclo-2-redes-neuronales/`; las rutas siguientes se resuelven bajo esa base):

| Grupo | Archivos concretos |
|---|---|
| Entrada y entorno | `README.md`, `.gitignore`, `parcial/README.md`, `parcial/FLOOD_RUN.md`, `parcial/FLOOD_STATUS.json`, `parcial/Dockerfile`, `parcial/.dockerignore`, `parcial/.env.example`, `parcial/pyproject.toml`, `parcial/requirements.txt`, `parcial/compose.yaml`, `parcial/compose.flood.yaml`, `parcial/configs/default.toml`, `parcial/configs/flood.toml`. |
| Scripts usados por el flujo flood | `parcial/scripts/run_flood.ps1`, `parcial/scripts/cache_flood_data.py`, `parcial/scripts/audit_flood.py`, `parcial/scripts/audit_flood_results.py`, `parcial/scripts/compile_reports.py`. |
| Paquete compartido de entrada | Bajo `parcial/src/neural_exam/`: `__init__.py`, `__main__.py`, `cli.py`, `config.py`, `reporting.py`. |
| Código común | Bajo `parcial/src/neural_exam/common/`: `__init__.py`, `cam.py`, `download.py`, `figures.py`, `io.py`, `locking.py`, `metrics.py`, `training.py`. |
| Código flood | Bajo `parcial/src/neural_exam/flood/`: `__init__.py`, `data.py`, `loss.py`, `model.py`, `pipeline.py`, `qualitative.py`, `reporting.py`. |
| Otros módulos del paquete compartido existente | Bajo `parcial/src/neural_exam/mri/`: `__init__.py`, `data.py`, `labels.py`, `loss.py`, `model.py`, `pipeline.py`; bajo `parcial/src/neural_exam/wildlife/`: `__init__.py`, `data.py`, `figures.py`, `metrics.py`, `monitoring.py`, `pipeline.py`, `review.py`, `runtime.py`. Conservarlos para no recortar dependencias del paquete, sin modificar su contenido en este cierre. |
| Pruebas invocadas por el comando | `parcial/tests/test_flood.py`, `parcial/tests/test_metrics.py`, `parcial/tests/test_utilities.py`; mantener también sus dependencias del paquete compartido. |
| Notebook e informe fuente | `parcial/notebooks/02_flood.ipynb`, `parcial/reports/preamble.tex`, `parcial/reports/references.bib`; bajo `parcial/reports/flood/`: `main.tex`, `discussion.tex`, `flood_references.bib`, `qualitative_cases.csv` y, después de la revisión real, `finalized.json`. |

Para que las cifras puedan revisarse sin acceso al disco E: del equipo, conviene acompañar la publicación con un archivo de resultados o una ubicación accesible y documentada. Es una mejora de trazabilidad, no un requisito de subir todos los datos a Git. Su conjunto concreto es:

- Bajo `E:/Kaggle/exam-output-flood/flood/`: `experiments.csv`, `manifest.json`, `coverage.csv`; para **cada una de las seis variantes de la tabla de la sección 5**, `run.json`, `execution.json`, `complete.json`, `history.csv`, `test/metrics.json`, `test/per_chip.csv`, `bolivia/metrics.json`, `bolivia/per_chip.csv`.
- Bajo `E:/Kaggle/exam-output-flood/audit/`: `flood-results.json`, `data-integrity.json`, `delivery-verification.json`, `final-source-sha256.json`, `qualitative-statistics.json`, `event_metadata.geojson`, `event_metadata_source.json`, `remote-weak-inventory.json`, `tests.log`, `doctor.log`, `training-attempt-1.log`, `notebook.log`, `report-build.log`, `pip-freeze.txt`.
- Bajo `E:/Kaggle/exam-output-flood/reports/flood/`: `abstract_results.tex`, `conclusions.tex`, `figures.tex`, `inventory.tex`, `qualitative_figures.tex`, `resources.tex`, `results_analysis.tex`, `tables.tex`, `status.json`, `iou_conventions.csv`, `metrics_all.csv`, `resources.csv`; en `figures/`, **`flood_0.png` a `flood_5.png`**. Son los activos concretos del informe actual; no se necesitan los `figure_*.png` heredados para reconstruir este borrador.

Los seis `best.pt`/`last.pt` y 618 GeoTIFF permanecen disponibles en E5 como evidencia pesada. Pueden conservarse o compartirse mediante un archivo externo con hashes, si se desea revisión sin reentrenar; no hace falta introducirlos en Git. Los datos originales quedan en `E:/Kaggle/Sen1Floods11/`; debe documentarse su obtención y estructura. No basta remitir a una ruta E: privada como si fuera accesible para el profesor.

## 10. Tareas humanas realmente necesarias y cierre

El usuario ya proporcionó los cinco nombres y la afiliación del grupo; no es necesario volver a solicitarlos. El grupo debe leer y aprobar el contenido que firma y realizar la presentación. La entrega por la plataforma necesita una acción real y su comprobante. Incorporar los nombres al informe, compilar, ordenar documentación, actualizar hashes o publicar archivos son operaciones técnicas posteriores, no revisiones humanas que deban inventarse ni razones para pedir nuevos experimentos.

**Cierre de esta auditoría:** implementación, ejecución, validación local y notebook cumplen; informe final y entrega siguen abiertos. Los cinco nombres y la afiliación están disponibles. La prioridad es incorporarlos al informe, completar la revisión editorial, aclarar el comando del README, compilar el PDF final y acreditar GitHub/plataforma. No hay un entrenamiento científico obligatorio pendiente para la pregunta 2.

## 11. Ejecución autorizada del plan: 4 de octubre de 2026

El usuario pidió ejecutar todo el plan. Se incorporaron los cinco nombres y la afiliación en `parcial/reports/preamble.tex`; se aclararon el comando completo, las entradas, el inventario fijado necesario para la caché y el uso de una salida independiente. Se preparó `parcial/PRESENTACION_FLOOD.md` como guion de ocho minutos, con asignaciones propuestas para el grupo.

El nuevo [PDF de revisión](E:/Kaggle/exam-delivery-flood/review-2026-10-04/reports/flood_draft.pdf) contiene nueve páginas, siete referencias y los cinco integrantes. Se compilaron los activos existentes en una salida editorial separada y se inspeccionaron visualmente todas las páginas. No hay placeholders, referencias indefinidas, caracteres faltantes ni cajas desbordadas. SHA-256: `7f97c93e3f248ceb12fc700ed42332894e09b83e517afed248a208fbf8fb7e38`.

La [rama publicada](https://github.com/miguelsff/maestria-ia-uni-ciclo-2-redes-neuronales/tree/codex/flood-delivery) contiene código, configuración, dependencias, scripts, pruebas, notebook ejecutado, fuentes del informe, PDF y evidencia ligera. Commit final verificado: `3343729cf1de412938bf60f17e13667d217d491e`. Se recuperó desde GitHub en una carpeta nueva: los 168 archivos del manifiesto final coinciden byte a byte, el checkout está limpio, Compose resuelve y el control de entorno/importaciones pasa en CPU con acceso de solo lectura a Sen1Floods11. Los datos, checkpoints y GeoTIFF pesados permanecen externos.

La comprobación funcional del paquete preparado pasó 17 pruebas en CPU; tres parametrizaciones de AMP se omitieron porque requieren CUDA. Esto se distingue de las 20 pruebas históricas con GPU. No se repitió entrenamiento completo ni inferencia científica. Los hashes del PDF y notebook originales, la constancia `flood-results.json` y el inventario de descarga original siguen iguales; la verificación editorial anterior también preserva los hashes de los seis checkpoints seleccionados.

Constancias nuevas, separadas de los recibos históricos, bajo `E:/Kaggle/exam-delivery-flood/review-2026-10-04/audit/`: `review-verification.json`, `checkout-tests.log`, `checkout-doctor.log`, `published-doctor.log`, `publication.json`, `publication-head-verification.json` y `report-build.log`. Las fuentes previas a la edición se conservan en `audit/sources-before/`. El paquete publicado conserva su propio `parcial/delivery/flood/source-sha256.json`.

**Pendientes reales:** la aprobación editorial del grupo, la compilación final posterior a esa aprobación, el envío del PDF y enlace por la plataforma con comprobante real y la presentación del 30 de octubre. El control existente de `parcial/scripts/compile_reports.py` exige `reviewed=true` en `parcial/reports/flood/finalized.json`; el archivo conserva `false` porque la autorización para ejecutar operaciones técnicas no certifica que el grupo haya leído y aprobado el informe. El material concreto ya está listo para esa aprobación, sin un entrenamiento obligatorio pendiente.
