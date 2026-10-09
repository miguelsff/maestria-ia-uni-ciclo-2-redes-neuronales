# Etapa A: integración científica de fauna

La oclusión aceptada quedó integrada en el informe principal de diez páginas y en `03_wildlife.ipynb`, ejecutado con 16 celdas y cero errores. El anexo independiente tiene dos páginas y añade tablas por especie. La siguiente etapa es B2: Documentos debe crear los ODT editables y exportar sus PDF finales; los PDF de esta etapa son provisionales de LaTeX.

## Cifras y comparabilidad

| Nivel | Objetos | Clases con soporte | mAP50 | mAP50:95 | IC95 mAP50:95 |
|---|---:|---:|---:|---:|---|
| Baja | 340 | 13 | 0.1582 | 0.0875 | [0.0809, 0.1217] |
| Media | 69 | 10 | 0.0582 | 0.0271 | [0.0184, 0.0523] |
| Alta | 48 | 11 | 0.0563 | 0.0244 | [0.0160, 0.0567] |

Se conservan 125 ambiguos, 18 excluidos sin nivel y 398 notas ausentes. El caso 121 es medium según Miguel, con nota vacía. Los intervalos de ambas métricas tienen 200 réplicas definidas por nivel, semilla 42 y 7.438 secuencias. La firma aceptada es `6b2983c33abccbc18e293cc3a10ca30a710669c2ba8055a76bb462b3348a13a9`.

La AP no ponderada conserva 18.647 fotogramas, incluidos 1.778 vacíos y sus posibles falsos positivos. Cajas no revisadas, excluidas, ambiguas o de otro nivel se tratan como crowd. No se evalúan solo las fotos revisadas. Por el cambio de positivos evaluables y la composición de clases, no se compara directamente esa AP con la global V6 ni se interpreta la diferencia entre niveles como efecto causal. Las proporciones de los aceptados son descriptivas, sin reutilizar la garantía o intervalos poblacionales de los 600 originales.

## Seis entregables conservados

| Entregable original | Evidencia en el informe y notebook | Alcance |
|---|---|---|
| 1. YOLO ajustado y decisiones | Secciones 2–3; argumentos reales, 39 épocas, época 29, negativos y aumentos | Detector existente Ultralytics ajustado por decisión registrada; preparación, evaluación, Grad-CAM y monitoreo propios. La interpretación académica del requisito genérico de componentes propios no se atribuye al docente. |
| 2. AP por especie y desbalance | Tablas 2–4; soporte por clase y entrenamiento | Se conservan clases sin referencia como ND y limitaciones de fox/badger. |
| 3. Cámaras conocidas/nuevas e interpretabilidad | Tablas 2, 5–6; figuras 2–3 | V6: 0.4358 frente a 0.2625; trece clases comunes, doce mapas y tres planos. No prueba causal de fondo. |
| 4. Tamaño y oclusión | Tablas 7–9; anexo y notebook | Tamaños V6 intactos; oclusión limitada a 582 casos aceptados, no revisión exhaustiva de 17.620. |
| 5. Monitoreo de cámara nueva | Sección 6, tablas 10–12 | Cámara 7: 1.853 fotos, 1.420 con detecciones, 211 fechas y ráfaga de tres cuadros. No son individuos. |
| 6. Uso responsable | Sección 7 | Costos de falsos negativos y positivos, revisión de especialistas, esfuerzo de muestreo y validación local. |

Los cinco integrantes y las seis referencias se conservan en el principal. No se atribuyen aprobación del docente, acceso al repositorio, envío por plataforma ni presentación del curso.

## Verificación ejecutada

Desde la raíz del repositorio, con el Python empaquetado:

```powershell
& 'C:/Users/USER/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -X utf8 parcial/pregunta_3_fauna/notebooks/prepare_saved_delivery.py
Push-Location parcial/pregunta_3_fauna
docker run --rm --network none --cap-drop ALL --security-opt no-new-privileges --mount "type=bind,source=$PWD,target=/work" --mount "type=bind,source=$PWD/entregable/data,target=/work/entregable/data,readonly" --workdir /work/entregable --entrypoint python neural-exam-wildlife:0.2 /work/notebooks/execute_saved_delivery.py
Pop-Location
& 'C:/Users/USER/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -X utf8 parcial/pregunta_3_fauna/evidencias/verify_stage_a.py --originals
```

La preparación verificó 44 fuentes, copió 33 archivos portables y comprobó los tres niveles. La validación devolvió `passed`: 33 archivos de datos, 91 originales únicos (incluidos las 44 fuentes, los cinco retornos humanos y las predicciones vinculadas por el protocolo), conteos exactos, CSV y tablas coincidentes, notebooks de 16 y cuatro celdas sin errores, principal de diez páginas y anexo de dos. El notebook integrado se ejecutó mediante el comando completo de [LEEME](../entregable/LEEME.md), en un contenedor sin red ni GPU y con datos montados en solo lectura. Este control no recalcula AP ni bootstrap; valida los resultados completados y preserva su precisión.

Compilación provisional, desde `pregunta_3_fauna` (crear antes `evidencias/latex_provisional` si no existe):

```powershell
docker run --rm --network none --mount "type=bind,source=$PWD/fuentes_informe,target=/source,readonly" --mount "type=bind,source=$PWD/evidencias/latex_provisional,target=/build" --workdir /source --entrypoint latexmk neural-exam-wildlife-v6-reports:latest -pdf -silent -interaction=nonstopmode -halt-on-error -outdir=/build main.tex anexo_occlusion.tex
```

Se revisaron visualmente las diez páginas del principal y las dos del anexo a 110 dpi. Las páginas que cambiaron en la última composición se inspeccionaron nuevamente; las restantes mantuvieron hashes idénticos de renderizado. No se observaron recortes, superposición ni glifos ausentes. La composición original deja espacio libre en páginas dedicadas a figuras. Los logs de LaTeX muestran advertencias de destinos duplicados heredadas del formato del principal, sin afectar cifras o visualización; B2 debe verificar los enlaces al exportar desde ODT.

## Preservación y configuración

Las 44 fuentes físicas del checkout original coinciden con `source-freeze.json`. En este worktree, dos coinciden byte a byte y 42 difieren solo por CRLF; las 44 coinciden tras normalizar finales de línea. Ninguna fue modificada. Las copias portables usan `.gitattributes` para evitar conversiones de bytes. `procedencia.json`, `fuentes_congeladas.json`, `integridad_datos.json` y las constancias de ejecución documentan los hashes.

El worktree real es `C:/Users/USER/.codex/worktrees/34ab/maestria-ia-uni-ciclo-2-redes-neuronales`, rama `codex/fauna-cierre-occlusion`. La creación solicitó `gpt-6-astra` y esfuerzo `xhigh`; velocidad estándar heredada (`default`), sin parámetro individual en la herramienta. Son valores solicitados/documentados en el chat principal, no una certificación independiente de la configuración efectiva del servicio.

La base es `0dc2fd20f424941919c38ff213894541e60ab200`; el commit común `8bb5a39` fue incorporado como `4591a0f`. No se publicó ni se mezcló con main. Revertir esta etapa consiste en retirar `pregunta_3_fauna/` y restituir únicamente `reports/wildlife/main.tex`, `discussion.tex` y la tabla nueva `occlusion_tables.tex`; no afecta fuentes experimentales ni originales externos.

## Traspaso a B2

Usar `fuentes_informe/main.tex`, sus cuatro figuras originales y las tablas editables. El anexo se compone desde `anexo_occlusion.tex` y sus dos archivos de tablas. No convertir páginas PDF en imágenes. Conservar el máximo de diez páginas del principal, los cinco integrantes y las cifras/limitaciones; exportar los PDF finales desde sus ODT. Los PDF provisionales en `evidencias/` sirven exclusivamente para cotejo visual y de contenido.

El flujo científico común, la migración definitiva y la publicación pertenecen a las etapas coordinadas por el chat principal. No quedan revisiones humanas adicionales como requisito para la cohorte aceptada.
