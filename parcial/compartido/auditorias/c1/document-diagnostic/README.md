# Diagnóstico de conservación de contenido de B

La pérdida inicial del anexo se produjo al insertar texto con controles de fuente en Writer: no fue una diferencia de orden de tablas ni únicamente un problema de exportación PDF. Esta revisión leyó B y conservó snapshots en el worktree C1; no modificó sus documentos, scripts ni configuración. B trabajó en paralelo y 12 archivos cambiaron de hash después de la captura: los hallazgos se refieren a los hashes de `all-snapshot-manifest.json`, no a una aprobación de su estado posterior.

**Actualización posterior:** el PDF de ejemplo actual con SHA `aba450a722f020d6f5df300815095cd1915e585918d4c84b2de4e350be7b5c05` ya conserva los ocho valores de CLAHE en la página 9. Se contrastó ese SHA y su texto; el coordinador confirmó visualmente encabezados, grupos y todas las filas. El defecto de Tabla 8 descrito abajo es histórico y quedó resuelto en esa revisión. El anexo volvió a cambiar: el SHA observado fue `2a407a54b259f3f2aebf2b3eff5975d7b32bdff17c893768b393bb3dc8c3b967`, distinto del prefijo `ff05c121` aprobado por el coordinador. Sus marcadores 398, diferencias y dos firmas siguen presentes; no se transfiere la aprobación visual a un SHA diferente. Los problemas de geometría de fórmulas de ejemplo p4/p5 e inundaciones p3 siguen a cargo de B, según la evidencia del coordinador; no se duplicó su triage ni se emitió aprobación final de esos documentos.

## Causa comprobada del anexo

`glyph-cause.json` muestra diez renglones con U+001C o U+0015. Los diez están en el PDF fuente y en el payload de `initialize_exam_documents.extract_document`; ninguno aparece completo en el XML del ODT `anexo-before-glyph-correction.odt`. Los diez aparecen en el ODT corregido. Los controles representan, en esta fuente inspeccionada, la ligadura `fi` y el guion. La fuente PDF usa fuentes Type3 F33/F35; el código de carácter extraído no equivale a Unicode válido para XML.

Ejemplos precisos:

| Fragmento fuente afectado | Control | Antes en XML | Después en XML |
| --- | --- | --- | --- |
| 398 notas ausentes y `justificaciones` | U+001C | Renglón omitido | Renglón completo |
| Umbrales 75 %, 25-75 % y 25 % | U+0015 | Renglón omitido | Renglón completo |
| Semilla 42, checkpoint fijo y 200 réplicas definidas | U+001C | Renglón omitido | Renglón completo |
| Diferencias 6.04, 6.31 y 0.28 | U+001C | Renglón omitido | Renglón completo |
| Planificación de 600 casos y advertencia de representatividad | U+001C | Renglón omitido | Renglón completo |
| Etiqueta de firma del análisis | U+001C | Renglón omitido | Renglón completo |

Los dos valores hexadecimales de firma/SHA sí estaban en el ODT anterior: se perdieron etiquetas y explicaciones adyacentes, no ambos hashes. Buscar solamente `42` o `75` tampoco detecta el defecto: esos números pueden permanecer en celdas de otras tablas. El control debe conservar el contexto y la multiplicidad.

`extract_exam_document` y la selección de regiones no eliminan estos diez renglones del payload. La omisión ya existe antes de renderizar el PDF, por lo que la sustitución de fuentes o el recorte de cajas del PDF no explica esta pérdida inicial. En la copia del backend inspeccionada, `write_native_odt_uno.insert_paragraph` ya aplica `.replace("\\x1c", "fi").replace("\\x15", "-")` antes de `insertString` y rechaza controles pendientes. No se duplicó esa corrección.

El anexo corregido inspeccionado tiene cero pérdidas en los contadores normalizados fuente→ODT y fuente→PDF. Se inspeccionaron visualmente sus dos páginas, con muestra 600/582/18, 398 notas, umbrales, 42/200, diferencias, advertencia y trazabilidad conservadas. Esto verifica el contenido capturado; no sustituye la revisión editorial de todos los documentos.

## Defecto distinto en la Tabla 8 de ejemplo, página 9

La copia auditada conserva los ocho valores de las filas CLAHE en su XML, pero no los muestra en el PDF: `0.2107`, `0.9972`, `0.3412`, `0.7102`, `0.1799`, `0.9848`, `0.3420`, `0.7031`. Algunos también aparecen en la prosa, así que una búsqueda de presencia global da un falso positivo. El contador de ocurrencias y la inspección visual detectan la pérdida de las filas.

El payload confirma la causa de layout en `ejemplo-layout-cause.json`:

1. La detección automática divide una tabla de seis columnas en dos tablas parciales: encabezado/ablación y generalización. Las filas restantes se incorporan a un solo párrafo.
2. Ese párrafo contiene baselines 268.464, 279.373, 290.282, 335.114, 346.421 y 357.330 puntos. Hay un hueco de **44.832 puntos** por la región extraída.
3. `initialize_exam_documents` asigna a todo el párrafo la mediana **10.909 puntos**. `write_native_odt_uno` escribe saltos con ese paso fijo: los tres últimos renglones quedan aproximadamente en 301.191, 312.100 y 323.009.
4. La tabla parcial de generalización ocupa y=293.754–327.278 y se inserta después del párrafo. Las posiciones reempaquetadas caen dentro de esa caja. El PDF muestra superposición y omite las filas inferiores; el XML conserva el texto.

La geometría de baselines reempaquetados es una reconstrucción del algoritmo; la pérdida visual y los datos presentes en XML se observaron directamente. Para corregirlo, una tabla nativa única conserva mejor la estructura. Separar el párrafo en fragmentos cuando cambia el hueco vertical evita el reempaquetado, a costa de más marcos. Este diagnóstico no implementó ninguna alternativa en B.

## Control de los ocho documentos

| Documento | Tokens fuente ausentes en XML | Tokens fuente ausentes en PDF | Interpretación |
| --- | ---: | ---: | --- |
| MRI | 8 | 4 | Notación/subíndices de fórmula; el multiconjunto alfanumérico fuente→XML no pierde caracteres |
| Inundaciones | 15 | 10 | Diferencias de serialización de fórmulas; requieren comprobación matemática |
| Examen | 0 | 0 | Cobertura normalizada completa |
| Ejemplo | 22 | 51 | Fórmulas y pérdida visual confirmada en Tabla 8 |
| Fauna borrador histórico | 0 | 0 | Cobertura normalizada completa |
| Inundaciones borrador histórico | 15 | 10 | Diferencias de serialización de fórmulas |
| Fauna | 0 | 0 | Cobertura normalizada completa |
| Anexo de oclusión | 0 | 0 | Cobertura normalizada completa tras la corrección |

Estos contadores son diagnóstico, no una calificación de fidelidad. El script compara multiconjuntos de palabras/números para tolerar orden de celdas, interpreta espacios/saltos ODF, expande ligaduras conocidas, compara caracteres y conserva ubicaciones de líneas candidatas. También utiliza PyMuPDF y pypdf independientemente. Los residuos de fórmulas y las diferencias de extracción se separan de la prosa; no se afirma que todas las ecuaciones sean correctas por superar una prueba de texto. Los hashes antes/después impiden aceptar una lectura mientras otro proceso modifica los archivos.

## Repetición sin modificar B

Con la imagen local `exam-document-renderer:b1`, montar B como solo lectura y una carpeta privada como salida. Desde el directorio `/b`, su manifiesto de inicialización ya contiene las rutas de los ocho documentos:

```text
docker run --rm --network none --entrypoint python
  --mount type=bind,source=<worktree B>,target=/b,readonly
  --mount type=bind,source=<worktree C1>/parcial/compartido/scripts,target=/scripts,readonly
  --mount type=bind,source=<carpeta privada C1>,target=/qa
  -w /b exam-document-renderer:b1
  /scripts/audit_document_content.py
  /b/parcial/compartido/auditorias/documentos_b1/inicializacion.json
  --output /qa/coverage.json
```

La imagen contiene PyMuPDF 1.26.7, MuPDF 1.26.12 y pypdf. No se añadieron dependencias al entorno científico congelado. Para eliminar carreras con B, repetir sobre copias privadas fijadas por hash, como se hizo en este diagnóstico. El código de salida cero indica que la comparación se ejecutó; las diferencias del recibo requieren interpretación y no se convierten en aprobación automática.

Pasaron cuatro pruebas aisladas de normalización, orden, multiplicidad y espacios ODF, además de Ruff. Los snapshots/PDF/PNG están en `.runtime/document-diagnostic`, ignorado por Git. Los recibos públicos incluyen hashes, cobertura, prueba de glifos y geometría; no incluyen datos o checkpoints. Reversión: retirar el script de auditoría, sus pruebas y estos recibos, sin alterar fuentes científicas ni documentos B. C2 continúa pendiente del SHA autorizado de D2.
