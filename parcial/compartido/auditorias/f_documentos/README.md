# F: exportación documental desde clon limpio

**Resultado: aprobado.** Los ocho maestros de D2 producen 67 páginas idénticas a los PDF finales en píxeles RGB, texto y dimensiones. Se verificaron ediciones reales en ocho copias temporales mediante Writer y, después, mediante el exportador publicado. Los ocho ODT y los 18 PDF versionados del examen conservan sus hashes; no se sustituyeron documentos finales.

## Evidencia

| Control | Resultado | Recibo |
|---|---|---|
| Clon independiente con rutas largas, base `6b9db09ad76bfd698775515cabdbbdd9987a9cde` | 26 hashes coinciden con Git D2 | [entrada_clon.json](entrada_clon.json) |
| Exportación de los maestros existentes | 8 PDF; maestros intactos | [exportacion_originales.json](exportacion_originales.json) |
| Edición nativa en copias | 8 párrafos, 8 celdas y 4 ecuaciones aplicables; cambios visibles | [edicion_real.json](edicion_real.json) |
| Exportador publicado sobre las copias editadas | 8 PDF; copias intactas | [exportacion_copias_editadas.json](exportacion_copias_editadas.json) |
| Comparación con los PDF aprobados | 67/67 páginas sin diferencias a 120 dpi con Poppler | [comparacion_exportaciones.json](comparacion_exportaciones.json) |
| Preservación, entorno y cierre | 574 párrafos, 57 tablas y 15 ecuaciones nativas; 26 hashes intactos | [estado.json](estado.json) |

Los informes principales mantienen 9/9/10 páginas, los cinco autores en la primera página y las referencias en la última. Los 18 PDF del inventario, incluidos originales y archivos históricos, suman 153 páginas; las 67 páginas comparadas corresponden a los ocho PDF asociados a los maestros.

Las nuevas exportaciones tienen hashes binarios distintos y fechas de creación distintas. La comparación comprueba igualdad de texto, dimensiones y píxeles; no afirma identidad binaria ni sustituye los PDF aprobados. Los recibos B conservan sus rutas históricas: el control F los relaciona con D2 por nombres de maestro únicos y exige sus hashes aprobados.

El renderizador es `exam-document-renderer:b1`, ID `sha256:94b5aaf5c97134beb4a5602355ef97760a148992ac9ec774e6c835638224bd5a`. Los inventarios de paquetes y fuentes coinciden por hash con B y con ambas exportaciones F. La configuración del agente figura como confirmada por el coordinador, sin atribuirle una inspección independiente a este control.

## Repetir la verificación

Desde un clon que contenga estos controles y los maestros aprobados, usar Docker con el renderizador indicado. Los hashes de entrada están fijados a D2. Los comandos escriben únicamente recibos y salidas temporales; no inicializan ni regeneran los ODT.

```powershell
$taskMount = "type=bind,source=$((Get-Location).Path),target=/work"
./parcial/compartido/scripts/export_exam_documents.ps1 -InputDir parcial -OutputDir tmp/F/exported-baseline

docker run --rm --network none --mount $taskMount --workdir /work --entrypoint /usr/bin/python3 exam-document-renderer:b1 `
  parcial/compartido/scripts/verify_native_document_editability.py `
  parcial/compartido/auditorias/documentos_b1/inicializacion.json `
  --output parcial/compartido/auditorias/f_documentos/edicion_real.json --copies tmp/F/edited-copies

./parcial/compartido/scripts/export_exam_documents.ps1 -InputDir tmp/F/edited-copies -OutputDir tmp/F/exported-edited

docker run --rm --network none --mount $taskMount --workdir /work --entrypoint /usr/local/bin/python exam-document-renderer:b1 `
  parcial/compartido/scripts/verify_document_exports.py `
  --inventory parcial/compartido/auditorias/f_documentos/entrada_clon.json `
  --manifest parcial/compartido/auditorias/documentos_b1/inicializacion.json `
  --approved-receipt parcial/compartido/auditorias/documentos_b1/exportacion_documentos.json `
  --baseline tmp/F/exported-baseline --edited tmp/F/exported-edited `
  --editability parcial/compartido/auditorias/f_documentos/edicion_real.json `
  --pixels tmp/F/pixel-check --output parcial/compartido/auditorias/f_documentos/comparacion_exportaciones.json
```

El control falla ante diferencias de página y deja el detalle y las imágenes de diferencia en `tmp/F/pixel-check`. Su atributo LF conserva el hash del nuevo script al clonar en Windows. No reevalúa resultados científicos: comprueba exportación, edición y preservación documental. Las imágenes y copias temporales quedan fuera del commit. Para revertir F basta retirar este directorio de recibos, `../../scripts/verify_document_exports.py` y su nueva regla local en `../../scripts/.gitattributes`.
