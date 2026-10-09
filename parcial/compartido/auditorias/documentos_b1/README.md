# Documentos editables y PDF verificados

Los ocho maestros ODT conservan los informes completos como párrafos de Writer,
tablas con celdas editables y objetos de LibreOffice Math con MathML. Los PDF
finales se exportaron desde esos maestros. Las imágenes son las figuras originales;
ningún documento usa imágenes de páginas completas para reemplazar el texto.

## Editar y exportar

1. Abrir el ODT correspondiente en LibreOffice Writer y guardar los cambios.
2. Ejecutar desde la raíz del repositorio:

   ```powershell
   ./parcial/compartido/scripts/export_exam_documents.ps1 -InputDir parcial -OutputDir parcial
   ```

3. Revisar el PDF y el nuevo `parcial/exportacion_documentos.json`. El recibo
   compara el SHA-256 del ODT antes y después: exportar preserva sus bytes.

El exportador requiere PowerShell 7 y Docker Desktop. Usa LibreOffice y fuentes fijados en una
imagen administrada; no requiere instalar un renderer en Windows. Si falta la
imagen, la construye desde `compartido/plantillas/document_renderer/Dockerfile`.
Cada exportación trabaja sin red, con entrada de solo lectura y perfil temporal.
Un PDF nuevo debe existir antes de sustituir el PDF derivado anterior.

El exportador **no llama al inicializador**, no utiliza TEX ni DOCX intermedios y
no sobrescribe las modificaciones del ODT. Los metadatos de fecha del PDF pueden
cambiar entre exportaciones: se verifican los hashes de cada exportación concreta,
sin afirmar identidad binaria entre PDF exportados en momentos diferentes.

## Entregables completos

| Documento | Maestro | Páginas | Tablas | MathML |
|---|---|---:|---:|---:|
| MRI | `pregunta_1_mri/entregable/informe_mri.odt` | 9 | 6 | 1 |
| Inundaciones | `pregunta_2_inundaciones/entregable/informe_inundaciones.odt` | 9 | 6 | 3 |
| Fauna V6 aceptada | `pregunta_3_fauna/entregable/informe_fauna.odt` | 10 | 12 | 0 |
| Anexo de oclusión | `pregunta_3_fauna/entregable/anexo_occlusion.odt` | 2 | 3 | 0 |
| Enunciado | `examen-parcial.odt` | 7 | 3 | 0 |
| Ejemplo | `ejemplo.odt` | 11 | 8 | 8 |
| Fauna, borrador histórico | `archivo/documentos_historicos/fauna_borrador_inicial/fauna_borrador_inicial.odt` | 10 | 13 | 0 |
| Inundaciones, borrador científico histórico | `archivo/documentos_historicos/inundaciones_borrador_cientifico/inundaciones_borrador_cientifico.odt` | 9 | 6 | 3 |

Las rutas de la tabla son relativas a `parcial/`. Cada ODT tiene su PDF homónimo.
El inventario agrupa los PDF por SHA-256 y evita convertir duplicados. Las fuentes
PDF se conservan por separado; también se preservan los fuentes LaTeX disponibles
de MRI e inundaciones. El informe histórico de fauna conserva sus pendientes
originales y el de inundaciones conserva su autoría incompleta histórica.

## Evidencia de cierre

- [Cobertura completa](cobertura_contenido.json): 574 párrafos y 57 tablas
  comparados contra las fuentes; ninguna diferencia no registrada. Quince
  fórmulas coinciden con el manifiesto y no contienen errores del parser MathML.
- [Edición real](edicion_real.json): los ocho ODT se copiaron, editaron y exportaron.
  Los marcadores de párrafo y celda aparecen en los ocho PDF; los de ecuación,
  en los cuatro documentos con objetos matemáticos. Los maestros no cambiaron.
- [Revisión visual](revision_visual.json): inspección de las 67 páginas renderizadas
  por Poppler, con contacto visual de todas las páginas y revisión ampliada de
  fórmulas, tablas y los párrafos editoriales corregidos.
- [Inventario](inventario_fuentes.json) y [preservación](preservacion_fuentes.json):
  fuentes y copias identificadas por sus hashes.
- [Ajustes editoriales](ajustes_editoriales.json): cuatro párrafos registrados en
  `inicializacion.json` actualizan rutas, revisión base y alcance del cierre.
  El control adicional exige que cada sustitución aparezca completa en su página.

El [diagnóstico independiente](diagnostico_independiente.json), producido con la
herramienta C1 del commit `11d0366`, conserva las diferencias de extracción sin
declararlas automáticamente como pérdida. En enunciado, fauna histórica, fauna
final y anexo la cobertura es completa. En los demás documentos las diferencias
se explican por las cuatro sustituciones editoriales registradas y por la
representación separada de índices, símbolos y fórmulas nativas. Se comprobaron
contra las fórmulas del manifiesto, el texto nativo y las páginas renderizadas.

La comparación de tokens de palabras es un control de cobertura, no una prueba
de disposición visual. Por eso el cierre incluye inspección visual y comprobación
local del texto editorial; no se acepta solo un recuento global de palabras.

## Uso práctico y límites

Los párrafos y tablas están en marcos nativos anclados a cada página para conservar
la paginación. Las ediciones de contenido son reales, pero un cambio extenso puede
necesitar ajustar el marco o la página siguiente. Después de editar, revisar el
PDF. Las figuras conservan sus píxeles; no se reconstruyeron como gráficos editables.

La inicialización fue una operación separada y se niega a reemplazar cualquier
ODT existente. Su manifiesto documenta fuentes, fórmulas y ajustes, pero no es la
fuente de una exportación ordinaria: el maestro vigente siempre es el ODT editado.

Este cierre documental no ejecutó entrenamiento ni inferencia y no acredita
aprobación clínica, aprobación del docente ni envío a la plataforma del curso.
La integración y publicación quedan a cargo del chat coordinador.

## Validación y reversión de esta unidad

La validación enfocada ejecutó `audit_native_document_content.py` sobre los ocho
documentos y terminó con código 0. El escenario de ejecución editó copias mediante
`verify_native_document_editability.py`, exportó sus PDF y verificó todos los
marcadores con `pdftotext`; terminó con código 0. Ruff verificó los siete scripts
nuevos y no encontró errores. El exporter conservó los ocho hashes ODT.

La reversión retira los ocho pares ODT/PDF derivados, los scripts de documentos,
el renderer y sus recibos; restaura los PDF de referencia desde
`archivo/referencias_originales/`. No requiere modificar datasets, notebooks,
resultados guardados, originales de E: ni las 44 fuentes experimentales congeladas.
