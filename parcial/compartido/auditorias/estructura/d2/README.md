# D2: migración aplicada y verificada

Base integrada: `76557bc5aecb7995f1af0fed130ac22cd66ece82`. Rama: `codex/examen-estructura-d2`. Worktree: `C:/Users/USER/.codex/worktrees/b9e3/maestria-ia-uni-ciclo-2-redes-neuronales`. La rama se creó directamente desde esa base; no se mezclaron semillas ni ramas D1 duplicadas.

El árbol público de `parcial/` contiene únicamente `README.md`, las tres preguntas, `compartido/` y `archivo/`. El respaldo de reversión permanece privado e ignorado en `parcial/.runtime/d2-recovery/`; contiene recibo y bytes de cada archivo trasladado o retirado. La copia original y todos los worktrees ajenos permanecieron fuera de las operaciones.

## Resultado

| Comprobación | Resultado real |
|---|---|
| Inventario integrado | 1233 archivos |
| Versiones respaldadas de D1 | 64 copiadas al archivo, SHA verificado |
| Operaciones | 344 traslados, 293 conservaciones, 596 retiros de JPEG regenerables, 64 copias preservadas |
| Aplicación | 1004 operaciones físicas completadas y verificadas |
| Fuentes V6 congeladas | 44 preservadas byte por byte; originales E también cotejados |
| Protegidos C1 | 61 verificaciones firmadas conservadas |
| Documentos y notebooks finales | 8 maestros ODT, sus PDF y notebooks mantienen hashes; 91 comprobaciones de protección, algunas reiteran rutas C1 |
| Asociación PDF→ODT | 18 PDF conservados, 8 maestros; ninguna asociación pendiente |
| Salidas humanas | 14 entradas de procedencia verificadas: 13 archivos locales y 1 ZIP de 68 MB preservado externamente, sin publicarlo |
| Enlaces activos | README raíz, README de parcial y README/LEEME de las tres preguntas aprobados |
| Metadata documental | Las 8 rutas de fuente y maestro existen; examen/ejemplo resuelven `compartido/referencias` |
| Seguridad | 7 pruebas aprobadas; sintaxis Python nueva verificada |
| Espacios | Archivos nuevos/editoriales aprobados con `cr-at-eol`; el chequeo global conserva advertencias de espacios heredados en fuentes y extracciones históricas |
| Repetición científica | No se ejecutaron entrenamiento, inferencia ni bootstrap; el padre comunicó aprobación independiente de `verify_closure` |

La primera restauración física corrigió 43 conversiones CRLF desde blobs Git previamente cotejados. `update-index --refresh` devolvió 1 por metadatos de stat; el reintento refrescó los 61 archivos y comprobó que sus blobs del índice no cambiaron. La restauración no modificó contenido científico.

## Recibos y manifiesto

- [plan.json](plan.json) y [mapa_d2.csv](mapa_d2.csv): cada origen, destino, SHA-256, tamaño, operación y motivo; los retiros incluyen espejo de reconstrucción en E.
- [physical_before.json](physical_before.json): 44 congelados y 61 protegidos antes de migrar.
- [application.json](application.json): aplicación completa y ubicación de recuperación privada.
- [pdf_odt.json](pdf_odt.json): asociación por hashes de fuente o exportación acreditada por los recibos B; se conservan diferencias entre PDF fuente y PDF nativo.
- [editorial_edits.json](editorial_edits.json): cambios editoriales con hashes anterior/posterior y copias previas.
- [checks.json](checks.json): comandos y salidas reales de tests, controles de enlaces, metadata y fuentes originales E.
- [verificacion.json](verificacion.json): cierre de integridad y configuración confirmada por el principal.
- [Manifiesto del archivo](../../../../archivo/manifiesto.json): operaciones y procedencia consolidada, incluyendo PDF→ODT y ediciones.

Los PDF originales del examen/ejemplo se conservaron en `archivo/referencias_originales`; las referencias de `compartido/referencias` son las exportaciones de B. Las fuentes editoriales anteriores se trasladaron a `fuentes_informe/latex_previo_d1` de cada pregunta para evitar sobrescribir A/B. Las versiones previas de README/LEEME/metadata se conservan en `archivo/documentos_historicos/editorial_pre_d2`; la verificación original de Fauna también está en su carpeta de evidencias. Los recibos originales B/C y las 82 comprobaciones se conservaron.

`inicializacion.json` actualiza únicamente rutas de examen/ejemplo y registra `historical_paths_before_D2`; la copia previa y los recibos de exportación históricos conservan el origen. Ningún ODT fue editado o exportado nuevamente. Los scripts documentales reciben esa metadata como argumento; no necesitan alterar los helpers firmados.

## Comandos ejecutados

Con Python empaquetado y `-X utf8`, desde la raíz de este worktree:

```powershell
python parcial/compartido/auditorias/estructura/d2_migration.py prepare
python parcial/compartido/auditorias/estructura/d2_migration.py validate
python -m unittest discover -s parcial/compartido/auditorias/estructura -p test_migration_safety.py -v
python parcial/compartido/auditorias/estructura/d2_migration.py apply
python parcial/compartido/auditorias/estructura/d2_finalize.py
```

Resultado: preparación y validación aprobadas con cero mutaciones del plan; siete tests aprobados; aplicación completa; finalización aprobada con 18 PDF, ocho maestros y 91 comprobaciones de hashes protegidos. El control real de las operaciones en este worktree constituye el ensayo de integración de D2; no se atribuye reproducción científica nueva. `checks.json` conserva la ruta exacta del ejecutable usado para los tests.

La reversión debe usar el mapa y `.runtime/d2-recovery/receipt.json`, cotejar hashes, restaurar únicamente orígenes de D2 y retirar únicamente destinos creados por D2. No se automatizó una limpieza recursiva ni se tocó el respaldo externo. La restauración del commit D2 deberá preservar productos integrados A+B+C.

## Siguiente etapa

Pendiente F: revisión del chat principal y C/B en copia limpia, con `git -c core.longpaths=true clone ...`. No repetir verificaciones científicas que el principal ya aprobó; comprobar rutas, atributos, hashes protegidos, edición/exportación documental y consultas en el entorno de F. No se publicó, no se mezcló con main y no se acredita aún envío al curso.
