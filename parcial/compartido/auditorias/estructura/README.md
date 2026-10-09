# Diseño de migración del examen — etapa D1

El inventario y el mapa están completos. **No se ha trasladado ni eliminado ningún archivo del examen.** La ejecución D2 queda pendiente de los SHA integrados de A+B+C que enviará el chat principal. Este diseño conserva resultados, código, revisiones humanas y versiones históricas; no requiere nuevos entrenamientos ni inferencia.

## Evidencia verificable

| Elemento | Resultado de D1 |
|---|---|
| Worktree | `C:/Users/USER/.codex/worktrees/b9e3/maestria-ia-uni-ciclo-2-redes-neuronales` |
| Rama | `codex/examen-estructura-d1` |
| Base remota | `0dc2fd20f424941919c38ff213894541e60ab200` |
| Auditoría incorporada | `8bb5a39` mediante cherry-pick; SHA local `7bd7bece23638cb990ab7397b08796b11bf8a4cc` |
| Archivos tracked de la base / inventariados | 913 / 932; el commit de auditoría añade 19 rutas |
| Plan actual | 1 conservar, 335 trasladar, 596 copias JPEG retirables en D2 |
| Duplicados exactos | 7 grupos; se conservan sus funciones y contextos |
| Versiones locales respaldadas | 64 verificadas contra el manifiesto; 44 rutas ausentes aquí, 13 versiones iguales y 7 con bytes distintos |
| Fuentes congeladas | 44 originales verificados; 42 copias físicas difieren por CRLF y las otras 2 ya coinciden |
| Secretos ignorados | 6 rutas `.env.wildlife*`; no se leyó ni publicó su contenido |
| Configuración solicitada | `gpt-6.1-sol`, esfuerzo `high`; velocidad estándar heredada (`default`) |
| Configuración efectiva | El chat principal verificó `gpt-6.1-sol` / `high` desde `turn_context` y velocidad global `default`; comunicó esa evidencia en seguimiento de D1 |

[inventario.json](inventario.json) registra cada hash físico, grupo duplicado, excepción semántica, copia congelada y versión respaldada. [mapa.csv](mapa.csv) contiene origen, destino, SHA-256, tamaño, operación y motivo de cada archivo. [migration_plan.json](migration_plan.json) es el plan ejecutable, con `stage=D1` y aplicación deshabilitada. [evidencia_d1.json](evidencia_d1.json) resume el alcance; [verificaciones_d1.json](verificaciones_d1.json) conserva resultados reales.

Esta carpeta de auditoría se conserva en su destino aprobado y se excluye de las operaciones al regenerar el mapa para evitar hashes autorreferenciales de sus propios manifiestos. El alcance de 932 corresponde a los archivos existentes antes del commit D1.

## Destinos concretos

| Origen | Destino |
|---|---|
| `MRI_*.md`, estado, notebook y `reports/mri/` | `pregunta_1_mri/{docs,evidencias,notebooks,fuentes_informe}/` |
| `FLOOD_*.md`, `PRESENTACION_FLOOD.md`, estado, notebook y `reports/flood/` | `pregunta_2_inundaciones/{docs,evidencias,notebooks,fuentes_informe}/` |
| `WILDLIFE_*.md`, estado, notebook y `reports/wildlife/` | `pregunta_3_fauna/{docs,evidencias,notebooks,fuentes_informe}/` |
| `src/`, `scripts/`, `configs/`, `tests/`, `audits/` | `compartido/{src,scripts,configs,tests,auditorias}/` |
| `reviews/`, preámbulo y bibliografía comunes | `compartido/plantillas/{reviews,informes}/` |
| `examen-parcial.pdf`, `ejemplo.pdf` y sus ODT cuando se integren | `compartido/referencias/` |
| `delivery/flood/` | `archivo/entregas_anteriores/flood/` |
| Documentos de preparación/revisión y material único de `tmp/` | `archivo/documentos_historicos/` |
| Auxiliares únicos de QA de `tmp/` | `compartido/scripts/qa_historico/` |
| Dockerfile, Compose, dependencias, ejemplos de entorno y `.dockerignore` | `parcial/compartido/`; contexto compartido con fuentes congeladas intactas |
| README general | Raíz de `parcial/`; índice del examen |

En D2 cada pregunta debe conservar además `README.md` y `entregable/` con informe principal ODT/PDF, notebook final, `LEEME.md` y `verificacion.json` provenientes de A+B+C. Fauna añade anexo ODT/PDF, `04_occlusion_582.ipynb` y `metricas_occlusion.csv`. No se fabrica ninguno de estos productos en D1. `archivo/manifiesto.json` y `archivo/referencias_originales/` se completarán al integrar las versiones finales y su procedencia.

La ubicación del Dockerfile, Compose y dependencias en `compartido/` corrige el supuesto inicial del diseño D1. Con ese contexto, `COPY src`, `COPY configs`, `PYTHONPATH=/workspace/src` y `PROJECT_ROOT` mantienen su relación original con el motor. Los overlays/adaptadores nuevos suministrarán notebooks y fuentes editoriales desde las carpetas de cada pregunta, coordinados con Reproducción. No se necesita cambiar las fuentes congeladas para resolver el traslado del motor.

`configs/default.toml` y `configs/flood.toml` tienen los mismos bytes, pero el primero es el fallback de `load_settings()` y el segundo una entrada explícita de Flood. Ambas rutas se mantienen. Tampoco se deduplican logs que documentan ejecuciones distintas ni figuras citadas por varios nombres.

Los auxiliares, el notebook con notas de revisión, CSV de exportación, protocolos, instantáneas de revisión offline, logs y vistas renderizadas de `tmp/` se conservan como archivos históricos. Estar en `tmp/` no demuestra que puedan eliminarse sin pérdida. Los archivos de `preserved-before` se inventariaron sin copiarlos: una versión única debe incorporarse al destino activo; si A+B+C la reemplaza, se archivará bajo `archivo/documentos_historicos/preservado_2026-10-08/`. La coincidencia exacta permite conservar solo su referencia de procedencia. No se sobrescribe una versión distinta.

## Reconstrucción de los JPEG retirables

El protocolo tiene 600 casos y **596 fotografías únicas**: varias selecciones comparten una fotografía. Para cada JPEG, el mapa registra `reconstruction.source` en `E:/Kaggle/exam-output-wildlife-v6-occlusion-sample-600/images/`, el SHA-256 esperado y la ruta de reconstrucción. Las 596 copias externas coinciden byte por byte en D1.

En D2 el script vuelve a comprobar la fuente externa antes de retirar cada copia local y respalda la copia en el propio worktree. Para reconstruir una fotografía se copian sus bytes a la ruta registrada y se verifica el hash; no se vuelve a muestrear, no se modifica un CSV humano ni se ejecuta inferencia. La instantánea histórica offline se archiva con su protocolo, CSV, HTML y manifiesto: el CSV actual aceptado no sustituye la instantánea histórica. El paquete original histórico declaraba 0 revisiones completas; no representa la aceptación posterior de 582 casos.

Los secretos ignorados siguen fuera de la migración y fuera del versionado. Los datos y checkpoints grandes permanecen en E:. No se altera ningún original científico ni el checkout original.

## Seguridad del script y verificaciones

`migrate_structure.py` resuelve rutas absolutas y rechaza escapes, enlaces, otro worktree, archivos cambiados y colisiones. Solo usa operaciones nativas Python; no combina shells ni utiliza limpieza de Git. Su modo predeterminado comprueba el plan y produce cero mutaciones. D1 rechaza `--apply` antes de crear un respaldo.

La aplicación D2 requiere plan regenerado, `stage=D2`, `apply_authorized=true`, al menos tres commits integrados presentes como ancestros, y las 44 fuentes físicamente idénticas a los originales. Antes de modificar crea `.migration-recovery-d2/` en su propio worktree con copias verificadas y recibo. No sobrescribe destinos ni borra recursivamente. Un recibo existente obliga a inspeccionar el estado y evita repetir una aplicación parcial.

Comandos ejecutados desde este worktree, con Python empaquetado y `-X utf8`:

```powershell
& C:/Users/USER/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe -X utf8 parcial/compartido/auditorias/estructura/build_inventory.py
& C:/Users/USER/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe -X utf8 parcial/compartido/auditorias/estructura/migrate_structure.py
& C:/Users/USER/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe -X utf8 -m unittest discover -s parcial/compartido/auditorias/estructura -p test_migration_safety.py -v
```

Resultado: inventario de 932 archivos y 64 versiones verificadas; validación del plan aprobada con cero mutaciones; 7 pruebas de seguridad aprobadas. La validación con archivos reales constituye el ensayo de entrada del script. No se ejecutó el recorrido destructivo completo: D1 lo prohíbe. No se ejecutaron pruebas científicas ni notebooks, entrenamiento o inferencia.

La reversión de D1 se limita a esta carpeta de auditoría y su commit: no modifica resultados ni fuentes existentes. D2 tendrá una frontera de reversión distinta, respaldada en `.migration-recovery-d2/receipt.json`; ante un fallo, el chat integrador debe verificar hashes antes de restaurar fuentes y retirar destinos creados. El script no automatiza una reversión destructiva.

## Siguiente etapa

1. Recibir SHA integrado de A+B+C y actualizar **este** worktree. Regenerar mapa y hashes, incorporando archivos nuevos y las 64 versiones locales respaldadas; el plan D1 actual no incluye operaciones de copia externa y no basta para ejecutar D2.
2. Restaurar aquí las 44 fuentes desde `receipts/source-freeze/` sin transformación de texto; proteger cada destino congelado con atributos Git `-text`. Las 42 diferencias actuales se resuelven copiando bytes originales, no normalizando todas las fuentes: las otras 2 conservan CRLF en el original. Añadir adaptadores en archivos nuevos, coordinados con el responsable del motor común.
3. Resolver los cambios de comandos y tests de [actualizaciones_necesarias.md](actualizaciones_necesarias.md), comprobar las colisiones con productos integrados y generar el manifiesto histórico. Los originales de E: siguen en solo lectura.
4. Validar los hashes, preparar un plan D2 con las versiones integradas y aplicar únicamente en este worktree. Registrar hashes finales, pruebas de rutas y resultado de entrega. No publicar ni mezclar con main.
