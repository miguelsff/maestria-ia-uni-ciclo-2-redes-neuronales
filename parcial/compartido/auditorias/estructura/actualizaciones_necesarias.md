# Rutas y verificaciones que deben actualizarse en D2

La migración cambia la ubicación física del motor y de los productos de cada pregunta. Estos puntos fueron identificados mediante lectura del código y búsqueda de referencias; **no se modificaron en D1**. Las 44 fuentes congeladas deben conservar sus bytes; los ajustes se resuelven mediante adaptadores nuevos cuando afecten archivos congelados.

| Punto comprobado | Ajuste requerido y verificación posterior |
|---|---|
| `parcial/pyproject.toml`: paquetes en `src`, tests en `tests`, pythonpath `src` | Trasladar a `parcial/compartido/pyproject.toml`; sus rutas internas permanecen válidas. Instalar editable desde `parcial/compartido` en entorno aislado y comprobar importación/recopilación. |
| Dockerfile congelado: `COPY src`, `COPY configs`, compilador `/workspace/scripts` | Trasladar intacto a `parcial/compartido/Dockerfile` y usar contexto `parcial/compartido`. `COPY src/configs` permanece válido; overlays/adaptadores suministran fuentes editoriales desde cada pregunta. No reconstruirlo como prueba científica. |
| `compose.yaml`: `PYTHONPATH=/workspace/src`, notebooks `/workspace/scripts/start_notebooks.py` | Trasladar a `parcial/compartido/compose.yaml`; mantener `/workspace` como montaje del motor compartido, conservando `/workspace/src` y `/workspace/scripts`. Overlays nuevos suministran notebooks por pregunta; verificar configuración sin ejecutar servicios científicos. |
| `compose.wildlife*.yaml` congelados: configuración `/workspace/configs/` | Trasladar intactos a `parcial/compartido/`; las rutas congeladas siguen siendo coherentes con ese contexto. Crear overlays para notebooks/informes por pregunta y verificar hashes antes y después. |
| `src/neural_exam/config.py` congelado: `PROJECT_ROOT = parents[2]` | Tras traslado resuelve `compartido`, donde también están configs, Dockerfile, requirements y Compose. El motor conserva su layout sin editar este archivo. Adaptadores nuevos resuelven raíz editorial y notebooks de cada pregunta. |
| `scripts/freeze_wildlife_sources.py` congelado y `recover_wildlife_evaluation.py` | No refabricar recibos congelados ni ejecutar recuperación científica. Validar hashes mediante traducción de rutas origen→destino y fuente original. |
| `scripts/run_{mri,flood,wildlife}.ps1` y `evaluate_occlusion_sample.ps1` | La raíz deducida ahora es `compartido`; scripts, tests y configuración conservan sus relaciones. Notebooks/finalización editorial pasan a cada pregunta: usar nuevas entradas de cierre y overlays. `run_wildlife.ps1` sigue congelado. No ejecutar entrenamiento/inferencia. |
| `scripts/start_notebooks.py`, `populate_{mri,wildlife}_notebook.py`, `populate_notebooks.py` | Resolver notebooks de las tres preguntas. No ejecutar estos generadores sobre notebooks finales integrados: podrían reemplazar resultados/edición. |
| `scripts/compile_reports.py`: busca `reports/` junto a scripts | Separar fuentes por pregunta y plantillas comunes. El informe final es ODT y el PDF se exporta desde él; el compilador LaTeX queda como mecanismo histórico. |
| `reports/*/main.tex`, preámbulo/bibliografía e inclusiones relativas | Mantener fuentes históricas utilizables con sus includes; actualizar invocación fuera de originales congelados. No modificar PDF/ODT final para hacerlo coincidir con LaTeX histórico. |
| `scripts/package_occlusion_offline.py`: `reviews/occlusion_offline.{html,js}` | Resolver `compartido/plantillas/reviews/`; el empaquetador es de solo lectura sobre CSV original. No regenerar la instantánea histórica con el CSV aceptado actual. |
| `scripts/wildlife_occlusion_sample.py`: plantilla y comprobación congelada por ruta | Adaptar rutas con tabla de migración y preservar protocolo/aceptación. No exigir notas adicionales ni nueva revisión humana. |
| `tests/test_occlusion_offline.cjs`: `../reviews/occlusion_offline.js` | Cambiar a `../plantillas/reviews/occlusion_offline.js`; ejecutar control JavaScript sin navegador ni inferencia. |
| `tests/test_occlusion_offline.py`, `test_occlusion_sample.py` | Ajustar paths de importación de scripts y plantillas; ejecutar pruebas sintéticas pertinentes en entorno aislado. |
| `tests/test_wildlife_delivery.py` | Cambiar ruta del compilador; controlar estados históricos separadamente del contrato ODT/PDF de entrega. |
| `tests/test_wildlife_recovery.py`, `test_wildlife_corrections.py` | Rutas de bootstrap, recuperación, configs, fuentes y recibos. Recopilar y seleccionar pruebas que no disparen entrenamiento/inferencia nueva. |
| `tests/test_utilities.py` | Incluye comprobaciones de reanudación que pueden entrenar modelos sintéticos; no ejecutarlas automáticamente bajo el alcance de cierre. |
| `tmp/*/{check_package.py,check_browser_export.py,analyze.py}` | Se preservan en `scripts/qa_historico`; su cálculo `parents[...]` y paths heredados dejarán de ser válidos. Documentar o adaptar antes de reutilizar. |
| Notebooks fuente | Paths relativos pueden depender de `PROJECT_ROOT`; aplicar ajustes en copias editoriales nuevas sin reejecutar celdas. Verificar JSON y referencias estáticamente. |
| `README.md` de raíz del repositorio y de `parcial`, docs por pregunta | Actualizar tabla y enlaces activos del motor, notebook, informe y estados. Conservar el texto de auditorías históricas con índice de equivalencias; sus afirmaciones fechadas no son estado actual. |
| `.gitignore`, `.dockerignore`, `.env*.example` | Mantener secretos y datos fuera del versionado/contexto; ampliar ignores para temporales y recuperación propios. No copiar `.env.wildlife*` originales. |
| `.gitattributes` | Marcar cada destino congelado `-text`; no convertir CRLF indiscriminadamente. Verificar bytes físicos y blobs Git de los 44 archivos. |
| Nuevas pruebas/adaptadores A+B+C | Agregar su inventario después de recibir SHA; no asumir rutas ni efectos antes de inspeccionarlos. |

## Comandos de validación previstos

Desde `parcial/compartido/`, conservar la recopilación `python -m pytest --collect-only tests` y el linter `python -m ruff check src scripts tests`, usando allí el `pyproject.toml` trasladado. La selección funcional se limita a controles sintéticos que no ejecuten entrenamiento ni inferencia nueva. La prueba de seguridad D1 se mantiene en `auditorias/estructura/`.

Verificar los overlays desde `parcial/compartido/` con `docker compose ... config` y las importaciones con un entorno aislado cuyo `PYTHONPATH` apunte a `parcial/compartido/src` en el host y `/workspace/src` dentro del contenedor. No arrancar runners, notebooks ni servicios de recuperación sobre salidas originales. Verificar los notebooks como JSON, los manifiestos y hashes como archivos, y los documentos finales mediante los recibos de A+B+C.

Los comandos anteriores son **pendientes de D2**, no resultados atribuidos a D1. El inventario D1, su validación y siete pruebas de seguridad sí fueron ejecutados; sus salidas exactas están en `verificaciones_d1.json`.
