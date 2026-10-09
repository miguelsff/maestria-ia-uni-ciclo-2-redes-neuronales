# Evidencia de la etapa C1

C1 incorpora los scripts aceptados y las copias ejecutadas existentes, protege sus bytes en Git y prepara un runtime independiente para la migración. Los resultados originales se verificaron en solo lectura. No hubo entrenamiento, inferencia, publicación ni repetición del bootstrap científico.

| Evidencia | Resultado |
| --- | --- |
| `byte-preservation.json` | 44 fuentes verificadas contra el original y el snapshot; 42 conversiones CRLF del checkout corregidas; siete insumos firmados y diez archivos de entregables copiados byte por byte |
| `git-checkout-roundtrip.json` | 61/61 hashes coincidentes tras extracción del índice con `core.autocrlf=true` |
| `verification-native.json` | Firma exacta, 582 aceptados, 18 excluidos, 398 casos sin nota, 200 réplicas existentes por nivel |
| `verification-container.json` | Mismos controles, con versiones de las cuatro dependencias congeladas comprobadas |
| `accepted-tests-output.txt` | Tres pruebas del flujo aceptado aprobadas |
| `tests-output.txt` | 15 pruebas aprobadas: flujo aceptado, muestreo y adaptadores de migración |
| `notebook-output.txt` | Notebook de oclusión reejecutado sobre los resultados guardados; cuatro celdas sin errores |
| `evidence.json` | Worktree, configuración del chat, identidad de imagen, comandos ejecutados, limitaciones y pendientes |
| `source-unit.json` | Unidad de incorporación, evidencia y frontera de reversión |

Los notebooks MRI e inundaciones se copiaron con todas sus salidas: ocho celdas ejecutadas por notebook, cero errores registrados. No se volvieron a ejecutar. La reejecución de oclusión produjo únicamente `parcial/.runtime/derived/04_occlusion_582_replayed.ipynb`; no reemplazó el entregable original.

La extracción del índice valida las reglas de conversión y los bytes comprometidos. No es una clonación remota ni demuestra reproducción científica completa. C2 deberá incorporar el SHA de la migración, preparar una nueva copia limpia, comprobar las rutas definitivas y repetir los controles pertinentes. Los informes y ODT pertenecen a las etapas de documentos.

La reversión de la unidad de adaptadores elimina `closure_runtime.py`, `verify_closure.py`, sus pruebas, el Compose, el ejemplo de entorno y esta documentación/evidencia; deja intactos los scripts aceptados, los entregables incorporados y los originales en E:. Las carpetas privadas `.runtime` están ignoradas y no forman parte de los commits.
