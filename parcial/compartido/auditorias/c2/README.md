# C2/F: cierre verificado desde un clon limpio

La aceptación pasó sobre D2 `6b9db09ad76bfd698775515cabdbbdd9987a9cde` más la corrección `d16959ce236cda4f43d25b67a4d70eacba85ff64`. El adaptador reconstruye MRI e inundaciones desde `fuentes_informe/latex_previo_d1` cuando falta `main.tex` en raíz; fauna conserva la fuente A de la raíz. Las fuentes `latex_original` de B y los maestros ODT/PDF siguen separados de la compilación científica histórica.

## Resultado y alcance

| Control | Resultado real |
| --- | --- |
| Checkout con rutas largas y conversión CRLF activadas | Limpio; ninguna reparación posterior de bytes |
| Integridad protegida, bytes físicos y blobs Git | 91 comprobaciones sobre 88 archivos únicos |
| Fuentes congeladas V6 | 44/44 |
| Helpers, prueba y plantilla aceptados de C1 | 7/7; 61 comprobaciones protegidas C1 conservadas |
| Archivos de la entrega portable de fauna | 42 hashes coincidentes con su verificación final |
| Pruebas enfocadas del adaptador | 13 aprobadas; incluyen tres nuevos casos de informes migrados |
| Cierre con `--tests` | Código 0; 23 aprobadas |
| Suite funcional sintética | Código 0; 124 aprobadas, tres omitidas por CUDA, una excluida porque entrena |
| Contenido documental en renderer | Código 0; cuatro aprobadas |
| Revisión offline JavaScript | Código 0; CSV, conflictos y preservación comprobados |
| Notebooks finales | Cuatro esquemas válidos; 8/8/16/4 celdas con ejecución registrada; cero errores |
| Planes MRI, inundaciones y fauna | Tres planes sin ejecutar runners; recursos presentes, sintaxis PowerShell y Compose válidos |

Los grupos de pruebas se superponen: sus cantidades no deben sumarse como pruebas únicas. Los notebooks se inspeccionaron sin reejecutarlos. MRI e inundaciones coinciden con sus entregas originales E; los dos notebooks de fauna y sus datos coinciden con `pregunta_3_fauna/entregable/verificacion.json`. El manifiesto de integridad conserva las rutas y hashes concretos de procedencia.

La firma sigue siendo `6b2983c33abccbc18e293cc3a10ca30a710669c2ba8055a76bb462b3348a13a9` y los resultados conservan SHA-256 `b5ede4e0f8d46aef5b579f64c3b7f7c476b4aeb93eafaeffaafcdd4e0da124f9`: 600 casos originales, 582 aceptados, 18 excluidos, 398 notas opcionales ausentes y 200 réplicas ya existentes por nivel. El cierre no recalculó esas réplicas.

## Clon y comandos

Clon privado conservado en:

```text
C:/Users/USER/.codex/worktrees/959d/maestria-ia-uni-ciclo-2-redes-neuronales/parcial/.runtime/f-clean-clone
```

Se creó con `git -c core.longpaths=true -c core.autocrlf=true clone --no-local --no-checkout` y se abrió una rama nueva desde el commit corregido. La rama de trabajo C2 también se creó directamente desde D2, sin reaplicar semillas A/B/C. Los argumentos completos, directorios, salidas y códigos se encuentran en [commands.json](commands.json).

Desde el clon, con el Python empaquetado indicado en los recibos:

```powershell
python -X utf8 parcial/compartido/scripts/run_exam.py closure --tests
python -X utf8 parcial/compartido/scripts/run_exam.py mri --plan-only --scientific-output <salida nueva MRI>
python -X utf8 parcial/compartido/scripts/run_exam.py flood --plan-only --scientific-output <salida nueva inundaciones>
python -X utf8 parcial/compartido/scripts/run_exam.py wildlife --plan-only --scientific-output <salida nueva fauna>
```

Las salidas científicas nuevas no se crearon. Cada plan produjo únicamente un workspace privado y su recibo. Los tres runners conservaron sus bytes; se analizaron como texto sin ejecutarlos. `docker compose config --quiet` pasó con código cero para cada configuración original. El contexto compartido conserva Dockerfile, paquete, configuración y requisitos; la construcción real de imagen quedó sin ejecutar.

La suite funcional usó la imagen local `neural-exam-wildlife:0.2`, workspace de solo lectura, derivados privados escribibles, CPU y red desactivada. Se invocó `pytest -m "not slow"` excluyendo `test_document_content.py`; esa prueba documental se ejecutó por separado mediante `unittest` en `exam-document-renderer:b1`. La prueba de reanudación que llama `fit` quedó excluida. Los casos de métricas, modelos y workers utilizan tensores, imágenes y predicciones sintéticas; no usan checkpoints científicos ni inferencia sobre datasets reales.

## Recibos

- [summary.json](summary.json): commits, cantidades precisas, imágenes, límites y reversión.
- [clean-clone-integrity.json](clean-clone-integrity.json): comprobaciones físico/Git, fuentes, notebooks y procedencia; estado limpio y hashes de anclas externas antes/después.
- [closure-verification.json](closure-verification.json): firma, estado humano, entorno y resultados conservados.
- [scientific-plans.json](scientific-plans.json) y [runner-static-parsing.json](runner-static-parsing.json): recursos y entrypoints reales de los tres planes.
- [functional-summary.json](functional-summary.json): cantidades y motivos de omisión.
- [commands.json](commands.json): invocaciones y códigos reales; los logs adyacentes conservan las salidas.

Los originales E y el checkout original no se modificaron. Ningún montaje científico del cierre fue escribible. No hubo publicación, merge, entrenamiento, inferencia científica ni bootstrap científico. Las imágenes existentes se identificaron por hash; **la reconstrucción limpia desde Internet y el entrenamiento completo siguen sin verificar**. B realiza en paralelo la revisión documental final de edición, exportación y fidelidad visual; las cuatro pruebas de contenido aquí no sustituyen esa revisión.

Reversión: retirar únicamente la selección de raíz histórica del adaptador, sus tres casos de regresión y la documentación/recibos C2. No alterar fuentes firmadas, datos, notebooks ni documentos maestros. Los recibos C1 y D2 permanecen como antecedentes de sus respectivas etapas.
