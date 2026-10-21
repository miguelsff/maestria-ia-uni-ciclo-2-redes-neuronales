# Fauna V6: entrega, verificación y reproducción

La [entrega vigente](../entregable/LEEME.md) contiene el informe y anexo ODT/PDF, los notebooks ejecutados y las métricas de oclusión. El entrenamiento V6 terminó y la evaluación aceptada utiliza 582 de los 600 casos revisados. Los resultados, revisiones humanas, checkpoints y fuentes congeladas se conservan sin cambios.

## Verificar los resultados existentes

Desde la raíz del repositorio, con el entorno Python y Docker documentados en el [procedimiento compartido](../../compartido/REPRODUCIBILIDAD_CIERRE.md):

```powershell
python parcial/compartido/scripts/run_exam.py closure --tests
```

El comando comprueba las 44 fuentes congeladas, los resultados originales y la entrega portátil. No repite entrenamiento, inferencia ni los 200 remuestreos científicos por nivel. La [matriz por subrequisito](../evidencias/matriz_revision.md) conserva el alcance y las limitaciones de la entrega.

## Insumos que se conservan

| Ubicación en E:/Kaggle | Función |
| --- | --- |
| `exam-input-wildlife/backgrounds` | Cohorte verificada de 1.000 fondos, metadatos originales y recibos de descarga. |
| `exam-input-wildlife/parent-v5` | Once archivos que V6 importó de V5, más dos recibos de aprobación y corrección. |
| `exam-output-wildlife-v6` | Modelo entrenado, predicciones, manifiestos, revisiones, métricas, fuentes congeladas y resultados originales. |
| `exam-output-wildlife-v6-occlusion-sample-600` | Revisión de 600 casos, aceptación de 582, retornos humanos originales y evaluación terminada. |

Los originales CCT20 y las fotografías ECCV permanecen en sus ubicaciones anteriores. `exam-input-wildlife/migration.json` vincula cada insumo trasladado con su origen y SHA-256. Las salidas de Wildlife original y v2–v5 se retiraron después de verificar esas copias; no son necesarias para el flujo vigente.

## Reproducción científica en una salida nueva

```powershell
python parcial/compartido/scripts/run_exam.py wildlife --scientific-output E:/Kaggle/reproduccion-wildlife-nueva --plan-only
```

El plan usa el padre mínimo V5 y la cohorte de fondos conservada; `PARENT_OUTPUT_HOST_DIR` y `CCT_BACKGROUND_HOST_DIR` permiten indicar otros insumos compatibles. La salida nueva debe no existir. Para ejecutar entrenamiento, se sustituye `--plan-only` por `--execute-science`; requiere GPU, entorno y datos compatibles. La limpieza solo comprobó el plan: la reproducción completa desde entrenamiento continúa sin verificarse.

El launcher materializa las rutas históricas dentro de una copia privada. Carga `compose.wildlife.yaml` y todas las capas v2–v6: v2 aporta los fondos y la galería, v4 define la red de esos servicios y v6 selecciona la configuración científica vigente. Estas capas y los runners son fuentes congeladas y conservan sus bytes. Los ejemplos y configuraciones experimentales anteriores que ya no usa V6 se retiraron.

## Revisión y límites

V6 mantiene las 90 aprobaciones auténticas de etiquetas y la corrección de la tercera caja del ave. La evaluación independiente de oclusión conserva 340 casos bajos, 69 medios, 48 altos y 125 ambiguos; excluye 18 casos sin nivel según la decisión humana registrada. Las 398 notas ausentes permanecen ausentes. Esta aceptación no convierte la galería original de 17.620 objetos en una revisión completa: consultar el [protocolo aceptado](WILDLIFE_OCCLUSION_ACCEPTED.md).

La galería local de V6 continúa disponible en `http://127.0.0.1:8901/`, con el montaje de fondos actualizado. Los borradores, guías anteriores y pruebas sintéticas históricas retiradas están identificados en el [registro de limpieza](../../compartido/auditorias/limpieza-2026-10-12/README.md). Los recibos históricos conservados registran su contexto original y no sustituyen los comandos vigentes de esta guía.
