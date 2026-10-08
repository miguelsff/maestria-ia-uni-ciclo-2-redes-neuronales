# Pregunta 3: tercera ave corregida según la referencia humana

V5 reemplaza la tercera caja incorrecta de v4 por el recuadro rojo proporcionado en `ave.png`. En la referencia de 960 × 700 píxeles, sus límites son `(681, 333, 707, 381)`. Las otras dos cajas de esa fotografía se conservan. El registro reproducible está en `configs/wildlife_label_corrections_v5.json`, con el hash de la referencia y la instrucción humana original.

La [galería v5](http://127.0.0.1:8899/label_review.html) muestra la corrección. El CSV actual está en `E:/Kaggle/exam-output-wildlife-v5/wildlife/label_review.csv`: sus 90 fotografías están aprobadas por Miguel, incluida la corregida. Las 89 fotografías sin cambios mantienen su evaluador y aprobación; las notas individuales siguen siendo opcionales. La comprobación y el hash del CSV humano se conservan en `receipts/human-label-review-completion.json`; el agente no modificó esas decisiones.

Guía histórica: para continuar usar [WILDLIFE_V6_RUN.md](WILDLIFE_V6_RUN.md). Para reproducir la preparación v5 con sus fuentes preservadas:

```powershell
.\scripts\run_wildlife.ps1 -Variant v5 -PrepareOnly
```

La salida v4 se conserva íntegra como padre. La referencia original se archiva en `inputs/user-reference/ave.png` de v5. Las particiones, fondos y controles científicos conservan el protocolo de la [guía v4](WILDLIFE_V4_RUN.md).

El usuario definió el alcance: utilizar YOLO existente y ajustarlo, sin reprogramar una red YOLO propia. Se conserva YOLOv8s de Ultralytics y el código propio de datos, evaluación y monitoreo. Esa decisión real se registra en `receipts/architecture-review.json`, con su instrucción literal y autoría de usuario; no se presenta como una autorización del docente ni como prueba de componentes de red implementados por el grupo. Los controles previos quedan satisfechos con las 90 revisiones humanas y esta decisión de alcance. El intento v5 se interrumpió durante el escaneo inicial: Ultralytics intentaba reescribir JPEG con espacios después del marcador final sobre enlaces de solo lectura. No completó ninguna época ni generó checkpoints o métricas científicas. V6 conserva ese intento y corrige el formato de copias de trabajo sin cambiar píxeles.

La comprobación de ejecución contrasta el manifiesto, la etiqueta YOLO materializada y la imagen de la galería con la referencia; también verifica las 89 revisiones auténticas y los hashes del padre. Los resultados históricos anteriores a la última aprobación se registran en `receipts/red-reference-verification.json`; la revisión humana completa consta en `receipts/human-label-review-completion.json`. La prueba del importador de versiones pasó para los tres padres admitidos: 3 pruebas correctas.

Para revertir esta corrección, conservar v5 y usar el snapshot de fuentes y datos congelados de v4. No se modifican sus resultados ni los artefactos de las preguntas 1 y 2.
