# Guion de presentación: segmentación de agua con Sentinel-1

Versión para revisión del grupo. Presentación en clase: **30 de octubre de 2026**. Duración propuesta: ocho minutos más preguntas. La asignación siguiente es una propuesta organizativa, no una constancia de responsabilidades ya realizadas.

UNI · Unidad de Posgrado — Facultad de Ingeniería Industrial y de Sistemas.

| Integrante | Tiempo propuesto | Tema |
|---|---:|---|
| Chilón Cruzado Josué | 1 min | Problema, objetivo y datos SAR. |
| Concepción Tiza Miguel Ángel | 1,5 min | Particiones, exclusión de Bolivia y normalización. |
| Neyra Dianderas Enzo Israel | 1,5 min | U-Net propia, pérdidas y supervisión débil. |
| Tello Guerra Omar Salvador | 2 min | Resultados, matriz de confusión y generalización. |
| Alexander Vladimir Paco Cornejo | 2 min | Tres fallos, hipótesis, límites y reproducción. |

## 1. Problema y alcance

El objetivo es segmentar agua píxel a píxel usando VV/VH de Sentinel-1. El radar permite observar bajo nubosidad, pero una señal oscura no identifica agua por sí sola. El experimento mide agua observada: no separa agua permanente de inundación nueva ni estima daño o profundidad.

Mostrar la introducción del PDF y señalar los dos factores estudiados: fuente de supervisión y pérdida.

## 2. Datos y particiones

Sen1Floods11 v1.1 tiene 446 chips manuales totales. No se entrenó con los 446: los CSV originales son 252 train, 89 validación, 90 test y 15 Bolivia. Después de excluir chips totalmente nodata quedan 251/86/88/15. Se usan 4.150 débiles válidos, sin Bolivia ni solapamientos manuales. El inventario guardado tiene 4.384 débiles, frente a 4.385 en el enunciado; la diferencia se documenta.

Bolivia no participa en entrenamiento, normalización, validación ni selección. La normalización utiliza solo los píxeles válidos del entrenamiento manual, con clipping fijo de −50 a 10 dB. La integridad pasó para 9.216 archivos y las 618 predicciones persistidas se auditaron contra las máscaras y la geometría originales.

## 3. Modelo y ablación

La U-Net se implementa directamente en PyTorch, con dos canales de entrada, conexiones de salto y un logit por píxel; tiene 7.762.753 parámetros. Se cruzan manual y manual+débil con BCE, Dice y BCE+Dice: seis corridas completas. El peso débil es 0,3 por chip. Se usa semilla 42, batch cuatro, AdamW, hasta 50 épocas y paciencia diez. Cada checkpoint se elige por mínima pérdida de validación manual; el umbral es 0,5.

Explicar que la pérdida ignora nodata. Mostrar el diseño del modelo y las curvas de validación del PDF.

## 4. Resultados y generalización

Mostrar las tablas y figuras del PDF. La IoU promedio por chip ID está entre 0,3251 y 0,4100; Bolivia, entre 0,2690 y 0,4790. Añadir débiles mejora la IoU en cero de tres pérdidas en ID y dos de tres en Bolivia. Esto es una observación descriptiva: manual y mixto tienen 63 y 1.101 pasos por época y solo una semilla.

El baseline manual+BCE tiene IoU por chip ID=0,3939 y Bolivia=0,4555; brecha ID−Bolivia=−0,0616. Una brecha negativa significa mayor IoU externa en este conjunto. La IoU global ID=0,6642 responde a otra agregación. F1=0,7982, precisión=0,8538 y recall=0,7495.

En su matriz ID hay TP=1.884.956, FP=322.880, FN=630.052 y TN=17.606.071. Se omite 25,1 % del agua. Un 1,8 % de FP sobre píxeles no acuáticos no equivale a comisión: la comisión, 14,6 %, usa como denominador las predicciones de agua.

La IoU principal excluye uniones vacías y reporta cuántas hay; la alternativa con epsilon asigna uno y conserva todos los chips válidos. No ocultar esta diferencia entre variantes.

## 5. Fallos, hipótesis y límites

Mostrar los tres paneles del PDF:

- `Pakistan_70625`: falsos positivos en regiones de baja retrodispersión; 28.116 FP.
- `Pakistan_849790`: agua de referencia con respuesta más alta omitida; 138.480 FN.
- `Bolivia_379434`: fragmentación y errores de borde en cauces; 81,25 % de los errores en la banda morfológica definida.

No afirmar que esos casos prueban sombra, doble rebote o vegetación inundada. Son mecanismos posibles que requieren información independiente.

H1: cambian apariencia y prevalencia del agua. H2: cambian adquisición y sesgo de etiquetas débiles. Quince chips Bolivia de un solo evento, una semilla y presupuestos distintos limitan las inferencias. Un estudio posterior igualaría pasos y repetiría semillas.

Cerrar mostrando el comando `./scripts/run_flood.ps1` y el enlace de la rama publicada. Aclarar que los originales y resultados pesados están fuera de Git, mientras el notebook, las métricas ligeras, figuras y PDF de revisión permiten inspeccionar las cifras.

## Preguntas previsibles

| Pregunta | Respuesta respaldada por el estudio |
|---|---|
| ¿Por qué no se entrenó con los 446 manuales? | Son el total manual, incluyendo validación, test y Bolivia; usarlos todos filtraría evaluación al entrenamiento. |
| ¿Por qué Bolivia puede rendir mejor? | La brecha puede ser negativa; apariencia, prevalencia y composición del evento son hipótesis, no causas demostradas. |
| ¿Mejoran las etiquetas débiles? | En estas corridas mejoran cero de tres pérdidas ID y dos de tres Bolivia; el presupuesto de pasos está confundido con la fuente de supervisión. |
| ¿El notebook volvió a entrenar? | No: ocho celdas ejecutadas leen resultados persistidos con entrenamiento desactivado. Las seis corridas se acreditan por historiales y checkpoints. |
| ¿Falta Sentinel-2? | Es un bono opcional; este trabajo cumple el alcance SAR obligatorio. |

Antes de presentar: aprobar editorialmente el PDF, comprobar el enlace de GitHub y conservar el comprobante de entrega por la plataforma. No registrar esos pasos como completados antes de que ocurran.
