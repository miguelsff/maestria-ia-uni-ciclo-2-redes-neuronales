# Dictamen de la entrega publicada · Grupo 3 · MIA-203, UNI

**APROBADO.** La versión pública de PeruFoodNet identifica el mismo conjunto de entregables que se revisó. GitHub registra los seis adjuntos académicos en estado `uploaded`. Sus tamaños y huellas SHA-256 coinciden con los archivos locales verificados. [Publicación v1.0-final](https://github.com/miguelsff/maestria-ia-uni-ciclo-2-redes-neuronales/releases/tag/v1.0-final)

## Identidad de la versión

La etiqueta anotada `v1.0-final` apunta al commit `3381c69548ee895e8ab9a1e02b29b9a6310cbf83`, que en la fecha de revisión también encabezaba la rama predeterminada `main`. La entrega contiene 86 archivos en tres carpetas numeradas. El historial público conserva el código de entrenamiento en `0eeea4d` y la selección fijada antes de la prueba en `962e250`. Se comprobó la identidad de los árboles Git de ambas etapas y de la entrega final.

| Entregable | Comprobación |
|---|---|
| 3.1 · Artículo técnico en formato IEEE | PDF de 7 páginas, dentro del máximo de 12, con las cinco contribuciones confirmadas y la declaración de uso de IA. Los 10 archivos del ZIP LaTeX coinciden byte por byte con las fuentes de la copia local en Windows. En los textos versionados se comprobó el mismo contenido teniendo en cuenta los saltos de línea. La compilación autónoma verificada en la revisión del artículo reproduce sus siete páginas. |
| 3.2 · Diapositivas y sustentación | PDF de 20 diapositivas con los cinco integrantes en portada y guion de defensa de 30 páginas. El cronograma asigna 25 minutos a la exposición, 5 por integrante, con 7 minutos para resultados. Reserva otros 5 minutos para preguntas. Los tiempos corresponden a la planificación. |
| 3.3 · Código | ZIP íntegro con 72 archivos que coinciden byte por byte con la versión Git etiquetada. Al extraerlo, el único README del proyecto, en español, queda en la raíz. Incluye dos cuadernos, código, configuración, evidencias y una foto de ejemplo. El inventario de entrega está libre de planes internos, archivos de reserva, registros de ejecución auxiliares y cachés. |

Ambos cuadernos indican la URL publicada del ZIP de código y la misma fuente de PeruFoodNet v3. El cuaderno de demostración indica además la URL del adjunto de pesos. Se verificaron los **41.802.085 bytes** de `perufoodnet_best.pt` y su SHA-256: `c62eb62e651740587ba8b0c994778269e2d85f05ae55ee06312ffdf6800cf129`.

## Reproducción y resultados

La prueba local del paquete final ejecutó los dos cuadernos y regeneró **13 reportes y 5 recursos del artículo**. Se cotejaron sus salidas y los archivos producidos. El cuaderno exploratorio registró 4.000 imágenes de 40 platos y cero errores de lectura. Los subconjuntos de entrenamiento de 400 y 1.600 imágenes son anidados. En la demostración, el peso verificado clasificó la foto incluida como *aguadito*, con puntuación 0,997.

La selección por validación favoreció a YOLO26m-cls con ajuste completo y 40 fotos de entrenamiento por clase. Su F1 macro medio fue 0,8903 ± 0,0076 en tres semillas, frente a 0,8810 ± 0,0103 de ResNet-18 con ajuste completo. El modelo fijado antes de la prueba acertó **543 de 600 fotos**: 90,5 % de exactitud y F1 macro 0,9043, con las 50 predicciones inciertas incluidas.

**Alcance.** La consulta a GitHub verificó metadatos de publicación e integridad; la prueba funcional documentada se realizó localmente. Las descargas de usuarios finales, la ejecución en Colab y el desempeño oral en aula quedan fuera de esta verificación. Las métricas corresponden a una partición secuencial que podría compartir vistas de una misma ración entre conjuntos. La puntuación de la foto ilustrativa y la ficha documental de receta no acreditan detección visual de ingredientes ni seguridad alimentaria.
