# Examen parcial: entrega organizada

Los tres informes principales están en ODT editable y PDF exportado desde ese ODT. Los notebooks finales conservan sus resultados y hashes; la migración no repitió entrenamiento, inferencia ni bootstrap.

| Pregunta | Entrega |
|---|---|
| MRI | [Informe y notebook](pregunta_1_mri/entregable/LEEME.md) |
| Inundaciones | [Informe y notebook](pregunta_2_inundaciones/entregable/LEEME.md) |
| Fauna y oclusión | [Informe, anexo y notebooks](pregunta_3_fauna/entregable/LEEME.md) |

[Procedimiento compartido](compartido/REPRODUCIBILIDAD_CIERRE.md), [82 comprobaciones](compartido/auditorias/cierre-2026-10-08/matrices_completas.md) y [recibo de migración](compartido/auditorias/estructura/d2/verificacion.json).

El motor, Dockerfile, Compose, dependencias y ejemplos públicos de entorno están en `compartido/`. Los datos y checkpoints grandes permanecen en E:/Kaggle. Las copias de runtime y sus secretos son privados e ignorados en `.runtime/`.

El [enunciado ODT](compartido/referencias/examen-parcial.odt) y su [PDF exportado](compartido/referencias/examen-parcial.pdf), junto con el [ejemplo ODT](compartido/referencias/ejemplo.odt) y [PDF exportado](compartido/referencias/ejemplo.pdf), son las referencias editables. Los [PDF originales](archivo/referencias_originales/) permanecen inalterados.

La [limpieza del 12 de octubre de 2026](compartido/auditorias/limpieza-2026-10-12/README.md) retira borradores, respaldos y salidas antiguas. Conserva los entregables completos, las referencias originales, las fuentes de reproducción y las evidencias de aceptación final. Las menciones a archivos históricos en recibos anteriores describen su estado original; el inventario de limpieza registra lo retirado sin reescribir esos recibos.

Las capas `compose.wildlife.v2.yaml` a `v6` siguen siendo dependencias de Fauna V6 y fuentes congeladas; no son salidas descartables. Los fondos compartidos y el padre mínimo V5 están en `E:/Kaggle/exam-input-wildlife/`. El [procedimiento compartido](compartido/REPRODUCIBILIDAD_CIERRE.md) es la entrada vigente para verificar y reproducir el parcial. La reproducción completa desde entrenamiento y el envío al curso siguen sin acreditarse.
