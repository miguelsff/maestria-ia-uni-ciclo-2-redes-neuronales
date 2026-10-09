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

El [archivo](archivo/manifiesto.json) conserva versiones históricas, fuentes, las 64 versiones respaldadas y las 13 salidas humanas previas con su procedencia. Cada PDF conservado se asocia a un maestro ODT por hash de fuente o exportación; los borradores no se promueven a entregas. La reproducción completa desde entrenamiento y el envío al curso siguen sin acreditarse.
