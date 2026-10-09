# Revisión de la pregunta 2

Cada fila corresponde a una comprobación independiente. Los originales científicos permanecen inalterados.

Las rutas de entrega se interpretan desde `parcial/`; los recibos científicos y `validation-final.json`, desde `compartido/auditorias/cierre-2026-10-08/`. `structure.json` corresponde a `compartido/auditorias/revision-2026-10-08/structure.json`.

| Pregunta | Entregable | Subrequisito | Evidencia concreta | Estado | Limitación | Pendiente |
|---|---|---|---|---|---|---|
| 2 | 1 | U-Net personalizada implementada en PyTorch | compartido/src/neural_exam/flood/model.py; flood-scientific.json; pregunta_2_inundaciones/entregable/informe_inundaciones.pdf | cumple | Ninguna adicional. | Ninguno técnico. |
| 2 | 1 | Entrada SAR con las dos bandas VV/VH | flood-scientific.json; pregunta_2_inundaciones/entregable/informe_inundaciones.pdf; dataset, configuración y carga GeoTIFF de dos canales | cumple | Ninguna adicional. | Ninguno técnico. |
| 2 | 2 | Etiquetas doradas frente a doradas más débiles | flood-scientific.json; pregunta_2_inundaciones/entregable/informe_inundaciones.pdf; seis corridas gold_* y gold_weak_*; entrenamiento gold=251, weak=4150 | cumple | Se respetan las particiones oficiales; no se usan todos los dorados para entrenar. | Ninguno técnico. |
| 2 | 2 | Comparación de pérdidas o arquitectura | flood-scientific.json; pregunta_2_inundaciones/entregable/informe_inundaciones.pdf; BCE, Dice y combinada cruzadas con las dos fuentes de supervisión | cumple | Ninguna adicional. | Ninguno técnico. |
| 2 | 2 | Configuración y finalización de las seis corridas | flood-scientific.json; pregunta_2_inundaciones/entregable/informe_inundaciones.pdf; execution.json, history.csv, complete.json, best.pt y predicciones guardadas | cumple | Una semilla; los presupuestos efectivos difieren por la cantidad de etiquetas débiles. | Ninguno técnico. |
| 2 | 3 | IoU, F1, precisión y recall en distribución | flood-scientific.json; pregunta_2_inundaciones/entregable/informe_inundaciones.pdf; 88 chips test; baseline mean-chip IoU=0.393926867, F1 global=0.798229203 | cumple | Ninguna adicional. | Ninguno técnico. |
| 2 | 3 | Definición y agregación de IoU | flood-scientific.json; pregunta_2_inundaciones/entregable/informe_inundaciones.pdf; IoU por chip con unión no vacía; global=0.664210850 para baseline | cumple | Media por chip, IoU global y convención oficial con epsilon en chips vacíos son cantidades distintas. | Ninguno técnico. |
| 2 | 3 | Tratamiento de píxeles sin datos | flood-scientific.json; pregunta_2_inundaciones/entregable/informe_inundaciones.pdf; 618 GeoTIFF, nodata=-1, máscaras válidas y georreferenciación comprobados | cumple | Ninguna adicional. | Ninguno técnico. |
| 2 | 3 | Matriz de confusión y comentario | flood-scientific.json; pregunta_2_inundaciones/entregable/informe_inundaciones.pdf; baseline TP1884956 FP322880 FN630052 TN17606071; figura y discusión | cumple | Ninguna adicional. | Ninguno técnico. |
| 2 | 4 | Bolivia excluida del entrenamiento y selección | flood-scientific.json; pregunta_2_inundaciones/entregable/informe_inundaciones.pdf; cuatro particiones oficiales; Bolivia=15, train251/val86/test88 sin intersecciones | cumple | Ninguna adicional. | Ninguno técnico. |
| 2 | 4 | Evaluación externa y brecha frente a ID | flood-scientific.json; pregunta_2_inundaciones/entregable/informe_inundaciones.pdf; seis variantes test/Bolivia; baseline mean-chip Bolivia=0.455477085 | cumple | La brecha es firmada: algunos modelos mejoran en Bolivia. No se fuerza una caída inexistente. | Ninguno técnico. |
| 2 | 4 | Al menos dos hipótesis explicativas | pregunta_2_inundaciones/entregable/informe_inundaciones.pdf: distribución de agua/cobertura y condiciones SAR | cumple | Hipótesis plausibles; no causas demostradas experimentalmente. | Ninguno técnico. |
| 2 | 5 | Tres tipos de fallo con ejemplos visuales concretos | flood-qualitative.json; Pakistan_70625, Pakistan_849790, Bolivia_379434; figuras del informe | cumple | Ninguna adicional. | Ninguno técnico. |
| 2 | 5 | Explicaciones sustentadas por observaciones | flood-qualitative.json: backscatter, FP/FN y errores de borde recalculados; 81.25% cerca del borde en tercer caso | cumple | Sombra y doble rebote son interpretaciones posibles; sin información adicional no se certifica mecanismo físico. | Ninguno técnico. |
| 2 | 6 | Bono de fusión Sentinel-2 frente a SAR | No hay corrida SAR+óptico en los seis experimentos preservados. | falta | Bono opcional; no bloquea los cinco entregables obligatorios. | Ninguno obligatorio; no se añade entrenamiento nuevo. |
| 2 | común | Informe NeurIPS de 6–10 páginas | validation-final.json; entregable de la pregunta correspondiente | cumple | Ninguna adicional. | Ninguno técnico. |
| 2 | común | Cinco autores y al menos cinco referencias | validation-final.json; entregable de la pregunta correspondiente | cumple | Ninguna adicional. | Ninguno técnico. |
| 2 | común | Motivación, antecedentes, metodología, experimentos, resultados y discusión | validation-final.json; entregable de la pregunta correspondiente | cumple | Ninguna adicional. | Ninguno técnico. |
| 2 | común | ODT editable y PDF exportado desde él | validation-final.json; entregable de la pregunta correspondiente | cumple | Ninguna adicional. | Ninguno técnico. |
| 2 | común | Notebook final ejecutado sin errores | validation-final.json; entregable de la pregunta correspondiente | cumple | Ninguna adicional. | Ninguno técnico. |
