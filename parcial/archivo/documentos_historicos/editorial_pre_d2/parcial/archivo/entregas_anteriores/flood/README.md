# Paquete de revisión de la pregunta 2

El [informe](flood_review.pdf) tiene nueve páginas, siete referencias y los cinco integrantes proporcionados por el usuario. Se verificaron visualmente todas las páginas. La aprobación editorial del grupo y la entrega en la plataforma siguen pendientes; `reports/flood/finalized.json` conserva `reviewed=false`.

El [notebook ejecutado](../../notebooks/02_flood.ipynb), las métricas y los recibos históricos de [evidence](evidence/) corresponden a las seis corridas originales. No se repitió el entrenamiento completo para publicar este paquete. Los datos originales, checkpoints y GeoTIFF se conservan externamente.

La revisión del código preparado para publicación pasó 17 pruebas en CPU; tres casos de AMP requieren CUDA y se omitieron. Los recibos históricos conservan la validación anterior con GPU. El control del entorno e importaciones pasó, utilizando una salida temporal para evitar escribir sobre los originales. [review-verification.json](evidence/audit/review-verification.json) distingue ambas verificaciones.

El inventario fijado [download_manifest.json](evidence/audit/download_manifest.json) es necesario para preparar la caché. El [README principal](../../../README.md) explica su instalación y el comando completo de reproducción. [source-sha256.json](source-sha256.json) registra los bytes de este paquete, excluyendo el propio manifiesto para evitar un hash circular.

El [guion de exposición](../../PRESENTACION_FLOOD.md) está preparado para la presentación del 30 de octubre de 2026; las asignaciones son propuestas para el grupo.
