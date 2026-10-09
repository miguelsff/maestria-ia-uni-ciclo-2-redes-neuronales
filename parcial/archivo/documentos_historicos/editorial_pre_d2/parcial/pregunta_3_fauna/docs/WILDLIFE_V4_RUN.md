# Pregunta 3: confirmación global y tercera ave

La tercera caja propuesta aquí resultó incorrecta. La referencia humana posterior se aplica en [WILDLIFE_V5_RUN.md](WILDLIFE_V5_RUN.md); v4 se conserva como historial.

Las 89 fotografías sin cambios conservan sus aprobaciones de Miguel. Las notas visuales por foto correcta son **opcionales**, según la confirmación directa del revisor: «las otras 89 no necesitan notas visuales pq las ví bien». No se rellenó ninguna observación individual. La constancia real está en `E:/Kaggle/exam-output-wildlife-v4/receipts/human-label-confirmation.json`.

V4 conserva v3 y propone añadir una caja para la tercera ave parcialmente oculta entre las rocas, señalada por el revisor en `596bdb10-23d2-11e8-a6a3-ec086b02610b`. Las dos cajas anteriores se conservan. Esa nueva caja es una propuesta del agente: la fotografía modificada queda `pending`, sin evaluador ni aprobación inventados. El comentario humano original permanece en `wildlife/parent_label_review.csv`.

## Continuar

Abrir la [galería v4](http://127.0.0.1:8897/label_review.html) e inspeccionar únicamente la nueva propuesta en la foto pendiente. Si está correcta, registrar el evaluador y aprobarla. Guardar el CSV exportado como `E:/Kaggle/exam-output-wildlife-v4/wildlife/label_review.csv`. Las otras 89 fotografías no requieren notas adicionales.

Desde `parcial/`:

```powershell
.\scripts\run_wildlife.ps1 -PrepareOnly
```

El comando selecciona v4 por defecto. `.env.wildlife-v4.example` identifica los originales, fondos de v2 en solo lectura y salida padre v3. Para una reproducción nueva, copiar el ejemplo, definir un token privado y elegir una salida vacía y puertos disponibles. La preparación importa los metadatos y pesos, conserva las decisiones sobre objetivos idénticos, audita las particiones y congela las fuentes. Los snapshots de v2/v3 no se reescriben.

Cuando la foto modificada y el requisito de arquitectura estén resueltos:

```powershell
.\scripts\run_wildlife.ps1
```

`-Resume` recupera un entrenamiento compatible. El requisito de arquitectura del enunciado sigue pendiente: el detector completo procede de Ultralytics. La resolución humana debe tener evidencia real en `receipts/architecture-review.json`, como explica la [guía v3](WILDLIFE_V3_RUN.md#resolver-el-requisito-de-arquitectura). No se atribuyó autorización al docente.

## Verificación

Pasaron 44 pruebas enfocadas. Incluyen ambos modos de revisión: las notas opcionales permiten aprobar fotografías revisadas y el modo histórico estricto conserva su comportamiento. El control científico respeta la configuración, requiere evaluador y sigue rechazando una foto pendiente o con `needs_changes`. La aprobación de las 89 fotografías no aprueba automáticamente una modificación posterior de la foto restante.

La auditoría real y los resultados actuales están en [WILDLIFE_STATUS.json](WILDLIFE_STATUS.json) y `receipts/` de v4. Únicamente cambia una caja de train respecto de v3; los holdouts y fotogramas de monitoreo se conservan. No comenzó entrenamiento científico y no existe PDF científico. El protocolo de entrenamiento, los intervalos por secuencia, la revisión de oclusión y los requisitos de entrega conservan lo documentado en v3.

Límite de reversión: conservar la salida v4 y volver al snapshot histórico de v3 para reproducir aquel protocolo. El cambio de política está en `configs/wildlife_v4.toml` y el control respeta ese parámetro; no modifica datos o artefactos científicos de las preguntas 1/2.
