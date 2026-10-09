# Entrega de Fauna

[Maestro ODT](informe_fauna.odt), [PDF exportado desde el ODT](informe_fauna.pdf) y [notebook ejecutado](03_wildlife.ipynb). Informe principal: 10 páginas según los recibos de B. Los hashes se conservan respecto de la base integrada.

Además: [anexo ODT](anexo_occlusion.odt), [anexo PDF](anexo_occlusion.pdf), [notebook de oclusión](04_occlusion_582.ipynb) y [métricas](metricas_occlusion.csv). El anexo tiene 2 páginas según el recibo de B.

Abrir los notebooks de entrega desde esta carpeta. Conservan las salidas calculadas; no es necesario entrenar ni ejecutar inferencia para consultarlos. Los originales científicos y archivos grandes permanecen en E:/Kaggle.

La constancia local está en [verificacion.json](verificacion.json). La matriz por subrequisito está en [../evidencias/matriz_revision.md](../evidencias/matriz_revision.md).

Para comprobar el cierre desde la raíz del repositorio:

```powershell
python parcial/compartido/scripts/run_exam.py closure
```

El comando usa copias privadas y montajes originales de solo lectura; no repite experimentos. La reproducción desde entrenamiento sigue sin verificarse. Consulta [el procedimiento compartido](../../compartido/REPRODUCIBILIDAD_CIERRE.md). Los borradores anteriores siguen en el archivo y en evidencias con su contexto. Este paquete no acredita el envío a la plataforma del curso.
