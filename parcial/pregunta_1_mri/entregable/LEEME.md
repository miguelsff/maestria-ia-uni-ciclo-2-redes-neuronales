# Entrega de MRI

[Maestro ODT](informe_mri.odt), [PDF exportado desde el ODT](informe_mri.pdf) y [notebook ejecutado](01_mri.ipynb). Informe principal: 9 páginas según los recibos de B. Los hashes se conservan respecto de la base integrada.

Abrir los notebooks de entrega desde esta carpeta. Conservan las salidas calculadas; no es necesario entrenar ni ejecutar inferencia para consultarlos. Los originales científicos y archivos grandes permanecen en E:/Kaggle.

La constancia local está en [verificacion.json](verificacion.json). La matriz por subrequisito está en [../evidencias/matriz_revision.md](../evidencias/matriz_revision.md).

Para comprobar el cierre desde la raíz del repositorio:

```powershell
python parcial/compartido/scripts/run_exam.py closure
```

El comando usa copias privadas y montajes originales de solo lectura; no repite experimentos. La reproducción desde entrenamiento sigue sin verificarse. Consulta [el procedimiento compartido](../../compartido/REPRODUCIBILIDAD_CIERRE.md). Los borradores anteriores siguen en el archivo y en evidencias con su contexto. Este paquete no acredita el envío a la plataforma del curso.
