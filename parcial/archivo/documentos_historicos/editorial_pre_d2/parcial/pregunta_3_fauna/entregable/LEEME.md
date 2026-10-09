# Entrega de fauna: consulta reproducible

`03_wildlife.ipynb` integra V6 y oclusión: 16 celdas de código ejecutadas sin errores. `04_occlusion_582.ipynb` conserva el complemento original ejecutado de cuatro celdas. `metricas_occlusion.csv` contiene los tres niveles, soportes, métricas e intervalos sin redondeo.

Abrir los notebooks desde esta carpeta, junto con `data/` e `integridad_datos.json`. El primero verifica 33 archivos por SHA-256 antes de mostrar tablas. Requiere Python, pandas e IPython; el ejecutor requiere además nbformat, nbclient e ipykernel. No carga modelos ni prepara datos, entrena o ejecuta inferencia.

Desde la carpeta superior `pregunta_3_fauna`, el siguiente comando vuelve a ejecutar el notebook en el entorno Docker ya disponible:

```powershell
docker run --rm --network none --cap-drop ALL --security-opt no-new-privileges --mount "type=bind,source=$PWD,target=/work" --mount "type=bind,source=$PWD/entregable/data,target=/work/entregable/data,readonly" --workdir /work/entregable --entrypoint python neural-exam-wildlife:0.2 /work/notebooks/execute_saved_delivery.py
```

La ejecución actualiza el notebook y su constancia; su hash puede cambiar por metadatos de ejecución. Los hashes de los resultados de entrada permanecen fijos. `notebooks/prepare_saved_delivery.py` reconstruye el paquete desde los originales locales: es una herramienta de preparación, no un requisito para consultar lo entregado.

La revisión aceptó 340 casos bajos, 69 medios, 48 altos y 125 ambiguos; excluyó 18 sin nivel. El caso 121 conserva medium según Miguel, sin nota inventada. Las 398 notas faltantes permanecen vacías. Los tres niveles tienen 200 réplicas bootstrap por secuencia.

La AP usa todos los fotogramas de trans_test y regiones crowd ignoradas para cajas ajenas al nivel. No es directamente comparable con la AP global V6, no mide un efecto causal aislado ni acredita representatividad de la muestra. Los intervalos son condicionales al modelo y las secuencias observadas.

**Documentos:** la etapa A deja las fuentes y el PDF provisional de revisión en `../fuentes_informe/` y `../evidencias/`. La etapa B2 debe incorporar `informe_fauna.odt/.pdf` y `anexo_occlusion.odt/.pdf`, con los PDF exportados desde sus ODT. Esta carpeta aún no constituye el cierre documental final ni una constancia de envío al curso.
