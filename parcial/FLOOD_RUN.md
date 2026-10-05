# Pregunta 2: segmentación de agua con Sentinel-1

Esta ejecución usa U-Net propia, VV/VH, tres pérdidas y dos fuentes de supervisión. Bolivia queda reservada para evaluación externa. La configuración científica se congela en `configs/flood.toml`; los datos originales permanecen en solo lectura.

## Resultado verificado: 3 de octubre de 2026

Las seis configuraciones terminaron. La auditoría de resultados pasó para las 618 predicciones GeoTIFF y sus checkpoints: geometría, nodata, rangos, conteos y métricas coinciden con lo persistido. El notebook tiene ocho celdas de código ejecutadas sin errores, con todos los flags de entrenamiento desactivados. Se conserva una copia ejecutada en el proyecto y otra en la salida externa. El PDF borrador tiene nueve páginas; todas se renderizaron e inspeccionaron después de corregir el layout.

| Experimento | IoU ID | IoU Bolivia | Brecha ID − Bolivia | Épocas / mejor |
|---|---:|---:|---:|---:|
| `gold_bce` | 0,3939 | 0,4555 | −0,0616 | 50 / 47 |
| `gold_dice` | 0,4100 | 0,4463 | −0,0364 | 41 / 31 |
| `gold_combined` | 0,4068 | 0,4057 | +0,0010 | 50 / 49 |
| `gold_weak_bce` | 0,3251 | 0,2690 | +0,0560 | 38 / 28 |
| `gold_weak_dice` | 0,3487 | 0,4790 | −0,1302 | 11 / 1 |
| `gold_weak_combined` | 0,3593 | 0,4511 | −0,0918 | 22 / 12 |

Se usa la IoU principal sobre chips con unión no vacía. Las columnas con la convención oficial y las métricas globales están en el notebook y las tablas del informe. Añadir débiles mejora la IoU principal en cero de tres pérdidas en ID y dos de tres en Bolivia. No se elige un modelo por test ni se interpreta esa diferencia como efecto causal de la supervisión: también cambia el número de actualizaciones.

La discusión incluye falsos positivos de baja retrodispersión, omisión de agua de respuesta alta y fragmentación de cauces estrechos; las hipótesis de apariencia/prevalencia y adquisición/sesgo débil requieren evidencia adicional. Los casos son `Pakistan_70625`, `Pakistan_849790` y `Bolivia_379434` del baseline fijado `gold_bce`. Las estadísticas están en `audit/qualitative-statistics.json`.

La evidencia científica original está en `audit/flood-results.json`, `audit/delivery-verification.json` y `audit/final-source-sha256.json`. El estado guardado es `FLOOD_STATUS.json`. La autoría ya incorpora los cinco nombres y la afiliación proporcionados por el usuario, sin inventar correos. La versión editorial se compila separadamente en `E:/Kaggle/exam-delivery-flood/review-2026-10-04/reports/flood_draft.pdf`; quedan pendientes la aprobación editorial real del grupo y la entrega por la plataforma.

## Ejecutar con un comando

Desde `parcial/`, con Docker Desktop, WSL2 y GPU NVIDIA disponibles:

```powershell
.\scripts\run_flood.ps1
```

Requiere los datos SAR, máscaras y CSV oficiales bajo `E:/Kaggle/Sen1Floods11/v1.1`, junto con `download_manifest.json` y `download_receipt.json` en la raíz `E:/Kaggle/Sen1Floods11`. El descargador genera el recibo; la publicación de revisión incluye el inventario fijado de los 9.216 archivos en `parcial/delivery/flood/evidence/audit/download_manifest.json`, que debe copiarse a la raíz de los datos antes de usar la caché si no existe. Los resultados se guardan en `E:/Kaggle/exam-output-flood`. Para cambiar estas rutas antes de ejecutar, usar `SEN1_HOST_DIR` y `FLOOD_OUTPUT_DIR`. Una configuración o dataset diferente requiere una salida nueva.

El comando construye las imágenes, copia una caché Linux verificada con MD5, comprueba CUDA y pruebas relevantes, ejecuta el smoke de seis variantes y después los seis experimentos completos. Genera tablas, figuras, notebook ejecutado y PDF de revisión. Descarga únicamente el estilo NeurIPS oficial si falta. El entrenamiento funciona sin red y no necesita pesos preentrenados.

Para retomar una ejecución interrumpida:

```powershell
.\scripts\run_flood.ps1 -Resume
```

Cada época terminada conserva modelo, optimizador, escalador AMP, aleatoriedad y orden del loader. La firma de datos y configuración evita mezclar experimentos. Si existe el contenedor `neural-exam-flood-training`, inspeccionarlo antes de iniciar otra ejecución. Una falla conserva sus logs y checkpoints.

El script espera si el runner del ejercicio 3 está activo o WSL tiene menos de 4 GiB disponibles. Si ese runner aparece durante inundaciones, detiene únicamente su propio contenedor, conserva la última época completa, espera y reanuda. No detiene servicios ajenos. Un fallo de memoria sin ese contexto exige revisar el log antes de reintentar; no repite indefinidamente una configuración que falla.

## Protocolo y evidencia

- Auditoría de 9.216 archivos: tamaños y MD5 publicados coinciden; 7.709.124.682 bytes verificados.
- Particiones originales manuales: 252/89/90/15. Tras excluir seis chips totalmente inválidos: 251/86/88/15. Hay 4.150 débiles válidos, sin Bolivia ni solapamientos manuales.
- Los 16 chips totalmente inválidos y hashes SHA-256 se registran en `flood/manifest.json`; la cobertura por chip queda en `flood/coverage.csv`.
- Normalización: únicamente entrenamiento manual válido, clipping fijo de −50 a 10 dB. Media VV/VH −10,4551/−17,3253; desviación 4,0661/4,7918 dB.
- Semilla 42, batch 4, hasta 50 épocas, paciencia 10, AdamW y umbral 0,5. La confianza débil es 0,3 por chip. Los términos de pérdidas se reducen en FP32 con AMP.
- La IoU principal promedia chips con unión no vacía. Se conserva otra columna con epsilon 10⁻⁷ y unión vacía igual a uno, siguiendo el tratamiento del notebook oficial. La IoU global por píxeles se informa por separado.
- La condición manual usa 63 pasos por época y la mixta 1.101; no se atribuyen diferencias únicamente a calidad de etiquetas.

El inventario remoto consultado contiene 4.384 pares débiles, aunque la documentación menciona 4.385. De los publicados, 224 son de Bolivia y se excluyen. Los 4.160 restantes están completos localmente; diez carecen de píxeles válidos. No hace falta descargar débiles de Bolivia para este protocolo.

Las comprobaciones de CUDA/AMP, las pruebas y el smoke son evidencia funcional. Las métricas de `smoke/` no son resultados científicos. El baseline de inspección cualitativa se fija en `gold_bce` antes de observar los tests.

## Salidas

| Ruta dentro de la salida | Contenido |
|---|---|
| `audit/` | Integridad, caché, entorno, pruebas, logs y muestras de recursos. |
| `flood/<variante>/` | `best.pt`, `last.pt`, historial, configuración y metadatos de ejecución. |
| `flood/<variante>/test/` y `bolivia/` | Métricas, conteos por chip, GeoTIFF de probabilidad y ejemplos FP/FN. |
| `flood/experiments.csv` | Comparación principal de las seis variantes. |
| `reports/flood/` | Tablas, curvas, comparación ID/Bolivia y recursos. |
| `notebooks/02_flood.ipynb` | Notebook ejecutado leyendo resultados persistidos. |
| `reports/flood_draft.pdf` | Borrador científico original preservado; la revisión con autoría completa se guarda en la salida editorial indicada arriba. |

Los límites del runner son ocho CPU, 12 GiB de RAM y 2 GiB de memoria compartida. Son límites configurados. `execution.json` registra memoria CUDA asignada/reservada máxima; `audit/training-resources.jsonl` contiene muestras del contenedor y driver, que no garantizan capturar el máximo instantáneo.

En esta ejecución se tomaron 235 muestras: el mayor valor observado de RAM del contenedor fue 7,129 GiB y el del driver GPU 5.944 MiB (incluye otros procesos). El máximo del asignador CUDA por corrida fue aproximadamente 3,147 GiB. La suma de tiempos de entrenamiento fue 62,68 minutos; no incluye preparación, evaluación ni compilación. Los valores exactos quedan en `audit/delivery-verification.json` y `reports/flood/resources.csv`.

## Revisión y reversión

El notebook mantiene desactivado el entrenamiento automático. Para revisión científica deben comprobarse al menos tres patrones de error con imágenes reales y dos hipótesis de generalización; un valor bajo de VV no demuestra sombra de radar o cobertura terrestre. La geometría y las máscaras válidas se conservan al exportar.

La plantilla contiene los cinco nombres reales y la afiliación UNI, Unidad de Posgrado, Facultad de Ingeniería Industrial y de Sistemas. El enunciado no exige correos; se omiten al no haber sido proporcionados. El PDF actualizado conserva el estado de revisión hasta la aprobación del grupo. La versión publicada debe identificarse mediante `codex/flood-delivery` y su commit verificable; el enlace y la constancia de publicación se registran en el cierre de entrega.

| Unidad de trabajo | Verificación | Límite de reversión |
|---|---|---|
| Lectura y pérdidas | Nodata, alineación, peso débil, AMP y métricas sintéticas. | Restaurar solo archivos de inundaciones; conservar originales. |
| Ejecución aislada | Compose propio, caché con MD5, firmas y checkpoints. | Preservar salida y volumen; no reutilizarlos con una configuración distinta. |
| Entregables | Notebook ejecutado, cifras persistidas, autoría completa y PDF renderizado. | Mantener explícita la aprobación editorial pendiente; conservar resultados científicos y borrador anterior. |

El usuario autorizó ejecutar el plan de cierre y publicar el material reproducible. La publicación se prepara en una rama aislada; los datos originales y las corridas permanecen preservados. No ejecutar `docker compose down --volumes`: eliminaría la caché y obliga a verificarla de nuevo.
