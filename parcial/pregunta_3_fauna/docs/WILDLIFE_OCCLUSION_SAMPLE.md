# Revisión de oclusión: muestra separada de 600 casos

La muestra autorizada se revisa en [la galería local](http://127.0.0.1:8902/occlusion_review.html). Contiene **600 recuadros en 596 fotos y 552 secuencias** de `trans_test`, con las nueve cámaras y las 13 especies presentes. El conjunto V6 y sus revisiones originales se conservan.

## Revisión humana

1. Ingresar el nombre del evaluador.
2. Observar el animal del recuadro amarillo; pulsar la foto para ampliarla.
3. Seleccionar baja (>75 % visible), media (25–75 %), alta (<25 %) o ambigua/truncada. Completar una evidencia breve.
4. Continuar con los siguientes casos. El guardado automático permite cerrar y retomar; **Guardar ahora** confirma los cambios pendientes y **Descargar CSV** genera una copia.

Un animal pequeño, una foto oscura o desenfocada no definen oclusión por sí solos. El borde que corta al animal corresponde a truncación y se registra como ambigua según este protocolo. No se infieren niveles a partir del detector.

La revisión queda pendiente mientras alguna fila no tenga nivel, evidencia y evaluador. Se conservan los campos de otras filas y la autoría previa. El navegador respalda los cambios todavía no guardados; el servidor conserva el CSV anterior y registra cada actualización humana. El servidor solo escucha en `127.0.0.1` y admite cambios en los tres campos humanos de los 600 casos seleccionados.

## Archivos y método

Los entregables de esta revisión están en `E:/Kaggle/exam-output-wildlife-v6-occlusion-sample-600`:

| Archivo | Uso |
| --- | --- |
| `occlusion_review_sample_600.csv` | Niveles, evidencias y evaluadores humanos, junto con los pesos de muestreo |
| `protocol.json` y `protocol.sha256` | Lista fija de casos, recuadros, probabilidades, pesos e identidad de V6 |
| `sampling_analysis.json` | Reparto aceptado de los 209 grupos |
| `images/` | Copias exactas de las 596 fotos originales, verificadas por SHA-256 |
| `review_status.json` | Avance real y soporte por nivel |
| `human_changes.jsonl` | Historial creado a partir de cambios reales del evaluador |

La [justificación del tamaño](../../compartido/auditorias/occlusion-sampling-2026-10-05/calculo_muestra_occlusion.md) establece un objetivo aproximado del 95 % y ±5 puntos porcentuales para proporciones globales de oclusión, no para mAP. Se toman al menos dos casos de cada combinación cámara × especie × tamaño cuando están disponibles; los 19 grupos con un caso se revisan completos. El resto se asigna para reducir la varianza máxima. El reparto es 133 pequeños, 317 medianos y 150 grandes.

La selección utiliza `random.Random(42).sample`, sobre IDs ordenados dentro de cada grupo. Una permutación posterior cambia únicamente el orden de presentación. El recibo fija los 600 IDs para no depender de futuros cambios del generador. No se utilizan confianza, aciertos ni fallos del modelo para seleccionar. Los porcentajes globales se calculan con los pesos `N_h/n_h`; una media simple sobre los 600 casos sobrerrepresentaría los grupos escasos. Las ráfagas repetidas se mantienen dentro del diseño probabilístico y se agrupan para los intervalos de rendimiento.

## Evaluación posterior sin entrenamiento

Cuando se complete la revisión, `scripts/evaluate_occlusion_sample.ps1` ejecuta la evaluación separada usando el entorno existente, las predicciones y el `best.pt` de V6. `-CheckOnly` comprueba el estado sin calcular métricas. El padre se monta en lectura y las salidas se escriben únicamente en esta revisión.

Los 600 niveles, evidencias y nombres deben estar completos. El protocolo actual exige al menos 30 casos en cada nivel bajo, medio y alto para publicar su mAP; si algún nivel no llega, se informa el déficit y cualquier suplemento se documenta aparte de la muestra base.

Se generan proporciones ponderadas con intervalos marginales aproximados y una cota conservadora de varianza binaria. Para detección se reutilizan las funciones COCO y bootstrap de V6: las cajas no revisadas o de otro nivel se ignoran como regiones crowd y se conserva el conjunto completo de fotogramas de `trans_test`. Este mAP describe la muestra y ese conjunto de falsos positivos; no es un estimador ponderado del mAP poblacional.

Los intervalos de detección se calculan con **200 réplicas por secuencia**, semilla 42 y un proceso CPU nuevo por nivel. Se verifica el modelo, las fuentes, los datos y la revisión antes de publicar cada resultado. Los resultados quedan vinculados a la identidad del protocolo y al hash del CSV humano en una carpeta propia; una revisión distinta no reutiliza resultados anteriores. No se entrena ni se ejecuta inferencia nueva.

## Comprobación y alcance de los cambios

Las pruebas de `tests/test_occlusion_sample.py` usan exclusivamente datos sintéticos temporales: cuotas e identidad reproducibles, pesos, bloqueos de revisión incompleta, guardado y descarga, conflictos entre pestañas, conservación al repetir la preparación y equivalencia numérica con la evaluación COCO/bootstrap congelada.

La comprobación real verifica 600 IDs únicos, 209 grupos cubiertos, imágenes decodificables y campos humanos inicialmente vacíos. También verifica los hashes del entrenamiento, predicciones, notebook, PDF, los dos CSV originales y las 44 fuentes congeladas. La nota y el recibo de validación documentan el resultado; no se generan aprobaciones humanas.

La frontera de reversión es el nuevo directorio de muestra, el servidor local 8902 y los archivos nuevos de este flujo: `scripts/wildlife_occlusion_sample.py`, `scripts/evaluate_occlusion_sample.py`, `scripts/evaluate_occlusion_sample.ps1`, `reviews/occlusion_sample.html`, `tests/test_occlusion_sample.py` y este documento. Las revisiones humanas generadas deben conservarse antes de retirar el flujo. No se requiere revertir V6 ni las preguntas 1 y 2. El seguimiento periódico previo continúa pausado.

## Paquete distribuible sin conexión

`outputs/occlusion-offline/revision_occlusion_600_offline.zip` incluye `index.html`, las 596 fotos originales, una copia del CSV humano al empaquetar, instrucciones y recibos de identidad. Se extrae completo y se abre `index.html` en Chrome, Edge o Firefox. El HTML incorpora los 600 casos y sus recuadros; sus estilos y código están incluidos. No usa servidor, peticiones de datos, fuentes externas ni instalaciones.

Cada evaluador ingresa su nombre y descarga su CSV al terminar cada sesión. **Cargar CSV** permite retomar o reunir archivos completos de la misma muestra; conserva el orden y los campos inmutables. La combinación compara cada revisión con la copia inicial: incorpora cambios no superpuestos, ignora la copia inicial o entradas idénticas y rechaza diferencias sobre casos ya modificados, incluidas revisiones parciales. Ante un conflicto no incorpora ninguna fila ni sobrescribe los datos actuales. Se envían ambos archivos al responsable para resolver la diferencia humana.

El respaldo del navegador es opcional: al abrir archivos locales puede depender del navegador y de la carpeta. El CSV descargado es el respaldo portátil y no se modifica automáticamente en disco. El HTML informa cuando no puede usar almacenamiento local; un respaldo inválido se conserva y requiere cargar un CSV válido. Los textos humanos se incorporan como datos y nunca se ejecutan como HTML.

El empaquetador `scripts/package_occlusion_offline.py` lee una única copia coherente del CSV, verifica las identidades y hashes y publica el ZIP mediante reemplazo de archivo temporal. No escribe en la revisión activa ni incluye historial privado, credenciales o pesos del modelo. `manifest_package.json` vincula el protocolo, el CSV inicial y cada archivo por SHA-256. La configuración de entrenamiento y las fuentes congeladas se conservan.

Verificación: cuatro pruebas de empaquetado con `python -m unittest discover -s tests -p test_occlusion_offline.py -v`; pruebas de CSV, recuperación, combinación y conflictos con `node tests/test_occlusion_offline.cjs`; revisión de estilo de los dos archivos Python nuevos. La validación real comprueba los 600 IDs y recuadros, 596 fotos idénticas, hashes de todos los archivos, integridad ZIP y los 12 artefactos y 44 fuentes de V6. En una copia extraída temporal se verificaron fotos, ampliación, campos, contadores, descarga real de 600 filas y recuperación del CSV en otra página. Los datos de prueba se identifican como sintéticos y no se incorporan al paquete ni a la revisión humana. La apertura directa mediante `file://` no se probó visualmente porque el navegador de automatización bloquea ese protocolo; la interfaz se probó mediante una vista estática local permitida.

La reversión de este paquete comprende únicamente `scripts/package_occlusion_offline.py`, `reviews/occlusion_offline.html`, `reviews/occlusion_offline.js`, `tests/test_occlusion_offline.py`, `tests/test_occlusion_offline.cjs`, la salida del ZIP y esta sección. Antes de retirarlo se conservan los CSV humanos recibidos. No requiere revertir la galería activa ni ninguna parte de V6.
