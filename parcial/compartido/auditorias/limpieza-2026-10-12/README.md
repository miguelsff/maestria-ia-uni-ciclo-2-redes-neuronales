# Limpieza del parcial: entrega final y reproducción

Limpieza autorizada y ejecutada el 12 de octubre de 2026. Se conservaron íntegros los entregables de las tres preguntas, los resultados científicos V6 y la muestra de oclusión aceptada. El [inventario](inventory.json) registra archivos, tamaños, hashes, rutas retiradas y la base Git de la operación.

Se retiraron 3.401 archivos históricos y entornos obsoletos: 2,249 GiB de archivos, conservando 0,652 GiB de insumos necesarios en su nueva ubicación. La reducción neta de datos es aproximadamente 1,60 GiB, antes de contar los recibos nuevos y la reducción adicional de los volúmenes Docker.

## Resultado

- Retirados los respaldos y entregas antiguas de `parcial/archivo/`, conservando `referencias_originales/`.
- Retirados el notebook histórico, los borradores de Fauna, las guías v1–v5 y las configuraciones y entornos privados anteriores que ya no usa el flujo V6.
- Retiradas las salidas `exam-output-wildlife` y `exam-output-wildlife-v2` a `v5` de E:/Kaggle, después de conservar y comprobar sus insumos necesarios.
- Retirados diez volúmenes Docker sin contenedores asociados y el contenedor de recuperación V6 terminado, cuyo estado coincide con el recibo original preservado.
- Actualizadas las rutas, las guías y el launcher para reutilizar los insumos conservados. Los secretos de V6 permanecen privados; los recibos no incluyen valores de entorno.

Los fondos completos están en `E:/Kaggle/exam-input-wildlife/backgrounds`. El padre mínimo está en `E:/Kaggle/exam-input-wildlife/parent-v5`: once archivos importados por V6 y dos recibos humanos originales. `E:/Kaggle/exam-input-wildlife/migration.json` registra cada copia y su SHA-256. Los originales CCT20/ECCV, MRI e inundaciones conservan sus ubicaciones.

Los Compose v2–v6 permanecen porque son capas que utiliza el runner V6 y forman parte de las 44 fuentes congeladas. No se modificaron sus bytes ni los de los helpers firmados. Los recibos anteriores conservan sus rutas históricas como procedencia; el inventario de esta limpieza documenta qué rutas dejaron de estar disponibles.

## Verificación

| Comprobación | Evidencia |
| --- | --- |
| 44 fuentes congeladas y 5.190 archivos protegidos sin cambios | [verification.json](verification.json) |
| Cierre en el entorno científico, también después del borrado | [final-closure-verification.json](final-closure-verification.json), [comando](final-closure-command.json) |
| 23 pruebas del cierre | [closure.log](closure.log) |
| 27 pruebas del padre, revisión y launcher | [focused-tests.log](focused-tests.log) |
| Planes reales de MRI, inundaciones y Fauna; Compose válido y sin crear salidas científicas | [scientific-plans.json](scientific-plans.json) |
| Bootstrap original desde el padre mínimo: once hashes idénticos | [parent-bootstrap.log](parent-bootstrap.log) |
| Galerías accesibles, fotografía y fondo servidos; píxeles del fondo idénticos | [runtime.json](runtime.json) |
| Diez volúmenes retirados sin usuarios | [docker-volumes.json](docker-volumes.json) |
| Estado del contenedor terminado cotejado antes de retirarlo | [removed-container.json](removed-container.json) |
| Entornos privados obsoletos retirados, sin guardar sus contenidos | [private-environments.json](private-environments.json) |
| Guías vigentes: 88 enlaces locales válidos en doce documentos | [links.json](links.json) |
| Revisión de estilo de los tres archivos Python modificados | [lint.log](lint.log) |
| Cinco copias privadas de verificación retiradas | [temporary-runtimes.json](temporary-runtimes.json) |

La firma aceptada permanece `6b2983c33abccbc18e293cc3a10ca30a710669c2ba8055a76bb462b3348a13a9`: 582 casos, 18 exclusiones y 200 réplicas por nivel evaluado. No se repitieron entrenamiento, inferencia ni bootstrap científico. La reproducción completa desde entrenamiento continúa sin acreditarse.

## Revisión y reversión

Los cambios del repositorio se pueden revisar con Git; no se crearon commits ni se reescribió el historial. El inventario identifica la base y las rutas modificadas o retiradas para restaurarlas selectivamente, sin afectar cambios posteriores. Las salidas experimentales externas retiradas no se respaldaron: se conservaron sus insumos necesarios y los originales finales, conforme al criterio de mantener solo entrega y reproducción.

Las copias privadas creadas para estas comprobaciones se retiran al terminar; los recibos permanecen aquí. Para volver a verificar, el [procedimiento compartido](../../REPRODUCIBILIDAD_CIERRE.md) crea una copia privada nueva.
