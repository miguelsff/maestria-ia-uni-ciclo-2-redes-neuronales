# Preparación D2 sin aplicación

D2 permanece pendiente del SHA integrado A+B+C comunicado por el chat principal. Esta preparación no incorpora otras ramas, no modifica el checkout principal y no traslada archivos. La unidad de reversión de este seguimiento es el ajuste del generador, sus mapas derivados y estos documentos; el único ensayo ejecutado es la validación del plan D1 sin mutaciones.

## Inicio desde la integración

Cuando el chat principal entregue el SHA, crear `codex/examen-estructura-d2` desde ese commit en este mismo worktree limpio. No fusionar la rama D1 ni volver a incorporar el commit semilla de auditoría: el historial integrado ya contiene las decisiones comunes. Los documentos D1 deben estar disponibles en la integración; si faltan, incorporar solo sus commits de diseño identificados, nunca una segunda semilla.

La secuencia preparada, todavía no ejecutada, es:

```powershell
# integrationSha debe contener el SHA entregado por el chat principal.
git status --short
git -c core.longpaths=true switch -c codex/examen-estructura-d2 $integrationSha
git log -1 --oneline
```

El chat principal informó `core.longpaths=true` configurado en el repositorio. Todo clone de verificación deberá usar `git -c core.longpaths=true clone ...` para las rutas humanas largas; comprobar también que las operaciones posteriores no vuelvan a deshabilitar esa opción. No hay autorización para publicar el respaldo ni archivos privados.

## Asociación completa PDF→ODT

El manifiesto final debe cubrir **cada PDF conservado**, incluyendo originales, borradores, provisionales, entregas antiguas y exportaciones finales. Comparar inventario y manifiesto como conjuntos de rutas: ningún PDF puede quedar sin asociación. La relación expresa procedencia editorial; no exige igualdad de hashes entre un PDF fuente y otro exportado desde el ODT.

Cada registro final debe incluir `pdf_path`, `pdf_sha256`, `role`, `master_id`, `master_odt_path`, `master_odt_sha256`, `relationship`, `exported_pdf_path`, `exported_pdf_sha256`, y una explicación de diferencias cuando corresponda. En el registro de preparación los hashes y rutas de maestros no disponibles son `null` con estado pendiente, nunca resultados fabricados. D2 solo cerrará la asociación cuando los maestros realmente existan y sus hashes se hayan calculado.

| Familia de PDF | Maestro ODT correspondiente | Tratamiento |
|---|---|---|
| Informe final MRI | `pregunta_1_mri/entregable/informe_mri.odt` | PDF final exportado de este maestro; fuentes históricas asociadas como versiones anteriores si existen. |
| Informe final de inundaciones | `pregunta_2_inundaciones/entregable/informe_inundaciones.odt` | Mantener distinción frente a la entrega histórica. |
| Informe final/provisional Fauna | `pregunta_3_fauna/entregable/informe_fauna.odt` | El provisional de A es antecedente, no PDF final exportado. |
| Anexo final/provisional de oclusión | `pregunta_3_fauna/entregable/anexo_occlusion.odt` | Vincular el provisional de A y la exportación final con sus hashes separados. |
| Examen original y exportación de referencia | `compartido/referencias/examen-parcial.odt` | Original PDF a `archivo/referencias_originales/`; referencia PDF a `compartido/referencias/`, exportada del ODT. |
| Ejemplo original y exportación de referencia | `compartido/referencias/ejemplo.odt` | Mismo contrato: original intacto archivado y exportación de referencia separada. |
| `delivery/flood/flood_review.pdf` | Maestro histórico nativo de inundaciones de B | Archivar entrega antigua y resolver ruta exacta del maestro tras integración. |
| `tmp/pdfs/wildlife_v6_review/initial_draft.pdf` | Maestro histórico nativo de Fauna de B | Archivar borrador; identificar si es antecedente editorial del maestro histórico y explicar diferencias. No promoverlo como entrega final. |

El chat principal comunicó ocho ODT nativos de B y 67 páginas en total: MRI 9, inundaciones 9, Fauna 10, anexo 2, examen 7, ejemplo 11, histórico Fauna 10 e histórico inundaciones 9. Estos son resultados comunicados por el responsable documental, **pendientes de comprobación en la base integrada de D2**. La correspondencia de un borrador histórico se comprobará por su contenido/procedencia; el parecido del nombre no basta para declarar que sea exportación del maestro.

## Alcance integrado y respaldo

La inspección de `git ls-tree -r --name-only 00a334d parcial` confirmó rutas de las salidas humanas en `archivo/documentos_historicos/salidas_locales_2026-10-08/` y los PDF `pregunta_3_fauna/evidencias/{informe_fauna_provisional,anexo_occlusion_provisional}.pdf`. No se copiaron esos archivos a este worktree.

Según el chat principal, la integración incluirá 13 salidas humanas pequeñas con procedencia y atributos, y la matriz de 82 comprobaciones. Deben permanecer registradas en su contexto, conservando revisiones/CSV recibidos originales y decisiones aceptadas; un borrador consolidado no sustituye una revisión aceptada.

Respaldos informados por el principal, sin copiarlos ni publicarlos desde D1:

- `E:/Kaggle/exam-closure-2026-10-08/checkout-before-final`: respaldo completo, 1010 archivos y 199773980 bytes comunicados; verificar manifiesto cuando se use.
- `E:/Kaggle/exam-closure-2026-10-08/ignored-before`: ZIP de aproximadamente 68 MB, `train.csv`, `train_series.csv` y seis entornos privados. Permanecen externos; no promoverlos al repositorio.
- `E:/Kaggle/exam-closure-2026-10-08/preserved-before`: las 64 versiones ya verificadas por D1. Integrar solo productos necesarios y archivar diferencias de versiones sin sobrescribirlas.

## Cierre preparado para D2

1. Inventariar nuevamente la integración y todos los archivos locales preservados que corresponda conservar; resolver colisiones antes de operaciones.
2. Completar las asociaciones PDF→ODT y el manifiesto histórico. Cada borrador conserva su rol y cada exportación final apunta al maestro real.
3. Conservar los bytes de las 44 fuentes congeladas y usar motor/Docker/Compose/dependencias en `compartido/`. Incorporar overlays/adaptadores de C sin sustituir estas fuentes.
4. Actualizar únicamente rutas editoriales, README/LEEME y referencias fuera del conjunto congelado; validar enlaces y manifests. No reejecutar notebooks científicos ni entrenamientos/inferencia.
5. Validar el plan D2 regenerado, ejecutar la migración en este worktree y registrar hashes y verificaciones. La autorización y los SHA A+B+C deben constar antes de aplicar.
