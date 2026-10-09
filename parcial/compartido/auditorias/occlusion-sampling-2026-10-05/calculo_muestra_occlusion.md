# Muestra propuesta para revisar la oclusión: 600 casos

Análisis del CSV y manifiesto reales de fauna V6. Es una propuesta de tamaño y reparto: todavía no se seleccionaron casos ni se registraron valoraciones humanas. Los casos son apariciones de animales anotadas en recuadros, no individuos únicos.

La población de esta revisión es `trans_test`: 17,620 recuadros en 16,869 fotos y 6,723 secuencias, procedentes de nueve cámaras y 13 especies. `deer` está ausente en esta partición. La revisión de etiquetas de 90 imágenes es otro proceso.

## Cálculo y decisión

Se adopta como objetivo una confianza aproximada del 95 % y un margen marginal de ±5 puntos porcentuales para estimar la proporción global de cada nivel de oclusión, con el peor caso de variabilidad binaria (`p=0.5`). Esta precisión es una decisión metodológica propuesta.

Para muestreo aleatorio simple sin reemplazo:

`n = N*z²*p*(1-p) / [e²*(N-1) + z²*p*(1-p)]`

Con `N=17620`, `z≈1.96`, `p=0.5` y `e=0.05`, el resultado es `375.970427`, que se redondea hacia arriba a **376**. Es el mínimo de referencia para porcentajes generales bajo ese diseño.

Los datos contienen **209 combinaciones presentes de cámara × especie × tamaño**. Para cubrirlas, el diseño propuesto asigna al menos dos casos a cada combinación; las 19 combinaciones con un solo caso se revisan completas. Esto consume **399 casos** antes de distribuir los restantes según la reducción de la varianza máxima.

Con selección aleatoria independiente dentro de cada combinación y porcentajes ponderados según su población, **600 casos** alcanzan un margen previsto de **±4.9958 puntos**; con 599 es ±5.0040. Por ello se recomiendan 600 para este diseño particular, equivalentes al 3.41 % de los recuadros. El mínimo de dos por combinación da cobertura y no garantiza precisión dentro de cada combinación.

Se propone semilla 42. La selección utilizaría únicamente las anotaciones originales y metadatos de cámara, especie y tamaño. Las predicciones del detector no intervienen en la selección. Los porcentajes globales requieren pesos `N_h/n_h`, porque las combinaciones escasas quedan sobrerrepresentadas. El reparto de las 209 combinaciones y sus pesos están en [analysis.json](analysis.json).

## Alcance y condición de finalización

La precisión calculada corresponde a proporciones globales de oclusión en esta población fija; no asegura un margen de ±5 puntos en mAP, por especie o por cámara. El mAP del subconjunto sería un análisis de muestra y tendría sus propios intervalos agrupados por secuencia. El evaluador COCO existente no incorpora estos pesos de encuesta.

Todavía no sabemos cuántos casos tienen oclusión alta: los 17,620 niveles humanos están vacíos. El protocolo local pide al menos 30 casos en cada nivel bajo, medio y alto, pero ese umbral no demuestra representatividad. Si un nivel queda por debajo de 30, cualquier suplemento se documentaría separado de esta muestra probabilística fija. No se mezclan sin ponderación los casos añadidos de forma dirigida con los porcentajes de la muestra base.

Las fotos de una misma ráfaga pueden estar correlacionadas; los casos no son individuos independientes. El muestreo aleatorio propuesto estima la población fija de recuadros y conserva probabilidades conocidas. Evitar repeticiones de secuencia mediante reemplazos posteriores alteraría esas probabilidades. Para incertidumbre de rendimiento se requiere considerar la secuencia como grupo.

Los casos ambiguos se conservan como resultado y no se sustituyen en silencio. Se registran nivel, evidencia visual y autoría humanos. La configuración V6 mantiene su revisión completa original: una ejecución por muestra necesitaría un protocolo separado. No se modifica V6 ni se vuelve a entrenar.

## Distribución propuesta por cámara

| Grupo | Población | Muestra propuesta |
| --- | ---: | ---: |
| 0 | 734 | 43 |
| 100 | 4,537 | 114 |
| 105 | 1,938 | 61 |
| 130 | 1,172 | 58 |
| 28 | 967 | 51 |
| 40 | 848 | 49 |
| 46 | 4,319 | 117 |
| 7 | 1,410 | 49 |
| 78 | 1,695 | 58 |

## Distribución propuesta por especie

| Grupo | Población | Muestra propuesta |
| --- | ---: | ---: |
| badger | 15 | 9 |
| bird | 179 | 36 |
| bobcat | 1,974 | 64 |
| cat | 1,366 | 49 |
| coyote | 1,901 | 58 |
| dog | 958 | 43 |
| fox | 1 | 1 |
| opossum | 4,659 | 111 |
| rabbit | 722 | 35 |
| raccoon | 4,769 | 117 |
| rodent | 26 | 11 |
| skunk | 273 | 25 |
| squirrel | 777 | 41 |

## Distribución propuesta por tamaño

| Grupo | Población | Muestra propuesta |
| --- | ---: | ---: |
| large | 3,529 | 150 |
| medium | 12,169 | 317 |
| small | 1,922 | 133 |

Tamaños iguales a los del evaluador actual: pequeño <1 % de área de imagen, mediano entre 1 % y 5 % inclusive y grande >5 %.

## Fuente metodológica y trazabilidad

Valliant, Dever y Kreuter (2015), [PracTools: Computations for Design of Finite Population Samples](https://www.rti.org/publication/practools-computations-design-finite-population-samples/fulltext.pdf), ecuación (2) y sección de muestreo estratificado. La elección de las cuotas mínimas y el cálculo sobre los datos de fauna son específicos de este análisis.

Los hashes de `manifest.json`, `occlusion_review.csv` y `label_review.csv` se verificaron iguales antes y después. No se cambiaron el entrenamiento, los resultados, las revisiones humanas, el notebook ni el PDF. Los datos y cálculos auditables están en `analysis.json`.
