# Analisis de presencia de dron - reinterpretacion del Experimento 2 (run02_60ep)

Documento complementario a `YOLO_PROGRESO.md`. Reinterpreta los resultados
finales del Stage 1 desde el punto de vista que de verdad importa para el
TFG: **dada una ventana de espectrograma, el sistema confirma la presencia
del dron si encuentra al menos un burst etiquetado como drone**. No nos
importa cuantos bursts detecta, ni si los localiza con precision
geometrica perfecta. Solo nos importa si hay o no hay senal de OcuSync en
la ventana.

Ultima actualizacion: 2026-05-26.

---

## 1. Cambio de optica respecto a YOLO_PROGRESO.md

`YOLO_PROGRESO.md` reporta las metricas que YOLOv8 genera por defecto:
mAP@50, mAP@50-95, precision y recall **a nivel de bounding box**. Cada
caja predicha se compara con cajas de ground truth via IoU, y el modelo es
penalizado cuando localiza mal una caja, aunque la deteccion sea correcta
en el sentido de "hay un burst aqui".

Esa lectura es la apropiada para evaluar al detector como tal, pero **no es
la pregunta operativa del TFG**. La pregunta operativa es:

> Dada una ventana de espectrograma de 0.1 s, el sistema dice "el dron
> esta presente" si y solo si encuentra al menos un burst que clasifica
> como drone. Cuantos detecta, donde exactamente los situa, o que
> confianza individual asigna, son detalles secundarios.

Bajo esa lente cambian dos cosas importantes:

1. **El recall ya no se mide sobre bursts individuales** sino sobre
   imagenes. Una imagen con 20 bursts reales en la que el modelo detecta
   1 sigue contando como una deteccion correcta de presencia. La mAP/recall
   geometrico bajaria por los 19 perdidos, pero la respuesta operativa
   ("hay dron") es correcta.
2. **Los falsos positivos se cuentan diferente**. Un false positive
   relevante es una imagen sin dron en la que el modelo encuentra al
   menos un bbox drone. Cuantas cajas espurias dibuje sobre esa imagen
   da igual.

---

## 2. Definicion formal de las metricas de presencia

Sea `I` una imagen del test set y sea `B_d(I)` el conjunto de bounding boxes
predichos con clase `drone` y confianza >= `thr`.

Definimos:

- `GT(I) = 1` si la imagen procede de una captura `mini3_drone_*` (la
  fuente RF dominante es el dron). En caso contrario, `GT(I) = 0`.
- `Pred(I) = 1` si `|B_d(I)| >= K` (al menos K bboxes drone que superan
  el umbral de confianza). En caso contrario, `Pred(I) = 0`.

Los dos hiperparametros operativos del clasificador de presencia son:

- `thr` (conf_threshold): confianza minima por bbox para que cuente.
- `K` (min_drone_bursts): numero minimo de bboxes que deben superar el
  umbral para confirmar presencia en la imagen.

El caso `K=1` es la version mas permisiva (la usada en la version
inicial del documento). Aumentar K exige evidencia repetida en la misma
ventana: en lugar de un solo burst sospechoso, se requieren varios.
Esto sube la robustez frente a falsos positivos puntuales a costa de
recall.

Con esa pareja `(GT, Pred)` por imagen se construye una matriz de
confusion binaria 2x2:

|              | Pred=1 (dron) | Pred=0 (no dron) |
|--------------|---------------|------------------|
| GT=1 (dron)  | TP            | FN               |
| GT=0 (no dron) | FP          | TN               |

Y a partir de ahi:

- `accuracy = (TP + TN) / N`
- `precision = TP / (TP + FP)`
- `recall = TP / (TP + FN)`             (sensibilidad, TPR)
- `specificity = TN / (TN + FP)`         (TNR)
- `false_positive_rate = FP / (TN + FP)` (1 - specificity)
- `F1 = 2 * precision * recall / (precision + recall)`

Esta es la metrica que el script `scripts/presence_analysis.py` calcula
sobre el test set del Stage 1.

---

## 3. Datos sobre los que se mide

Test set del Stage 1 (sin solapamiento de captura con train/val):

| Captura                              | Tipo    | Imagenes | Notas                       |
|--------------------------------------|---------|----------|-----------------------------|
| mini3_drone_d5_w1_s01_cap01          | drone   | 150      | dron a 5 m, WiFi nivel 1    |
| mini3_drone_d10_w3_s02_cap01         | drone   | 150      | dron a 10 m, WiFi nivel 3   |
| mini3_nodrone_w1_s01_cap03           | nodrone | 150      | WiFi nivel 1, sin dron      |
| mini3_nodrone_w3_s01_cap03           | nodrone | 150      | WiFi nivel 3, sin dron      |
| **Total**                            | **-**   | **600**  | balance perfecto 300/300    |

Cada imagen es un PNG de 1024x576 que representa 0.1 s de espectrograma.

---

## 4. Como reproducir las cifras

Desde la raiz del repo, con el entorno uv ya sincronizado:

```powershell
uv run python scripts/presence_analysis.py
```

Salidas en `data/mini3_detector_python_v3/artifacts/yolo/run02_60ep_test/presence/`:

- `per_image_predictions.csv` - una fila por imagen con
  `(path, capture_id, gt_presence, n_drone_bboxes, max_drone_conf, ...)`.
- `presence_metrics.json` - cifras agregadas a varios umbrales de
  confianza (0.10, 0.25, 0.40, 0.50) mas el desglose por captura.
- `presence_report.txt` - lectura legible del JSON.

El umbral por defecto (`--conf 0.25`) coincide con el que YOLO usa en
`model.val()`, de modo que los resultados son comparables con la matriz
de confusion reportada en `YOLO_PROGRESO.md`.

---

## 5. Resultados (ejecucion 2026-05-26)

### 5.1 Sweep bidimensional (conf x K)

Sweep completo sobre las 600 imagenes del test set (300 drone +
300 nodrone, balance 50/50). Cada fila es una combinacion de umbral de
confianza `conf` y minimo de bursts requeridos `K`:

| conf | K | TP  | FN | TN  | FP | Accuracy | Precision | Recall | Specificity | F1     |
|------|---|-----|----|-----|----|----------|-----------|--------|-------------|--------|
| 0.10 | 1 | 297 |  3 | 300 |  0 | 0.9950   | 1.0000    | 0.9900 | 1.0000      | 0.9950 |
| 0.10 | 2 | 286 | 14 | 300 |  0 | 0.9767   | 1.0000    | 0.9533 | 1.0000      | 0.9761 |
| 0.10 | 3 | 273 | 27 | 300 |  0 | 0.9550   | 1.0000    | 0.9100 | 1.0000      | 0.9529 |
| 0.10 | 4 | 267 | 33 | 300 |  0 | 0.9450   | 1.0000    | 0.8900 | 1.0000      | 0.9418 |
| 0.25 | 1 | 267 | 33 | 300 |  0 | 0.9450   | 1.0000    | 0.8900 | 1.0000      | 0.9418 |
| 0.25 | 2 | 249 | 51 | 300 |  0 | 0.9150   | 1.0000    | 0.8300 | 1.0000      | 0.9071 |
| 0.25 | 3 | 235 | 65 | 300 |  0 | 0.8917   | 1.0000    | 0.7833 | 1.0000      | 0.8785 |
| 0.25 | 4 | 226 | 74 | 300 |  0 | 0.8767   | 1.0000    | 0.7533 | 1.0000      | 0.8593 |
| 0.40 | 1 | 246 | 54 | 300 |  0 | 0.9100   | 1.0000    | 0.8200 | 1.0000      | 0.9011 |
| 0.40 | 2 | 228 | 72 | 300 |  0 | 0.8800   | 1.0000    | 0.7600 | 1.0000      | 0.8636 |
| 0.40 | 3 | 215 | 85 | 300 |  0 | 0.8583   | 1.0000    | 0.7167 | 1.0000      | 0.8350 |
| 0.40 | 4 | 211 | 89 | 300 |  0 | 0.8517   | 1.0000    | 0.7033 | 1.0000      | 0.8258 |
| 0.50 | 1 | 238 | 62 | 300 |  0 | 0.8967   | 1.0000    | 0.7933 | 1.0000      | 0.8848 |
| 0.50 | 2 | 221 | 79 | 300 |  0 | 0.8683   | 1.0000    | 0.7367 | 1.0000      | 0.8484 |
| 0.50 | 3 | 207 | 93 | 300 |  0 | 0.8450   | 1.0000    | 0.6900 | 1.0000      | 0.8166 |
| 0.50 | 4 | 201 | 99 | 300 |  0 | 0.8350   | 1.0000    | 0.6700 | 1.0000      | 0.8024 |

**Lo mas llamativo de la tabla (4 hechos invariantes en las 16 filas):**

- **Precision = 1.000 y Specificity = 1.000 en las 16 configuraciones.**
  Cero falsos positivos en ningun punto operativo. El sistema **nunca**
  confirma presencia de dron en una ventana de captura nodrone, ni
  cuando se exige un solo burst con confianza 0.10, ni cuando se exigen
  4 bursts con confianza 0.50.
- **FN crece monotonicamente con conf y con K**, como se esperaria.
  La recall decae suavemente al endurecer cualquiera de los dos ejes.
- **F1 maximo en (conf=0.10, K=1): 0.995**. Punto operativo mas
  permisivo. F1 minimo en (conf=0.50, K=4): 0.802.
- En las 16 configuraciones, **las 4 capturas del test set quedan
  correctamente clasificadas** (las dos drone CONFIRMADAS, las dos
  nodrone limpias).

### 5.2 Matriz de recall vs (conf, K)

Vista compacta del recall en porcentaje (las celdas con specificity y
precision son todas 1.000, no aportan informacion):

|        | K=1   | K=2   | K=3   | K=4   |
|--------|-------|-------|-------|-------|
| conf=0.10 | 99.0  | 95.3  | 91.0  | 89.0  |
| conf=0.25 | 89.0  | 83.0  | 78.3  | 75.3  |
| conf=0.40 | 82.0  | 76.0  | 71.7  | 70.3  |
| conf=0.50 | 79.3  | 73.7  | 69.0  | 67.0  |

El recall cae unos 10 puntos por columna (al subir K en una unidad) y
otros 10-15 puntos por fila (al subir conf en un escalon). Las dos
variables son aproximadamente igual de "duras" para el modelo.

### 5.3 Recall por captura (donde se reparte el coste)

El recall global oculta una asimetria fuerte entre las dos capturas
drone. Recall por captura en cada configuracion:

**Captura mini3_drone_d10_w3_s02_cap01 (10 m, WiFi nivel 3 saturado):**

|        | K=1     | K=2     | K=3     | K=4     |
|--------|---------|---------|---------|---------|
| conf=0.10 | 100.00% | 100.00% | 100.00% | 100.00% |
| conf=0.25 | 100.00% | 100.00% | 100.00% | 100.00% |
| conf=0.40 | 100.00% | 100.00% | 100.00% | 100.00% |
| conf=0.50 | 100.00% | 100.00% | 100.00% | 100.00% |

**Captura mini3_drone_d5_w1_s01_cap01 (5 m, WiFi nivel 1 limpio):**

|        | K=1   | K=2   | K=3   | K=4   |
|--------|-------|-------|-------|-------|
| conf=0.10 | 98.00 | 90.67 | 82.00 | 78.00 |
| conf=0.25 | 78.00 | 66.00 | 56.67 | 50.67 |
| conf=0.40 | 64.00 | 52.00 | 43.33 | 40.67 |
| conf=0.50 | 58.67 | 47.33 | 38.00 | 34.00 |

**Lectura:** la captura "dificil" (10 m con WiFi saturado) es perfecta
en las 16 configuraciones. Todo el FN del sistema viene de la captura
"facil" (5 m con WiFi limpio), donde la densidad de bursts visibles
por ventana parece ser baja. La asimetria es muy fuerte:

- En el punto operativo mas estricto (conf=0.50, K=4) la captura d10/w3
  sigue confirmada en 150/150 imagenes (100%), mientras la captura
  d5/w1 baja a 51/150 (34%).
- Aun asi, las dos capturas quedan CONFIRMADAS en las 16
  configuraciones porque basta una sola imagen positiva para que la
  captura completa quede confirmada.

### 5.4 Eleccion del punto operativo

De las 16 configuraciones, hay tres candidatos defendibles para la
memoria, segun la filosofia que se quiera transmitir:

| Filosofia | Punto operativo | F1 | Recall | Specificity | Argumento |
|-----------|-----------------|----|--------|-------------|-----------|
| **Maxima sensibilidad** | conf=0.10, K=1 | 0.995 | 99.0% | 100.0% | Confirma cuanto antes posible. Util para alerta temprana. |
| **Equilibrado (recomendado)** | conf=0.10, K=3 | 0.953 | 91.0% | 100.0% | Exige 3 bursts independientes en 0.1 s. Robusto frente a confianzas individuales bajas. |
| **Ultra-conservador** | conf=0.50, K=4 | 0.802 | 67.0% | 100.0% | Exige 4 bursts con confianza alta. Para aplicaciones donde el coste de un FP es altisimo. |

La fila marcada como **recomendada (conf=0.10, K=3)** es la que mejor
representa la pregunta cientifica del TFG. Exige evidencia repetida en
cada ventana, lo que reduce el riesgo conceptual de "una sola caja
afortunada por imagen", y sigue manteniendo precision y specificity
perfectas con un recall del 91%. Es la mejor combinacion entre
robustez interpretativa y rendimiento operativo.

### 5.2 Resumen por captura

Mismo sweep, ahora desglosado por las cuatro capturas del test set.
"Imgs positivas" = imagenes de la captura en las que el modelo emite
al menos un bbox drone con conf >= umbral.

**Umbral 0.10:**

| Captura                       | Tipo    | Imgs positivas / total | %     | Veredicto    |
|-------------------------------|---------|------------------------|-------|--------------|
| mini3_drone_d10_w3_s02_cap01  | drone   | 150 / 150              | 100.00 | CONFIRMADO   |
| mini3_drone_d5_w1_s01_cap01   | drone   | 147 / 150              |  98.00 | CONFIRMADO   |
| mini3_nodrone_w1_s01_cap03    | nodrone |   0 / 150              |   0.00 | limpia       |
| mini3_nodrone_w3_s01_cap03    | nodrone |   0 / 150              |   0.00 | limpia       |

**Umbral 0.25 (operativo de referencia):**

| Captura                       | Tipo    | Imgs positivas / total | %     | Veredicto    |
|-------------------------------|---------|------------------------|-------|--------------|
| mini3_drone_d10_w3_s02_cap01  | drone   | 150 / 150              | 100.00 | CONFIRMADO   |
| mini3_drone_d5_w1_s01_cap01   | drone   | 117 / 150              |  78.00 | CONFIRMADO   |
| mini3_nodrone_w1_s01_cap03    | nodrone |   0 / 150              |   0.00 | limpia       |
| mini3_nodrone_w3_s01_cap03    | nodrone |   0 / 150              |   0.00 | limpia       |

**Umbral 0.40:**

| Captura                       | Tipo    | Imgs positivas / total | %     | Veredicto    |
|-------------------------------|---------|------------------------|-------|--------------|
| mini3_drone_d10_w3_s02_cap01  | drone   | 150 / 150              | 100.00 | CONFIRMADO   |
| mini3_drone_d5_w1_s01_cap01   | drone   |  96 / 150              |  64.00 | CONFIRMADO   |
| mini3_nodrone_w1_s01_cap03    | nodrone |   0 / 150              |   0.00 | limpia       |
| mini3_nodrone_w3_s01_cap03    | nodrone |   0 / 150              |   0.00 | limpia       |

**Umbral 0.50:**

| Captura                       | Tipo    | Imgs positivas / total | %     | Veredicto    |
|-------------------------------|---------|------------------------|-------|--------------|
| mini3_drone_d10_w3_s02_cap01  | drone   | 150 / 150              | 100.00 | CONFIRMADO   |
| mini3_drone_d5_w1_s01_cap01   | drone   |  88 / 150              |  58.67 | CONFIRMADO   |
| mini3_nodrone_w1_s01_cap03    | nodrone |   0 / 150              |   0.00 | limpia       |
| mini3_nodrone_w3_s01_cap03    | nodrone |   0 / 150              |   0.00 | limpia       |

**En los cuatro umbrales, el veredicto por captura es correcto sin
excepcion**: las dos capturas con dron quedan confirmadas con mayoria
masiva de ventanas positivas, y las dos sin dron se mantienen sin una
sola alerta espuria.

### 5.5 Hallazgo contraintuitivo: la captura "facil" es la que falla

La asimetria entre las dos capturas drone (seccion 5.3) es el resultado
mas inesperado del sweep. A priori uno esperaria que un dron a 5 m con
WiFi limpio fuese mas facil de detectar que a 10 m con WiFi saturado.
El modelo se comporta al reves: la primera nunca falla en ninguna
configuracion, la segunda se degrada en cuanto se sube K o conf.

Hipotesis razonables para discutir en la memoria:

1. **Mayor densidad de bursts en d10/w3.** Cuanto mas congestionada
   esta la banda WiFi, mas tiene que retransmitir el enlace OcuSync
   para mantener la conexion. Eso aumenta la tasa de hopping efectiva
   y el numero de bursts por ventana de 0.1 s, lo que hace mas
   probable que al menos K bursts crucen cualquier umbral de
   confianza. La progresion vertical de la tabla d10/w3 (100% en las
   16 celdas) es la evidencia mas fuerte: incluso K=4 con conf=0.50
   se confirma en todas las ventanas, lo que implica que en cada
   imagen de 0.1 s hay como minimo 4 bursts con confianza >=0.50.
2. **Saturacion del receptor a 5 m.** El SDR esta a poca distancia y
   la ganancia es la misma en todas las capturas (g10). Es posible que
   algunos bursts saturen el rango dinamico del espectrograma y pierdan
   morfologia distinguible, bajando la confianza individual del modelo.
3. **Sesgo del training set.** Si el train mezcla varias distancias
   pero la representacion del 5 m es minoritaria, el modelo puede
   tener una preferencia implicita por la firma RF que ve a distancias
   medias. Conviene revisar la composicion del train.
4. **Variabilidad propia de la captura.** Es una sola sesion de 30 s
   de cada caso. Puede haber un efecto de captura individual (modo de
   vuelo, momento de la captura, etc.) que no se generalizaria a otra
   sesion de la misma distancia.

Cualquiera que sea la causa, el efecto operativo es **positivo**: el
sistema funciona mejor en el caso dificil (10 m + WiFi saturado), que
es precisamente el escenario realista de despliegue. Para una alerta
operativa, si la captura es completa de 30 s con varias decenas de
ventanas, el sistema confirma presencia con holgura incluso en el caso
"problematico" de 5 m.

---

## 6. Como leer las cifras

### 6.1 La metrica que importa

La cifra de cabecera para la memoria es el **recall de presencia** al
umbral 0.25:

> "El sistema confirma correctamente la presencia del dron en el X% de
> las ventanas de 0.1 s en las que el dron esta emitiendo OcuSync."

Esta cifra responde directamente a la pregunta de investigacion del TFG.
Es la metrica natural para un sistema de alerta: cuando el dron entra en
la zona observada, ¿en que fraccion de las ventanas analizadas levantamos
la mano?

### 6.2 Comparacion con las cifras bbox del Stage 1

`YOLO_PROGRESO.md` reporta recall bbox-level = 0.391. Es probable
(esperable) que el **recall de presencia sea sustancialmente mas alto**
que ese 0.391, porque:

- Una imagen tipica de la captura `mini3_drone_*` contiene varios bursts
  FHSS de OcuSync. Aunque el modelo solo detecte 1 de cada 3, la imagen
  sigue clasificandose como "dron presente".
- El paper trabaja con ventanas de 0.1 s en las que la tasa de hopping
  de OcuSync genera multiples bursts por segundo. La probabilidad de no
  detectar **ningun** burst es mucho menor que la de fallar bursts
  individuales.

La caida natural del recall bbox al recall de presencia debe ser
**positiva**: la presencia es mas facil de detectar que la localizacion
precisa. Si las cifras no muestran esa caida positiva, hay algo que
revisar (por ejemplo, capturas con muy pocos bursts en algunas ventanas).

### 6.3 Comparacion con la matriz de confusion bbox

La matriz de confusion bbox de `YOLO_PROGRESO.md` muestra:

- `drone <-> interference` cruzado: 0.00 (cero confusion de clase entre
  bursts).
- `background -> drone`: 0.82 (de los falsos positivos, el 82% se
  etiqueta como drone).

A nivel **bbox** ese 82% de falsos positivos hacia drone parece alarmante,
pero **a nivel imagen** se diluye:

- Si una imagen `nodrone` recibe varios FP de tipo drone, sigue contando
  como **una sola** imagen de "presencia incorrectamente confirmada". Los
  FP repetidos sobre la misma imagen no cuentan dos veces.
- Por tanto, la metrica de presencia es mas indulgente con la sobre-
  deteccion que la mAP. Y eso es exactamente lo que queremos para un
  sistema operativo: no penalizar al modelo por dibujar tres cajas
  donde habia un burst, siempre que las cajas esten en imagenes donde
  el dron de verdad esta.

### 6.4 El falso positivo que si importa

El unico fallo grave bajo esta optica es:

> Una imagen de captura `mini3_nodrone_*` en la que el modelo dibuja al
> menos un bbox drone con confianza >= 0.25.

Esa imagen se contabiliza como FP de presencia. La fila de la tabla
`Resumen por captura` te dice cuantas hay y en que captura. Si el numero
es bajo (< 5%) y se concentra en la captura `nodrone_w3` (la de WiFi
saturado), tienes una historia clara: **el sistema no se confunde con
interferencia ligera y solo levanta alguna alerta espuria bajo
congestion RF maxima**, que es justo el escenario mas dificil.

---

## 7. Texto sugerido para la memoria

### 7.1 Capitulo de Resultados, despues de las cifras bbox

> Las metricas mAP@50 y mAP@50-95 reportadas en la seccion anterior
> evaluan al detector como tal, comparando bounding boxes individuales
> contra ground truth via IoU. Esa lectura es apropiada para auditar la
> calidad geometrica del Stage 1, pero no responde directamente a la
> pregunta operativa del trabajo. La pregunta operativa es: dada una
> ventana de 0.1 s de espectrograma, ¿el sistema confirma o no la
> presencia del dron?
>
> Definimos por tanto una metrica binaria a nivel de imagen
> parametrizada por dos hiperparametros operativos: el umbral de
> confianza `conf` que debe superar cada bounding box predicho de clase
> drone, y el numero minimo `K` de bounding boxes que deben superar ese
> umbral para que la imagen se clasifique como "dron presente". El caso
> `K=1` es el mas permisivo (basta con un burst sospechoso); aumentar
> K exige evidencia repetida en la misma ventana y reduce el riesgo de
> falsos positivos por una sola caja afortunada.
>
> El sweep bidimensional sobre el test set (600 imagenes, balance 50/50,
> 4 umbrales x 4 valores de K = 16 configuraciones) arroja un resultado
> uniforme y muy fuerte: **precision = 1.000 y specificity = 1.000 en
> las 16 configuraciones**. Cero falsos positivos sobre las 300 imagenes
> sin dron en cualquier punto operativo. El sistema nunca confirma
> presencia de dron en una ventana de captura nodrone, ni cuando se
> exige un solo burst con confianza 0.10, ni cuando se exigen 4 bursts
> con confianza 0.50.
>
> El recall varia entre el 99.0% en la configuracion mas permisiva
> (conf=0.10, K=1) y el 67.0% en la mas estricta (conf=0.50, K=4),
> siempre con precision y specificity perfectas. Como punto operativo
> recomendado se selecciona **conf=0.10 con K=3** (recall 91.0%,
> F1=0.953), que exige al menos tres bursts independientes en la
> ventana de 0.1 s para confirmar la presencia. Esta configuracion
> protege frente al riesgo conceptual de "un solo burst afortunado por
> imagen" sin penalizar significativamente el recall global.

### 7.2 Capitulo de Discusion

> La asimetria entre el recall bbox (0.391) y el recall de presencia
> (entre 0.67 y 0.99 segun la configuracion operativa) muestra que el
> Stage 1 funciona muy bien como detector de "hay o no hay dron" pese a
> tener un rendimiento bbox modesto. Esta propiedad es deseable para el
> despliegue: el sistema final no necesita conocer la posicion exacta
> ni contar los bursts del enlace OcuSync; solo necesita levantar la
> alerta cuando el enlace esta activo.
>
> Igual de notable es la **specificity perfecta** (1.000) en las 16
> configuraciones del sweep. Las 300 imagenes de captura sin dron,
> incluyendo las 150 con WiFi saturado (nivel 3), no producen una sola
> alerta espuria, ni siquiera cuando se relaja al maximo el umbral de
> confianza (0.10) o se acepta una sola caja drone por imagen. El
> modelo distingue con holgura absoluta la firma FHSS de OcuSync de las
> tramas WiFi residuales y de los bursts BT presentes en las capturas
> nodrone. Es la prueba empirica de la hipotesis central del TFG: la
> firma OcuSync del DJI MINI3 es separable morfologicamente del resto
> de senales 2.4 GHz mediante una red de deteccion de objetos sobre
> espectrograma.
>
> Un resultado inesperado emerge al desglosar el recall por captura:
> la captura con peor escenario nominal (10 m de distancia, WiFi nivel 3
> saturado) acierta el **100% de las 150 ventanas en las 16
> configuraciones del sweep**, mientras que la captura aparentemente
> mas facil (5 m, WiFi nivel 1 limpio) baja al 78% en el punto
> operativo recomendado y hasta el 34% en el ultra-conservador. El
> efecto va en contra de la intuicion fisica (mas cerca y con menos
> interferencia deberia ser mas facil) y sugiere que la captura
> congestionada produce mas bursts visibles por ventana porque el
> enlace OcuSync retransmite mas en presencia de WiFi competidor, o que
> la senal a corta distancia satura parte del rango dinamico del
> espectrograma y baja la confianza individual del detector.
> Independientemente de la causa, el efecto operativo es favorable: el
> sistema funciona mejor en el escenario realista de despliegue.

### 7.3 Para el resumen ejecutivo

> El detector del Stage 1 confirma la presencia del dron con un recall
> del **91.0%** a nivel de ventana de 0.1 s en el punto operativo
> recomendado (conf=0.10, K=3 bursts minimos por imagen), manteniendo
> una tasa de falsos positivos del **0%** sobre las 300 imagenes sin
> dron. La specificity perfecta se conserva en las 16 configuraciones
> evaluadas (sweep de 4 umbrales de confianza x 4 valores de K). Sobre
> las capturas completas (30 s de grabacion), el veredicto de presencia
> es correcto en las 4 capturas del test set sin excepcion en cualquier
> punto operativo: las dos de dron quedan confirmadas y las dos sin
> dron permanecen sin una sola alerta espuria.

---

## 8. Sanity checks (ejecucion 2026-05-26)

Verificaciones sobre las cifras reales del sweep 4x4:

1. **Balance del test set:** 300 imgs drone + 300 imgs nodrone = 600.
   OK en las 16 configuraciones.
2. **Suma TP + FN = 300** en cada fila de la tabla. Verificado para las
   16 filas (todas las imgs drone tienen veredicto).
3. **Suma TN + FP = 300** en cada fila. Verificado para las 16 filas
   (todas las imgs nodrone tienen veredicto).
4. **FP = 0 en las 16 configuraciones.** Las 300 imagenes nodrone
   nunca activan ni siquiera una caja drone que supere el umbral mas
   permisivo (0.10). Confirma que el rechazo no es marginal: el
   `max_drone_conf` en las 300 imagenes nodrone esta estrictamente
   por debajo de 0.10.
5. **Monotonia de FN en conf y K.** El numero de falsos negativos crece
   tanto al subir conf (fila a fila) como al subir K (columna a columna)
   sin excepciones. La superficie es monotonica como se esperaria.
6. **Recall de presencia >= recall bbox (0.391).** OK en las 16
   configuraciones, con margen amplio incluso en la mas estricta
   (recall 0.67 vs 0.391 bbox).
7. **Veredicto por captura correcto sin excepcion.** Las 4 capturas
   del test set quedan correctamente clasificadas en las 16
   configuraciones del sweep.
8. **Captura d10/w3 con recall 100% en las 16 configuraciones.**
   Indica que cada imagen de 0.1 s de esa captura tiene al menos 4
   bursts con confianza >= 0.50. Estructuralmente robusta.

---

## 9. Limitaciones de la metrica de presencia

La metrica de presencia es la pregunta operativa correcta, pero conviene
documentar lo que **no** mide:

1. **No mide la calidad geometrica de la deteccion**. Una imagen con un
   bbox drone correctamente clasificado pero geometricamente desplazado
   cuenta como TP. Si el Stage 2 (clasificador sobre crops) usa esos
   bboxes como entrada, la calidad geometrica si importa y hay que
   mirar las metricas mAP del YOLO_PROGRESO.
2. **No mide capacidad de conteo**. Si la aplicacion requiere distinguir
   "un dron" de "varios drones", esta metrica no lo cubre.
3. **No considera la distribucion temporal de detecciones dentro de la
   captura**. Para un sistema real probablemente quieras exigir
   estabilidad temporal (k de N ventanas consecutivas positivas) para
   levantar la alerta. La metrica de presencia por ventana es la base
   sobre la que se construye esa regla, no la regla en si.

---

## 10. Ficheros relevantes

- `scripts/presence_analysis.py` - script que genera las cifras.
- `data/mini3_detector_python_v3/artifacts/yolo/run02_60ep/weights/best.pt` - modelo evaluado.
- `data/mini3_detector_python_v3/yolo/test.txt` - rutas del test set.
- `data/mini3_detector_python_v3/artifacts/yolo/run02_60ep_test/presence/` - salidas del script.
- `YOLO_PROGRESO.md` - documento base del Stage 1 con metricas bbox.
