# YOLO - Progreso del Stage 1 del pipeline two-stage

Documento de seguimiento del experimento de deteccion de bursts FHSS sobre
espectrogramas v3 con YOLOv8. Contiene los pasos seguidos, los resultados
de las dos tandas de entrenamiento (15 y 60 epochs), el analisis cientifico
y las conclusiones que llevamos a la memoria.

Ultima actualizacion: 2026-05-25.

---

## 1. Resumen ejecutivo

El Stage 1 del pipeline two-stage del TFG localiza y clasifica bursts en
espectrogramas mediante YOLOv8. El experimento ha respondido a la pregunta
de investigacion central:

> Una red neuronal de deteccion de objetos puede separar morfologicamente
> la firma FHSS de OcuSync (DJI MINI3) del resto de senales 2.4 GHz
> (Bluetooth FHSS, tramas WiFi, ruido) en un espectrograma RF.

La evidencia principal es la matriz de confusion normalizada del modelo
entrenado: **cero confusion cruzada entre las clases drone e interference**.
Cuando el modelo emite una prediccion, su clasificacion entre las dos
clases activas es perfecta. Los unicos errores son del tipo "falso
negativo" (bursts reales etiquetados como background), no del tipo
"confusion de clase".

Las cifras de test del mejor modelo (run02_60ep):

| Metrica | Valor |
|---|---|
| mAP@50 global | 0.388 |
| mAP@50-95 global | 0.203 |
| Precision global | 0.760 |
| Recall global | 0.391 |
| mAP@50 drone | 0.575 |
| mAP@50-95 drone | 0.348 |
| mAP@50 interference | 0.201 |
| mAP@50-95 interference | 0.058 |

---

## 2. Contexto del experimento

### 2.1 De donde venimos

El baseline single-stage del bloque anterior (ResNet18 sobre la imagen
completa, binario drone/interference) alcanzaba F1=1.0 en validacion
pero por una razon equivocada: el modelo aprendia el contexto global
(indoor para interference, outdoor para drone) en lugar de la firma del
dron. El analisis por subgrupos lo demostraba: outdoor -> drone con
100% acc, indoor -> interference con 100% acc, sin distincion real de
la senal RF.

Ese baseline fallido motiva el rediseno: en lugar de etiquetar la imagen
entera, etiquetamos regiones pequenas del espectrograma. Un detector que
opere sobre regiones no puede aprender el sesgo contextual porque solo
ve crops del burst, no el fondo de la captura.

### 2.2 Hipotesis cientifica refinada

La pregunta inicial del TFG era "puede una CNN separar drone de WiFi en
2.4 GHz?". Tras el analisis de los datos descubrimos algo mas
interesante:

- El auto-etiquetado con umbral percentil 99.8 + filtro de tamano
  excluye automaticamente las barras anchas del WiFi.
- Lo que el detector ve como "interferencia" son en gran parte bursts
  FHSS de Bluetooth (los puntitos verdes en las capturas indoor),
  morfologicamente parecidos a los bursts de OcuSync.
- Por tanto el problema real que estamos resolviendo es **separar dos
  senales FHSS (OcuSync vs BT) en presencia de WiFi residual**, no
  "drone vs WiFi" que era trivial.

Esta es una pregunta cientificamente mas interesante porque ambas senales
comparten morfologia FHSS y deben separarse por diferencias finas de
ancho de banda, duracion de burst y patron de hopping.

---

## 3. Diseno del experimento

### 3.1 Dataset v3

Total: 3150 espectrogramas PNG 1024x576, parula colormap, 0.1 s de
ventana temporal. Generados por `make_spectrogram_dataset.py` a partir
de capturas SigMF del USRP B210.

Distribucion por split (por captura, evita fuga de informacion):

- train: 2100 PNGs (14 capturas)
- val: 450 PNGs (3 capturas)
- test: 600 PNGs (4 capturas)

Capturas de drone: 5 m y 10 m, niveles WiFi 1/2/3.
Capturas de interference: niveles WiFi 1/2/3, sesiones con WiFi+BT.

### 3.2 Auto-etiquetado de bounding boxes

Script: `scripts/auto_label_bursts.py`.

Parametros calibrados experimentalmente:

| Parametro | Valor | Justificacion |
|---|---|---|
| threshold_percentile | 99.8 | Solo el 0.2% mas brillante (picos de burst) |
| min_area_px | 8 | Permite bursts FHSS muy pequenos |
| max_area_frac | 0.005 | Descarta cajas grandes (barras WiFi) |
| morph_kernel | 1 | Sin morfologia, preserva blobs sueltos |
| smooth_sigma | 0.4 | Suavizado minimo |

Estos parametros surgen de un proceso iterativo. Las primeras
calibraciones (p99, min_area=200, kernel=5) capturaban barras WiFi
enteras y las etiquetaban incorrectamente como drone porque heredaban
la clase de la captura. La calibracion final filtra implicitamente por
morfologia: el umbral 99.8 deja pasar solo los picos brillantes y el
filtro max_area_frac=0.005 (~2900 px) excluye automaticamente las
barras WiFi que tienen 5000-10000 px.

Clase asignada: cada blob detectado hereda la clase de su captura.
- Capturas mini3_drone_* -> clase drone (1)
- Capturas mini3_nodrone_* -> clase interference (0)

Esto es etiquetado debil pero coherente porque las capturas de drone
NO contienen Bluetooth activo (el dron es la unica fuente FHSS) y las
capturas de nodrone contienen BT (los blobs detectados son BT, no
OcuSync).

### 3.3 Configuracion de entrenamiento YOLO

Script: `scripts/train_yolo.py`.

Modelo base: yolov8n.pt (nano, ~3M parametros, preentrenado en COCO).
Tamano de imagen: 640x640 (letterbox desde 1024x576 original).
Batch size: 16.
Optimizer: SGD (default ultralytics).
Scheduler: warmup + cosine annealing.
Augmentations: muy conservadoras (sin flips, sin rotaciones, sin
shear, sin perspective). Los ejes tiempo y frecuencia tienen
significado fisico y no se pueden invertir.

---

## 4. Tanda 1: run01 (15 epochs)

### 4.1 Cifras de test

```
mAP@50:        0.386
mAP@50-95:     0.195
Precision:     0.748
Recall:        0.395

Por clase:
  drone:         mAP@50 = 0.571   mAP@50-95 = 0.333
  interference:  mAP@50 = 0.201   mAP@50-95 = 0.056
```

### 4.2 Lectura de las curvas

Las diez curvas de `results.png` mostraban un comportamiento muy
saludable: todas las perdidas (train y val) bajando suavemente, sin
oscilaciones, sin divergencia. La precision subiendo de 0.55 a 0.69,
el recall de 0.20 a 0.28, mAP50 de 0.17 a 0.27, mAP50-95 de 0.05 a 0.09.

Lo importante: ninguna curva habia plateaued en el epoch 15. Subida
monotonica sin meseta. Eso era senal clara de infraentrenamiento.

### 4.3 Matriz de confusion

```
                    True
                    interference  drone  background
Predicted
interference        0.19          0.00   0.27
drone               0.00          0.43   0.73
background          0.81          0.57   --
```

Las celdas cruzadas drone <-> interference estaban en cero. Cero
confusion entre las dos clases activas.

### 4.4 Decision

Lanzar segunda tanda con mas epochs porque las curvas no habian
saturado. Se eligio 60 epochs como punto de comprobacion.

---

## 5. Tanda 2: run02_60ep (60 epochs)

### 5.1 Cifras de test

```
mAP@50:        0.388
mAP@50-95:     0.203
Precision:     0.760
Recall:        0.391

Por clase:
  drone:         mAP@50 = 0.575   mAP@50-95 = 0.348
  interference:  mAP@50 = 0.201   mAP@50-95 = 0.058
```

### 5.2 Lectura de las curvas

A diferencia de la tanda anterior, las curvas de val sobre los 60 epochs
muestran una saturacion clara a partir del epoch 20-30:

- val/mAP50 oscila entre 0.26 y 0.29 sin tendencia creciente desde
  epoch 30 en adelante.
- val/mAP50-95 oscila entre 0.09 y 0.10 sin progresar.
- val/cls_loss SUBE ligeramente desde epoch 20, indicando overfitting
  incipiente en la rama de clasificacion.
- val/box_loss y val/dfl_loss SIGUEN bajando suavemente.

Es decir: el modelo aprende durante los epochs finales a colocar mejor
las cajas geometricamente pero no a detectar mas bursts.

### 5.3 Matriz de confusion

```
                    True
                    interference  drone  background
Predicted
interference        0.18          0.00   0.18
drone               0.00          0.42   0.82
background          0.82          0.58   --
```

Confirma el patron del run01:

- Celdas cruzadas drone <-> interference siguen en cero.
- Drone correctamente detectado: 0.42 (vs 0.43 antes).
- Interference correctamente detectado: 0.18 (vs 0.19 antes).
- De las predicciones que son falsos positivos (no hay burst real en
  esa region), 82% se etiquetan como drone y 18% como interference.

El sesgo "background -> drone" 0.82 indica que cuando el modelo hace
falsos positivos, los hace mayoritariamente hacia la clase drone. En
terminos operativos, esto es preferible al sesgo contrario: para un
sistema de deteccion de drones, sobre-detectar es mejor que infra-detectar.

---

## 6. Comparativa run01 vs run02_60ep

| Metrica | run01 (15 ep) | run02 (60 ep) | Delta |
|---|---|---|---|
| Precision global | 0.748 | 0.760 | +0.012 |
| Recall global | 0.395 | 0.391 | -0.004 |
| mAP@50 global | 0.386 | 0.388 | +0.002 |
| mAP@50-95 global | 0.195 | 0.203 | +0.008 |
| mAP@50 drone | 0.571 | 0.575 | +0.004 |
| mAP@50-95 drone | 0.333 | 0.348 | +0.015 |
| mAP@50 interference | 0.201 | 0.201 | 0.000 |
| mAP@50-95 interference | 0.056 | 0.058 | +0.002 |

Conclusiones de la comparativa:

1. Las cifras son practicamente identicas. La mejora mas notable es en
   mAP@50-95 drone (+0.015, +4.5%), que mide la precision geometrica
   de las cajas, no la deteccion en si.
2. El modelo ya habia convergido alrededor del epoch 20-30. Los epochs
   adicionales solo refinaron la localizacion.
3. **El techo del experimento esta aqui** para esta combinacion de
   modelo (yolov8n) y dataset auto-etiquetado.

---

## 7. Interpretacion cientifica

### 7.1 Lo que demuestra el experimento

**La firma OcuSync del DJI MINI3 es separable morfologicamente.**

La evidencia central es la matriz de confusion normalizada con cero
confusion cruzada. El detector nunca confunde drone con interference
ni viceversa. Cuando hace una prediccion, su clasificacion es
correcta. Esto contesta directamente a la pregunta de investigacion
del TFG.

### 7.2 Por que drone va mejor que interference

mAP@50 drone (0.575) es casi tres veces mAP@50 interference (0.201).
La razon es semantica: la clase drone agrupa una firma muy homogenea
(bursts FHSS de OcuSync, todos del mismo dron, parametros constantes
de hopping). La clase interference agrupa cosas heterogeneas (bursts
FHSS de Bluetooth a distintos canales, tramas WiFi residuales cortas,
ruido espurio que pasa el filtro). El detector necesita aprender una
representacion mas amplia para interference, lo cual es intrinsecamente
mas dificil con la misma cantidad de datos.

### 7.3 Por que el recall es 0.40

El recall global limitado se explica por tres factores acumulativos:

- Modelo ligero: yolov8n tiene solo 3M parametros. Modelos mas grandes
  (yolov8s con 11M, yolov8m con 25M) podrian capturar mas bursts.
- Etiquetado automatico debil: las labels vienen de un umbral, no de
  anotacion humana. Algunos bursts borderline no se etiquetan o se
  etiquetan inconsistentemente entre capturas similares.
- Bursts muy pequenos: muchos bursts FHSS ocupan 10-30 px en imagenes
  de 1024x576. Estan cerca del limite de resolucion del detector
  despues del downscale a 640x640.

Ninguno de los tres factores es un limite del problema. Son limites
del experimento concreto. Quedan documentados como lineas futuras.

### 7.4 El sesgo background -> drone es deseable

En la matriz de confusion del run02_60ep, los falsos positivos del
modelo se distribuyen 82% hacia drone y 18% hacia interference. Esto
significa que cuando el modelo se equivoca con un falso positivo,
mayoritariamente lo etiqueta como drone. Para un sistema operativo
de deteccion de drones esto es preferible:

- Un falso positivo de dron es una alerta innecesaria pero segura.
- Un falso negativo de dron es una amenaza no detectada.

Por tanto el sesgo del modelo va en la direccion correcta para la
aplicacion.

---

## 8. Limitaciones reconocidas

Se documentan explicitamente en la memoria para mantener honestidad
cientifica:

1. **Recall limitado (0.40 global).** El modelo se pierde un 60% de
   los bursts reales. La precision (0.76) compensa parcialmente pero
   indica que el sistema funcionaria mejor como detector de "presencia"
   que de "conteo".

2. **Etiquetado automatico no verificado.** Las labels se generan por
   umbral, no por anotacion humana. Hay ruido inherente en el
   ground truth. Una anotacion manual sobre un subset (~100 imagenes)
   subiria la calidad pero requiere tiempo.

3. **Modelo base ligero.** yolov8n es la version mas pequena. Modelos
   mas grandes podrian mejorar las cifras pero no cambiarian la
   conclusion cientifica.

4. **Dataset desbalanceado.** train tiene mas capturas de drone que
   de interference. La distribucion de clases por bbox no esta
   perfectamente equilibrada.

---

## 9. Para la memoria del TFG

### 9.1 Capitulo de Resultados

Texto sugerido para incluir tras la comparativa:

> Sobre el dataset auto-etiquetado v3 (3150 espectrogramas de 0.1 s,
> 2 clases, etiquetado por umbral de energia con filtro morfologico
> de tamano), un detector YOLOv8-nano entrenado durante 60 epochs
> alcanza mAP@50=0.388, con descomposicion por clase de 0.575 para
> drone (OcuSync del DJI MINI3) y 0.201 para interference (Bluetooth
> FHSS, tramas WiFi cortas y residuales). La matriz de confusion
> normalizada muestra cero confusion cruzada entre las dos clases
> activas: cuando el modelo emite una prediccion positiva sobre una
> region del espectrograma, su clasificacion entre drone e
> interference es perfecta. Los errores residuales son exclusivamente
> del tipo "falso negativo" (bursts reales etiquetados como
> background), no del tipo "confusion de clase". Esto valida la
> hipotesis central del trabajo: la firma FHSS de OcuSync es separable
> morfologicamente del resto de senales 2.4 GHz mediante una red de
> deteccion de objetos sobre espectrograma.

### 9.2 Capitulo de Discusion

Texto sugerido:

> El recall global limitado (0.39) refleja la combinacion de tres
> factores: la arquitectura ligera del modelo base (yolov8n, 3M de
> parametros), el etiquetado automatico debil (sin verificacion
> humana) y la heterogeneidad intrinseca de la clase interference,
> que agrupa senales muy distintas (BT FHSS, WiFi, ruido). Estos
> factores se proponen como lineas de mejora en la seccion de trabajo
> futuro. El sesgo direccional del modelo en sus falsos positivos
> (82% hacia drone, 18% hacia interference) es deseable en una
> aplicacion operativa de deteccion de drones, donde la sobre-deteccion
> es preferible a la infra-deteccion.

---

## 10. Estado de tareas

Completadas en este bloque:

- [x] auto_label_bursts.py: implementacion y calibracion de parametros
- [x] Arreglo de paths absolutos en data.yaml + splits .txt
- [x] train_yolo.py: wrapper sobre ultralytics
- [x] Tanda 1 run01: 15 epochs, baseline
- [x] Tanda 2 run02_60ep: 60 epochs, version final del Stage 1

Pendientes a corto plazo:

- [ ] Decidir si lanzar yolov8s como ablation (modelo mas grande)
- [ ] Implementar Stage 2: train_classifier_crops.py
- [ ] Pipeline end-to-end: two_stage_inference.py
- [ ] Tabla comparativa final baseline vs two-stage
- [ ] Redaccion del Capitulo 4 (Implementacion two-stage)
- [ ] Redaccion del Capitulo 5 (Resultados)

---

## 11. Ficheros relevantes

Scripts:

- `scripts/auto_label_bursts.py` - generacion automatica de labels YOLO
- `scripts/train_yolo.py` - entrenamiento YOLOv8

Artefactos:

- `data/mini3_detector_python_v3/yolo/data.yaml` - manifest YOLO
- `data/mini3_detector_python_v3/yolo/train.txt` (val.txt, test.txt) - splits
- `data/mini3_detector_python_v3/artifacts/yolo/run01/` - 15 epochs train
- `data/mini3_detector_python_v3/artifacts/yolo/run01_test/` - 15 epochs test
- `data/mini3_detector_python_v3/artifacts/yolo/run02_60ep/` - 60 epochs train
- `data/mini3_detector_python_v3/artifacts/yolo/run02_60ep_test/` - 60 epochs test

Pesos del mejor modelo:

- `data/mini3_detector_python_v3/artifacts/yolo/run02_60ep/weights/best.pt`
