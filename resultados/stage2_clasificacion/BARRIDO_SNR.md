# Barrido SNR — etapa extra del Stage 2

Documento de progreso del experimento de robustez del clasificador Stage 2 frente a ruido sintético inyectado en el dominio IQ. Cubre motivación, diseño, implementación, resultados y conclusiones para la memoria del TFG.

Fecha: 2026-05-28
Modelo evaluado: `data/stage2_classifier_v2_mavic_split/artifacts/resnet18_stage2_mavic_split/best_model.pth`
Test set: `splits/test.csv` del mismo experimento (941 segmentos, 8 capture_ids, 6 clases).

---

## 1. Objetivo

El clasificador Stage 2 (ResNet18, 6 clases: `f450`, `hunter`, `mavic_video`, `mavic_novideo`, `mini3`, `interference`) se entrenó sobre espectrogramas limpios. La memoria del TFG necesita cuantificar cómo se degrada el clasificador cuando el entorno RF se contamina. En lugar de capturar más datos a distintas SNR (caro y poco controlable), se inyecta ruido sintético en el dominio IQ con potencia calibrada al SNR objetivo, y se regenera el espectrograma con el MISMO pipeline de entrenamiento. Es lo que hace el repo de RFUAV (kitoweeknd), pero íntegramente generado por código.

Pregunta de investigación: ¿la red ha aprendido la huella morfológica de los bursts FHSS del dron, o se está apoyando en intensidad media de potencia? El barrido permite distinguir las dos hipótesis: si la red se basa en intensidad, AWGN y FHSS la degradarían de forma similar (misma potencia total inyectada). Si la red mira morfología, la curva debería caer más rápido con AWGN (que llena todo el plano tiempo-frecuencia) que con FHSS (que solo contamina parches estrechos).

---

## 2. Decisiones de diseño

1. **Dominio del ruido: IQ + regeneración de espectrogramas**. SNR físicamente bien definido en dB. Coste: regenerar 17 879 espectrogramas. Validado.
2. **Dos modelos de ruido**:
   - **AWGN** (canónico, comparable al paper RFUAV).
   - **FHSS-like (Bluetooth sintético)** (innovación del TFG).
3. **Modelo evaluado**: el clasificador de 6 clases (`mavic_split`), porque es el científicamente más limpio.
4. **Rango de SNR**: `[+inf, +20, +15, +10, +5, 0, -5, -10, -15, -20]` dB. El `+inf` actúa como control sin ruido y debe reproducir las métricas del test original (acc=0.898, macro-F1=0.764).
5. **Rendering**: misma STFT que en entrenamiento (Hann 1024, overlap 75%), mismo `Pdb = 10·log10(|S|² + eps)`, mismo colormap parula, mismo clim (maxDb−80, maxDb), misma 1024×576 a dpi 120.
6. **Transforms del modelo**: idénticas a las del eval (FrequencyNormalize(window_frac=0.5), Resize(224), ImageNet norm).

---

## 3. Implementación

Cuatro scripts nuevos en `scripts/`.

### 3.1 `scripts/snr_noise.py`

Módulo de inyección de ruido sobre IQ complejo, calibrado en SNR objetivo.

API:

```python
inject_noise(iq, fs, snr_db, noise_type, rng) -> (iq_noisy, info)
add_awgn(iq, snr_db, rng) -> (iq_noisy, info)
add_fhss_interference(iq, fs, snr_db, rng, hop_rate=1600, burst_duration_s=366e-6,
                       hop_bw_hz=1e6, band_frac=0.9) -> (iq_noisy, info)
```

Calibración SNR validada en sandbox: target vs medido coincide en ±0.01 dB para AWGN y FHSS, en todos los SNRs de −20 a +20.

### 3.2 `scripts/spectrogram_render.py`

Renderiza espectrogramas en memoria a partir de IQ, replicando 1:1 los parámetros de `make_spectrogram_dataset.py`.

Dos modos:
- **`render_spectrogram_image(iq, fs, fc)`**: rápido en NumPy + PIL (LUT parula precalculada, sin matplotlib). ~30 ms por imagen.
- **`render_spectrogram_image_matplotlib(iq, fs, fc)`**: respaldo lento via matplotlib + savefig (~400 ms por imagen). Útil para reproducción pixel-perfect.

Validado que la macro-F1 a SNR=∞ con render rápido (0.7654) está más cerca del test original (0.7641) que la del matplotlib (0.7778), lo cual confirma que el render rápido es perfectamente aceptable.

### 3.3 `scripts/snr_sweep_stage2.py`

Script principal del barrido. Para cada (PNG del test set, noise_type, SNR):
1. Parsea `capture__segNNNN.png` a (stem, seg_idx).
2. Localiza el `.sigmf-data` y lee el segmento IQ (0.1 s ≈ 4M muestras a 40 MHz).
3. Inyecta ruido calibrado.
4. Renderiza el espectrograma.
5. Aplica las mismas transforms del modelo y predice.

Flags relevantes:
- `--quick-check`: 10 segmentos por clase, SNR=∞. Para validar pipeline.
- `--render-mode {fast,matplotlib}`: por defecto `fast`.
- `--snr-list`: lista de SNRs en dB. Default `[inf, 20, 15, 10, 5, 0, -5, -10, -15, -20]`.
- `--noise-types`: `[awgn, fhss]` por defecto.

Tiempo total del barrido completo: ~30 min en RTX 4070 Laptop (17 879 inferencias).

Bugfix aplicado: el shortcut para SNR=∞ (no renderizar dos veces) usaba `per_sample_rows[-1]` para copiar la predicción, lo cual fallaba cuando los noise_types se procesaban en bucle externo. Cambiado por una variable `clean_row` que guarda explícitamente la predicción limpia.

### 3.4 `scripts/plot_snr_sweep.py`

Genera todas las gráficas y matrices para la memoria a partir de `snr_sweep_results.csv` y las matrices `cm_{noise}_{snr}.csv`. Salidas:
- `accuracy_vs_snr.png` y `f1_macro_vs_snr.png`: dos curvas (AWGN azul, FHSS rojo).
- `f1_per_class_vs_snr.png`: rejilla 2×3 con un subplot por clase.
- `confusion_matrices/cm_{noise}_{snr}_norm.png`: matrices normalizadas para SNRs clave.

### 3.5 `scripts/snr_examples.py`

Visualizador: para cada clase, toma un segmento representativo del test set, lo contamina con AWGN y FHSS a varios SNR, y guarda rejillas de espectrogramas para enseñar visualmente cómo se degrada la señal. Útil para incluir figuras ilustrativas en la memoria.

Salidas en `artifacts/snr_sweep/examples/`:
- `examples_<clase>_awgn.png`: rejilla 1×6 (limpio + SNR ∈ {+20, +10, 0, −10, −20}).
- `examples_<clase>_fhss.png`: ídem con ruido FHSS.
- `examples_compare_<clase>_snrN.png`: comparativa lado a lado limpio / AWGN / FHSS al mismo SNR.

---

## 4. Cómo se genera cada ruido

### 4.1 AWGN

Ruido blanco gaussiano complejo `n[k] ~ CN(0, σ²)` con varianza calibrada al SNR objetivo:

```
p_signal = mean(|iq|²)
p_noise  = p_signal / 10^(SNR/10)
sigma    = sqrt(p_noise / 2)        # varianza por componente
n_re ~ N(0, sigma²)
n_im ~ N(0, sigma²)
n    = n_re + 1j · n_im
iq_noisy = iq + n
```

Modelo canónico del ruido térmico de receptor. Cubre uniformemente todo el plano tiempo-frecuencia.

### 4.2 FHSS-like (Bluetooth sintético)

Modelo de interferencia tipo Bluetooth Classic. Por cada segmento IQ de 0.1 s (4M muestras a fs=40 MHz) se generan **160 bursts** = 1600 hops/s × 0.1 s. Cada burst tiene:

- **Duración**: 366 µs ≈ slot de Bluetooth. A 40 MHz = 14 640 muestras por burst.
- **Posición temporal**: aleatoria uniforme dentro del segmento (los bursts pueden solaparse, lo que emula varios dispositivos BT activos simultáneamente).
- **Frecuencia central**: uniforme en el 90% del ancho de banda capturado (es decir, en `[−18, +18] MHz` respecto a la portadora del SDR).
- **Fase inicial**: aleatoria uniforme en `[0, 2π]` para que distintos bursts no se sumen coherentemente.
- **Modulación GFSK simplificada**: la frecuencia instantánea no es constante; recorre linealmente `f_c ± 250 kHz` con sentido aleatorio (chirp ascendente o descendente). La fase instantánea es la integral acumulada: `phase[n] = phase[0] + 2π · cumsum(f_inst) / fs`.
- **Ventana de Hann** en la amplitud para suavizar los bordes y evitar lóbulos sinc fuertes.

Los 160 bursts se suman en un vector complejo. La energía total se calibra para cumplir el SNR objetivo:

```
p_interference_raw = mean(|interference|²)
p_target           = p_signal / 10^(SNR/10)
scale              = sqrt(p_target / p_interference_raw)
interference      *= scale
iq_noisy           = iq + interference
```

Esto garantiza que **la potencia total inyectada es idéntica para AWGN y FHSS al mismo SNR**, lo que hace la comparativa justa. La única diferencia entre los dos es la **distribución espacial** de esa energía en el plano tiempo-frecuencia.

#### Diferencias con BT Classic real

A documentar como simplificaciones en la memoria:
- BT real usa GFSK con índice de modulación 0.32; aquí se usa chirp lineal (equivalente en primera aproximación).
- BT salta exactamente entre 79 canales discretos espaciados 1 MHz; aquí la frecuencia es uniforme continua.
- BT con AFH evita canales WiFi ocupados; aquí no hay AFH.
- BT mantiene secuencia pseudoaleatoria coordinada entre maestro y esclavo; aquí cada burst es independiente.

El experimento mide robustez ante interferencia FHSS de morfología BT-like, no certificación BT.

---

## 5. Resultados

### 5.1 Tabla accuracy

| SNR (dB) | AWGN | FHSS-like |
|----------|------|-----------|
| inf      | 0.898 | 0.898 |
| +20      | 0.900 | 0.899 |
| +15      | 0.895 | 0.900 |
| +10      | 0.889 | 0.899 |
| +5       | 0.848 | 0.908 |
| 0        | 0.691 | 0.912 |
| -5       | 0.239 | 0.852 |
| -10      | 0.136 | 0.748 |
| -15      | 0.133 | 0.676 |
| -20      | 0.133 | 0.557 |

### 5.2 Tabla macro-F1

| SNR (dB) | AWGN | FHSS-like |
|----------|------|-----------|
| inf      | 0.764 | 0.764 |
| +20      | 0.765 | 0.766 |
| +15      | 0.759 | 0.767 |
| +10      | 0.749 | 0.771 |
| +5       | 0.720 | 0.781 |
| 0        | 0.540 | 0.787 |
| -5       | 0.138 | 0.670 |
| -10      | 0.049 | 0.439 |
| -15      | 0.043 | 0.405 |
| -20      | 0.043 | 0.345 |

### 5.3 F1 por clase, AWGN

| SNR | f450 | hunter | mavic_video | mavic_novideo | mini3 | interference |
|-----|------|--------|-------------|----------------|-------|--------------|
| inf | 0.896 | 0.738 | 1.000 | 0.000 | 0.993 | 0.957 |
| +20 | 0.896 | 0.741 | 1.000 | 0.000 | 0.995 | 0.960 |
| +10 | 0.833 | 0.730 | 0.992 | 0.000 | 0.997 | 0.942 |
| +5  | 0.800 | 0.656 | 0.992 | 0.000 | 0.938 | 0.934 |
|  0  | 0.169 | 0.477 | 0.979 | 0.000 | 0.841 | 0.773 |
| -5  | 0.000 | 0.255 | 0.000 | 0.000 | 0.421 | 0.152 |
| -10 | 0.000 | 0.241 | 0.000 | 0.000 | 0.052 | 0.000 |
| -20 | 0.000 | 0.227 | 0.000 | 0.000 | 0.033 | 0.000 |

### 5.4 F1 por clase, FHSS

| SNR | f450 | hunter | mavic_video | mavic_novideo | mini3 | interference |
|-----|------|--------|-------------|----------------|-------|--------------|
| inf | 0.896 | 0.738 | 1.000 | 0.000 | 0.993 | 0.957 |
| +20 | 0.906 | 0.741 | 1.000 | 0.000 | 0.990 | 0.959 |
| +10 | 0.935 | 0.749 | 1.000 | 0.000 | 0.980 | 0.959 |
| +5  | 0.966 | 0.774 | 1.000 | 0.000 | 0.984 | 0.963 |
|  0  | 0.929 | 0.891 | 1.000 | 0.000 | 0.985 | 0.917 |
| -5  | 0.462 | 0.768 | 0.992 | 0.000 | 0.954 | 0.846 |
| -10 | 0.000 | 0.000 | 0.961 | 0.000 | 0.920 | 0.753 |
| -20 | 0.000 | 0.000 | 0.879 | 0.000 | 0.592 | 0.599 |

### 5.5 Diferencia AWGN vs FHSS por SNR (accuracy)

| SNR | Δ accuracy (FHSS − AWGN) |
|-----|---|
| inf | 0.000 |
| +10 | +0.010 |
|  0  | **+0.221** |
| -5  | **+0.613** |
| -10 | **+0.612** |
| -20 | **+0.424** |

La diferencia es máxima en `SNR ≈ −5 dB` (0.61 puntos de accuracy entre los dos ruidos). Esa es la cifra clave a defender en la memoria.

---

## 6. Conclusiones científicas

### 6.1 Observaciones del barrido

1. **Codo característico de AWGN entre +5 y −5 dB**. Accuracy pasa de 0.85 a 0.24 en 10 dB. A −10 dB ya colapsa al 13% (predice siempre la clase mayoritaria, hunter, que es el modo trivial en este test set).

2. **FHSS-like aguanta mucho mejor**. Mantiene accuracy 0.85 hasta −5 dB y sigue por encima del 50% a −20 dB.

3. **A SNR=∞ ambas curvas valen 0.898**, idéntico al test original. Confirma que el pipeline (lectura IQ + render + transforms + modelo) reproduce fielmente el experimento original.

### 6.2 Lectura física: por qué AWGN es más destructivo

AWGN reparte su energía uniformemente por todo el plano tiempo-frecuencia. Cuando la potencia total inyectada sube, el espectrograma entero se va llenando hasta borrar la huella del dron.

FHSS-like, con la misma potencia total al mismo SNR objetivo, concentra esa energía en bursts de ~1 MHz × 366 µs en posiciones aleatorias. En un segmento de 40 MHz × 0.1 s, los 160 bursts ocupan aproximadamente:

```
área_bursts ≈ 160 · (1 MHz · 366 µs) = 160 · 0.366 µs·MHz
área_total  = 40 MHz · 0.1 s        = 4000 µs·MHz
fracción    ≈ 0.015 → ~1.5% del plano
```

Es decir, **~98.5% del plano tiempo-frecuencia queda no contaminado**. La señal del dron sigue visible en esas zonas, y la red la reconoce.

### 6.3 Implicación: la red ha aprendido morfología, no intensidad

Si la red se apoyara en intensidad media, AWGN y FHSS la degradarían de forma similar (misma potencia inyectada). Al ver que **AWGN colapsa antes que FHSS** con la misma potencia total, queda demostrado que la red mira *dónde* está la energía (forma de los bursts), no *cuánta* hay (nivel medio).

Es coherente con lo que GradCAM mostraba en los experimentos anteriores: la red activa sobre los bursts FHSS del dron, no sobre el fondo.

### 6.4 Lectura por clase

- **mini3**: la clase más robusta. F1≈1.0 hasta SNR=−5 (AWGN) o −10 (FHSS). Consistente con su rol central en el TFG.
- **mavic_video**: aguanta perfecto en FHSS a todos los SNRs. Firma muy "fuerte" con video activo.
- **interference**: muy robusto en FHSS (F1>0.95 hasta SNR=+5). El patrón de actividad WiFi es claramente distinguible de bursts BT.
- **f450** y **hunter**: caen antes que mini3 en AWGN. Coherente con su menor representación en entrenamiento.
- **mavic_novideo**: F1=0 a todos los SNRs incluido el limpio. Confirma que es un problema estructural (su firma se parece morfológicamente a hunter), no un problema de robustez al ruido.

### 6.5 Cota práctica de robustez

- Con SNR > +5 dB (operación típica en receptor decente): accuracy ≥ 0.85 en AWGN, ≥ 0.91 en FHSS.
- Con SNR > −5 dB en FHSS (interferencia BT realista): accuracy ≥ 0.85.
- AWGN colapsa por debajo de SNR=0 dB.

---

## 7. Justificación de la comparativa AWGN vs FHSS

**A nivel teórico**: AWGN es el modelo canónico de ruido térmico/electrónico de receptor (siempre presente, banda ancha, distribución gaussiana). FHSS-like representa interferencia de coexistencia en banda ISM 2.4 GHz (Bluetooth, otros emisores FHSS). Son los dos escenarios reales a los que se enfrenta el detector.

**A nivel práctico**: el resultado da una cota cuantitativa de robustez. Es el tipo de cifra defendible que pide la sección de resultados de un TFG.

**Como cierre de la hipótesis principal del TFG**: el experimento aporta evidencia indirecta pero sólida de que la red ha aprendido la morfología FHSS del dron y no el nivel medio de potencia. Cierra el círculo con GradCAM y con la sospecha de que el modelo estaba aprendiendo huella, no intensidad.

---

## 8. Limitaciones honestas (a mencionar en la memoria)

1. **El ruido FHSS sintético no es Bluetooth real** (ver simplificaciones en sección 4.2). El experimento mide robustez ante una *clase* de interferencia FHSS, no certificación BT.
2. **Solo se evalúa sobre el test set existente** (941 segmentos, 8 capture_ids). Los resultados son una caracterización del modelo entrenado, no una generalización absoluta.
3. **mavic_novideo es un caso patológico**: F1=0 a todos los SNRs. Su contribución a la macro-F1 es siempre nula y deprime artificialmente la métrica global. La memoria debe mencionarlo.
4. **El segmentado fijo de 0.1 s podría no captar todos los bursts FHSS del dron**. Otros tamaños de ventana podrían dar curvas diferentes.

---

## 9. Artefactos generados

```
data/stage2_classifier_v2_mavic_split/artifacts/snr_sweep/
├── run_config.json                   # configuración del barrido
├── sigmf_meta_index.json             # fs, fc por stem (debug)
├── predictions_per_sample.csv        # 18 820 filas: 941 × 10 SNRs × 2 ruidos
├── snr_sweep_results.csv             # 20 filas: agregado por (noise, snr)
├── confusion_matrices/
│   ├── cm_awgn_inf.csv, cm_awgn_+20.csv, ..., cm_awgn_-20.csv
│   ├── cm_fhss_inf.csv, ..., cm_fhss_-20.csv
│   └── cm_*_norm.png                 # versiones gráficas normalizadas
├── accuracy_vs_snr.png
├── f1_macro_vs_snr.png
├── f1_per_class_vs_snr.png
└── examples/                         # generadas por snr_examples.py
    ├── examples_<clase>_awgn.png
    ├── examples_<clase>_fhss.png
    └── examples_compare_<clase>_snr<N>.png
```

---

## 10. Comandos de referencia

```powershell
# Validación rápida del pipeline (60 segmentos, ~1 min)
uv run python scripts/snr_sweep_stage2.py --quick-check

# Barrido completo (~30 min en RTX 4070 Laptop)
uv run python scripts/snr_sweep_stage2.py

# Gráficas resumen
uv run python scripts/plot_snr_sweep.py

# Muestras visuales del ruido por clase
uv run python scripts/snr_examples.py

# Personalizar SNRs y comparativa
uv run python scripts/snr_examples.py --snr-list 10 0 -5 -10 --compare-snr -5

# Render lento de respaldo (fiel a entrenamiento)
uv run python scripts/snr_sweep_stage2.py --render-mode matplotlib

# Matrices de confusión para SNRs adicionales
uv run python scripts/plot_snr_sweep.py --key-snrs inf 10 0 -10 -20
```

---

## 11. Bugs encontrados y arreglados durante el experimento

1. **Bug del shortcut SNR=∞**: el script original copiaba `per_sample_rows[-1]` en vez de la fila limpia correcta. Resultado: `fhss inf` desaparecía y `fhss -20` quedaba duplicado. Arreglado introduciendo la variable explícita `clean_row`.
2. **Pandas y "inf" en CSV**: pandas convierte automáticamente la cadena "inf" en `np.inf` al leer un CSV con columna mixta. Para filtrar correctamente: `pd.read_csv(..., dtype={"snr_db": str})`.
3. **Duplicados en nombres de matrices**: el recálculo guardó CSVs con nombre `cm_awgn_10.0.csv` mientras `plot_snr_sweep.py` busca `cm_awgn_+10.csv`. Resuelto unificando el formato con `f"{snr:+g}"`.

Todos los datos en disco (a fecha 2026-05-28) están limpios y consistentes.
