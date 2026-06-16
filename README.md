# Spectre — Detección e identificación de drones mediante señales de radiofrecuencia

**TFG de Eduardo Mateos Ruiz** — Grado en Ingeniería de Telecomunicaciones

Pipeline de dos etapas para detectar e identificar UAVs a partir de espectrogramas
tiempo-frecuencia de la banda de 2,4 GHz, capturados con un SDR USRP B210.

---

## Estructura del repositorio

```
spectre-main/
├── MEMORIA_v2.pdf                   # Memoria del TFG (versión final entregada)
├── memoria_yolo/                    # Fuente LaTeX de la memoria (compilar en Overleaf)
│
├── scripts/                         # Código fuente organizado por etapa
│   ├── common.py                    # Utilidades compartidas (FrequencyNormalize, etc.)
│   ├── snr_noise.py                 # Inyección de ruido AWGN y FHSS sintético
│   ├── spectrogram_render.py        # Renderizado de espectrogramas en memoria
│   │
│   ├── stage0_captura/              # Captura y preprocesado de datos
│   ├── stage1_deteccion/            # Stage 1: detección de ráfagas con YOLOv8
│   ├── stage2_clasificacion/        # Stage 2: clasificación de modelos con ResNet18
│   ├── experimento_zeroshot/        # Experimento de generalización multi-dron
│   └── legacy/                      # Scripts históricos del baseline ResNet18 binario
│
├── resultados/                      # Resultados consolidados de todos los experimentos
│   ├── stage1_deteccion/
│   │   ├── metricas/                # JSON y CSV con métricas de YOLO y análisis de presencia
│   │   ├── figuras/                 # Curvas PR, matrices de confusión, análisis de presencia
│   │   ├── YOLO_PROGRESO.md         # Resumen de los entrenamientos YOLO
│   │   └── ANALISIS_PRESENCIA_DRON.md
│   ├── stage2_clasificacion/
│   │   ├── metricas/                # JSON y CSV con métricas de ResNet18 y barrido SNR
│   │   ├── figuras/                 # Matrices de confusión, curvas SNR, GradCAM, t-SNE
│   │   └── BARRIDO_SNR.md           # Análisis completo del barrido SNR
│   ├── experimento_zeroshot/
│   │   ├── metricas/                # zeroshot_summary.json, zeroshot_report.csv
│   │   └── figuras/                 # Gráficas de recall por tipo de dron
│   └── modelos/
│       └── yolo_stage1_best.pt      # Pesos del mejor modelo YOLO (run02_60ep, 60 épocas)
│
├── data/                            # Datasets y artefactos de entrenamiento
│   ├── mini3_detector_python_v3/    # Dataset v3 (Stage 1): 3150 espectrogramas DJI Mini 3
│   ├── stage2_classifier_v1/        # Dataset campo: F450, Mavic Pro, Hunter (experimento zero-shot)
│   └── stage2_classifier_v2_mavic_split/  # Dataset RFUAV (Stage 2): 6 clases
│
├── data_raw/sigmf/                  # Capturas IQ brutas en formato SigMF
├── notebooks/                       # Notebook de referencia RFUAV (ResNet18 5 clases)
└── pyproject.toml                   # Dependencias del proyecto (uv)
```

---

## Pipeline

### Stage 1 — Detección de ráfagas FHSS con YOLOv8

Dado un espectrograma de 0,1 s (1024×576 px, colormap parula), localiza y clasifica
las regiones activas como `drone` o `interference`.

**Resultado principal:** mAP@50 = 0,575 para la clase drone. Cero confusión cruzada
entre drone e interference. Precisión y especificidad = 1,000 en el análisis de
presencia a nivel de imagen (16 configuraciones evaluadas).

Punto operativo recomendado: `conf=0,10, K=3` → Recall=91%, F1=0,953.

### Stage 2 — Clasificación de modelos de UAV con ResNet18

Dado un espectrograma etiquetado como drone, identifica el modelo entre 6 clases
del dataset RFUAV: f450, hunter, mavic_video, mavic_novideo, mini3, interference.

**Resultado principal:** accuracy=0,898, macro-F1=0,764 en test limpio.
Robustez a ruido: FHSS-like mantiene accuracy=0,852 a −5 dB vs AWGN que cae a 0,239.

### Experimento zero-shot de generalización multi-dron

El modelo YOLO entrenado solo sobre DJI Mini 3 (OcuSync v2) detecta sin reentrenamiento:
- F450 (OcuSync v1): recall = 100%
- Mavic Pro (OcuSync v1): recall = 94,2%
- Hunter (DSSS, fuera de dominio): recall = 0,5% — el detector detecta FHSS, no DSSS.

---

## Instalación y ejecución

```powershell
# Requiere Python 3.12 y uv (https://docs.astral.sh/uv/)
cd C:\TFG\spectre-main\spectre-main
uv sync

# Stage 1: entrenar YOLOv8
uv run python scripts/stage1_deteccion/train_yolo.py

# Stage 1: análisis de presencia
uv run python scripts/stage1_deteccion/presence_analysis.py

# Stage 2: entrenar ResNet18
uv run python scripts/stage2_clasificacion/train_stage2_classifier_mavic_split.py

# Stage 2: barrido SNR
uv run python scripts/stage2_clasificacion/snr_sweep_stage2.py

# Experimento zero-shot
uv run python scripts/experimento_zeroshot/infer_zeroshot_campo.py
```

---

## Modelos entrenados

| Modelo | Ruta | Tamaño |
|--------|------|--------|
| YOLOv8n Stage 1 (mejor, 60 épocas) | `data/mini3_detector_python_v3/artifacts/yolo/run02_60ep/weights/best.pt` | 6 MB |
| YOLOv8n Stage 1 (copia en resultados) | `resultados/modelos/yolo_stage1_best.pt` | 6 MB |
| ResNet18 Stage 2 (6 clases) | `data/stage2_classifier_v2_mavic_split/artifacts/resnet18_stage2_mavic_split/best_model.pth` | 128 MB |

---

## Hardware

- Receptor SDR: USRP B210, 40 MHz de tasa de muestreo, FC=2450 MHz
- GPU de entrenamiento: NVIDIA RTX 4070 Laptop (8 GB VRAM)
- UAV capturado: DJI Mini 3 (protocolo OcuSync v2)
