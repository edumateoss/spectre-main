# Stage 1 — Detección de ráfagas FHSS con YOLOv8

Scripts para etiquetar, entrenar y evaluar el detector de ráfagas FHSS (Stage 1 del pipeline).

| Script | Descripción |
|--------|-------------|
| `auto_label_bursts.py` | Auto-etiquetado YOLO por umbral de energía (percentil 99,8). Genera ficheros `.txt` de bounding boxes. |
| `train_yolo.py` | Entrena YOLOv8n sobre el dataset v3. Modelo base: `yolov8n.pt`. 60 épocas, batch=16. |
| `presence_analysis.py` | Sweep bidimensional (conf × K) sobre el test set. Calcula Precision, Recall, F1 y Specificity a nivel de imagen. |
| `plot_yolo_results_clean.py` | Genera figuras de curvas de entrenamiento y métricas YOLO para la memoria. |
| `yolo_explainability.py` | EigenCAM sobre el detector YOLO: muestra qué zona del espectrograma activa el modelo. |

```powershell
# 1. Auto-etiquetar ráfagas (necesario antes de entrenar)
uv run python scripts/stage1_deteccion/auto_label_bursts.py

# 2. Entrenar YOLOv8 (~ 30 min en RTX 4070)
uv run python scripts/stage1_deteccion/train_yolo.py

# 3. Análisis de presencia (sweep conf × K)
uv run python scripts/stage1_deteccion/presence_analysis.py

# 4. Figuras para la memoria
uv run python scripts/stage1_deteccion/plot_yolo_results_clean.py
```

El mejor modelo entrenado está en:
`data/mini3_detector_python_v3/artifacts/yolo/run02_60ep/weights/best.pt`
