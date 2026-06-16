# Stage 2 — Clasificación de modelos de UAV con ResNet18

Scripts para entrenar y evaluar el clasificador de 6 clases (Stage 2 del pipeline),
incluyendo el barrido de robustez a ruido y las visualizaciones de interpretabilidad.

| Script | Descripción |
|--------|-------------|
| `train_stage2_classifier_mavic_split.py` | Entrena ResNet18 sobre el dataset RFUAV (6 clases). Split mavic_split. FrequencyNormalize activo. |
| `evaluate_rfuav.py` | Evalúa el clasificador sobre el split de validación RFUAV. Genera matriz de confusión y métricas por clase. |
| `snr_sweep_stage2.py` | Barrido SNR completo (AWGN y FHSS-like). Lee segmentos IQ, inyecta ruido y regenera el espectrograma. |
| `plot_snr_sweep.py` | Genera figuras de accuracy vs SNR y F1 por clase para la memoria. |
| `snr_examples.py` | Rejillas visuales de cómo degrada cada tipo de ruido el espectrograma a distintos SNR. |
| `interpret_stage2.py` | Embeddings t-SNE/PCA de la penúltima capa y GradCAM/CAM discriminativo por clase. |
| `gradcam.py` | GradCAM sobre el detector binario ResNet18 (baseline histórico). |
| `figura_confusion_mavic_hunter.py` | Figura comparativa de morfología Mavic vs Hunter para la memoria. |

**Dependencias:** estos scripts importan de `scripts/common.py`, `scripts/snr_noise.py`
y `scripts/spectrogram_render.py` (utilidades compartidas en el directorio padre).

```powershell
# 1. Entrenar clasificador 6 clases (~ 20 min en RTX 4070)
uv run python scripts/stage2_clasificacion/train_stage2_classifier_mavic_split.py

# 2. Barrido SNR (~ 30 min)
uv run python scripts/stage2_clasificacion/snr_sweep_stage2.py

# 3. Figuras del barrido SNR
uv run python scripts/stage2_clasificacion/plot_snr_sweep.py

# 4. Interpretabilidad (t-SNE + GradCAM)
uv run python scripts/stage2_clasificacion/interpret_stage2.py

# Validación rápida del barrido SNR
uv run python scripts/stage2_clasificacion/snr_sweep_stage2.py --quick-check
```

El modelo entrenado está en:
`data/stage2_classifier_v2_mavic_split/artifacts/resnet18_stage2_mavic_split/best_model.pth`
