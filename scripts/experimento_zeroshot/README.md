# Experimento de generalización zero-shot multi-dron

Scripts para el experimento que prueba si el detector YOLO (entrenado solo sobre
DJI Mini 3) generaliza a drones nunca vistos durante el entrenamiento.

| Script | Descripción |
|--------|-------------|
| `make_spectrograms_campo.py` | Genera espectrogramas PNG desde las capturas SigMF del día de campo (F450, Mavic Pro, Hunter). Solo necesario si no existen ya en `data/stage2_classifier_v1/all/`. |
| `infer_zeroshot_campo.py` | Inferencia zero-shot: aplica `run02_60ep/weights/best.pt` sobre los espectrogramas del campo. Genera `zeroshot_report.csv` y ejemplos visuales con cajas. |
| `analizar_zeroshot_detalle.py` | Análisis detallado por captura: recall por captura individual, distribución de cajas, etc. |
| `generar_galeria_campo.py` | Galería categorizada de imágenes según nivel de detección (alta/media/nula). Reproducible con `random.Random(42)`. |
| `generar_figuras_campo.py` | Figuras de publicación para el Cap. 5 de la memoria: barras de recall, grid comparativo, violinplot. |
| `generar_figuras_cap5.py` | Figuras adicionales para el Capítulo 5. |
| `generar_figuras_cap5_extra.py` | Figuras extra de análisis del experimento zero-shot. |

```powershell
# 1. Inferencia zero-shot (conf=0.10, K=1 por defecto)
uv run python scripts/experimento_zeroshot/infer_zeroshot_campo.py

# Con umbral personalizado
uv run python scripts/experimento_zeroshot/infer_zeroshot_campo.py --conf 0.25 --k 3

# 2. Galería visual categorizada
uv run python scripts/experimento_zeroshot/generar_galeria_campo.py

# 3. Figuras para la memoria
uv run python scripts/experimento_zeroshot/generar_figuras_campo.py
```

Resultados en:
`data/stage2_classifier_v1/artifacts/zeroshot/`
