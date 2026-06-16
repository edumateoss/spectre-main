# Stage 0 — Captura y preprocesado de datos

Scripts para generar el dataset de espectrogramas a partir de capturas IQ en formato SigMF.

| Script | Descripción |
|--------|-------------|
| `make_spectrogram_dataset.py` | Convierte ficheros SigMF a PNGs de espectrograma (STFT, colormap parula, 1024×576 px). **No modificar:** produce el dataset canónico. |
| `split_capture_level.py` | Genera los splits train/val/test estratificados por `capture_id`. Seed=42. |
| `build_stage2_csv.py` | Construye el CSV maestro del dataset Stage 2 (RFUAV + campo). |

```powershell
# Generar espectrogramas desde SigMF
uv run python scripts/stage0_captura/make_spectrogram_dataset.py

# Crear splits train/val/test
uv run python scripts/stage0_captura/split_capture_level.py
```
