"""
make_spectrograms_campo.py
==========================

Genera espectrogramas PNG a partir de las capturas SigMF del dia de campo
outdoor (experimento de generalizacion multidron, dataset_campo_v1).

Capturas incluidas:
  F450   (Samuel)  : data_raw/sigmf/samuel/f450_1..8
  Mavic  (Alvaro)  : data_raw/sigmf/mavic_pro/mavic_10_captura1..5
  Hunter           : data_raw/sigmf/hunter/hunter_1..6
  Interferencia    : data_raw/sigmf/legacy/mini3_nodrone_*(6 capturas v3)

La clase de cada espectrograma se hereda de la captura:
  - F450, Mavic, Hunter -> label = "drone"  (deteccion binaria, coherente con v3)
  - Capturas nodrone    -> label = "interference"
  La columna drone_type del CSV registra el modelo concreto para analisis.

Salida:
  data/dataset_campo_v1/all/{label}/{capture_id}__seg{k:04d}.png
  data/dataset_campo_v1/metadata/spectrograms_campo.csv

Uso:
  # Modo rapido: 30 segs por captura (~750 imagenes, para zero-shot)
  uv run python scripts/make_spectrograms_campo.py --quick

  # Solo drones (sin interferencia), 50 segs por captura
  uv run python scripts/make_spectrograms_campo.py --max-segs 50 --skip-interference

  # Dataset completo (~2400 imagenes)
  uv run python scripts/make_spectrograms_campo.py
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import spectrogram as scipy_spectrogram
from scipy.signal import windows


# ---------------------------------------------------------------------------
# Rutas por defecto
# ---------------------------------------------------------------------------
ROOT_DATASET = Path("data/dataset_campo_v1")
ROOT_RAWDATA = Path("data_raw/sigmf")

# ---------------------------------------------------------------------------
# Parametros STFT identicos a make_spectrogram_dataset.py
# ---------------------------------------------------------------------------
FS_HZ = 40e6
DURACION_SEG = 0.1          # segundos por espectrograma
NFFT = 1024
WIN = windows.hann(NFFT, sym=False)
OVERLAP = round(0.75 * NFFT)
IMG_W_PX = 1024
IMG_H_PX = 576
EXPORT_DPI = 120
RANGO_DIN_DB = 80

# ---------------------------------------------------------------------------
# Configuracion de capturas del dia de campo
# Formato: (sigmf_subdir, capture_name, drone_type, fc_hz_nominal)
# fc_hz_nominal se sobreescribe con el valor real del .sigmf-meta si existe.
# ---------------------------------------------------------------------------
CAPTURES_DRONE: list[tuple[str, str, str, float]] = [
    # F450 (Samuel) - 8 capturas outdoor, FC=2480 MHz
    ("samuel", "f450_1", "f450", 2480e6),
    ("samuel", "f450_2", "f450", 2480e6),
    ("samuel", "f450_3", "f450", 2480e6),
    ("samuel", "f450_4", "f450", 2480e6),
    ("samuel", "f450_5", "f450", 2480e6),
    ("samuel", "f450_6", "f450", 2480e6),
    ("samuel", "f450_7", "f450", 2480e6),
    ("samuel", "f450_8", "f450", 2480e6),
    # Mavic Pro (Alvaro) - 5 capturas outdoor, FC=2430 MHz
    ("mavic_pro", "mavic_10_captura1", "mavic", 2430e6),
    ("mavic_pro", "mavic_10_captura2", "mavic", 2430e6),
    ("mavic_pro", "mavic_10_captura3", "mavic", 2430e6),
    ("mavic_pro", "mavic_10_captura4", "mavic", 2430e6),
    ("mavic_pro", "mavic_10_captura5", "mavic", 2430e6),
    # Hunter - 6 capturas limpias (sin conwifi), FC=2470 MHz
    ("hunter", "hunter_1", "hunter", 2470e6),
    ("hunter", "hunter_2", "hunter", 2470e6),
    ("hunter", "hunter_3", "hunter", 2470e6),
    ("hunter", "hunter_4", "hunter", 2470e6),
    ("hunter", "hunter_5", "hunter", 2470e6),
    ("hunter", "hunter_6", "hunter", 2470e6),
]

# Capturas de interferencia: nodrone del dataset v3 (indoor, WiFi+BT)
# Se usan como clase negativa; 2 capturas por nivel de WiFi (1/2/3)
CAPTURES_INTERFERENCE: list[tuple[str, str, str, float]] = [
    ("legacy", "mini3_nodrone_w1_s01_cap01", "interference", 2450e6),
    ("legacy", "mini3_nodrone_w1_s01_cap02", "interference", 2450e6),
    ("legacy", "mini3_nodrone_w2_s01_cap01", "interference", 2450e6),
    ("legacy", "mini3_nodrone_w2_s01_cap02", "interference", 2450e6),
    ("legacy", "mini3_nodrone_w3_s01_cap01", "interference", 2450e6),
    ("legacy", "mini3_nodrone_w3_s01_cap02", "interference", 2450e6),
]


# ---------------------------------------------------------------------------
# Colormap parula (identico a make_spectrogram_dataset.py)
# ---------------------------------------------------------------------------
def get_parula_cmap() -> LinearSegmentedColormap:
    colors = [
        (0.2422, 0.1504, 0.6603),
        (0.2504, 0.1650, 0.7076),
        (0.2578, 0.1818, 0.7511),
        (0.2647, 0.1978, 0.7952),
        (0.2706, 0.2147, 0.8364),
        (0.2751, 0.2342, 0.8710),
        (0.2783, 0.2559, 0.8991),
        (0.2803, 0.2782, 0.9221),
        (0.2813, 0.3006, 0.9414),
        (0.2809, 0.3228, 0.9579),
        (0.2784, 0.3447, 0.9717),
        (0.2699, 0.4000, 0.9892),
        (0.2394, 0.5177, 0.9692),
        (0.1890, 0.6200, 0.9180),
        (0.1570, 0.7000, 0.8310),
        (0.1960, 0.7600, 0.6870),
        (0.3650, 0.8000, 0.4940),
        (0.6000, 0.8400, 0.2720),
        (0.8200, 0.8900, 0.1400),
        (0.9769, 0.9839, 0.0805),
    ]
    return LinearSegmentedColormap.from_list("parula_custom", colors, N=256)


PARULA = get_parula_cmap()


# ---------------------------------------------------------------------------
# Lectura de segmento IQ (identico a make_spectrogram_dataset.py)
# ---------------------------------------------------------------------------
def leer_segmento_iq(path: Path, start_iq: int, num_iq: int) -> np.ndarray | None:
    offset_bytes = start_iq * 2 * 4
    count_float32 = num_iq * 2
    raw = np.fromfile(path, dtype=np.float32, count=count_float32, offset=offset_bytes)
    if raw.size < count_float32:
        return None
    x = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
    return x - np.mean(x)


# ---------------------------------------------------------------------------
# Generacion de un espectrograma PNG
# ---------------------------------------------------------------------------
def generar_png(x: np.ndarray, fs: float, fc: float, ruta_png: Path) -> None:
    F, T, S = scipy_spectrogram(
        x,
        fs=fs,
        window=WIN,
        nperseg=NFFT,
        noverlap=OVERLAP,
        nfft=NFFT,
        detrend=False,
        return_onesided=False,
        scaling="density",
        mode="complex",
    )
    F = np.fft.fftshift(F)
    S = np.fft.fftshift(S, axes=0)
    Pdb = 10.0 * np.log10(np.abs(S) ** 2 + np.finfo(float).eps)
    F_abs = (F + fc) / 1e6

    t0 = float(T[0]) if len(T) > 0 else 0.0
    t1 = float(T[-1]) if len(T) > 0 else DURACION_SEG

    fig = plt.figure(
        figsize=(IMG_W_PX / EXPORT_DPI, IMG_H_PX / EXPORT_DPI),
        dpi=EXPORT_DPI,
    )
    ax = fig.add_axes([0, 0, 1, 1])
    im = ax.imshow(
        Pdb,
        aspect="auto",
        origin="lower",
        extent=[t0, t1, float(F_abs[0]), float(F_abs[-1])],
        cmap=PARULA,
        interpolation="nearest",
    )
    ax.set_ylim((fc - fs / 2) / 1e6, (fc + fs / 2) / 1e6)
    ax.axis("off")

    max_db = float(np.max(Pdb))
    im.set_clim(max_db - RANGO_DIN_DB, max_db)

    fig.savefig(ruta_png, dpi=EXPORT_DPI, pad_inches=0)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Procesado de una captura completa
# ---------------------------------------------------------------------------
def procesar_captura(
    rawdata_dir: Path,
    output_dir: Path,
    sigmf_subdir: str,
    capture_name: str,
    drone_type: str,
    fc_nominal: float,
    label: str,
    max_segs: int | None,
) -> list[dict]:
    data_path = rawdata_dir / sigmf_subdir / f"{capture_name}.sigmf-data"
    meta_path = rawdata_dir / sigmf_subdir / f"{capture_name}.sigmf-meta"

    if not data_path.exists():
        print(f"  [AVISO] No existe {data_path}, se omite.")
        return []

    # Leer FC real del meta (sobreescribe el nominal si difiere)
    fc = fc_nominal
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        captures_list = meta.get("captures", [])
        if captures_list:
            fc_meta = captures_list[0].get("core:frequency")
            if fc_meta is not None:
                fc = float(fc_meta)

    muestras_seg = round(DURACION_SEG * FS_HZ)
    num_float32 = data_path.stat().st_size // 4
    num_muestras = num_float32 // 2
    num_segs_total = num_muestras // muestras_seg
    num_segs = num_segs_total if max_segs is None else min(num_segs_total, max_segs)

    img_dir = output_dir / "all" / label
    img_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for k in range(num_segs):
        x = leer_segmento_iq(data_path, k * muestras_seg, muestras_seg)
        if x is None:
            print(f"  [AVISO] Segmento {k:04d} incompleto en {capture_name}, se omite.")
            continue

        nombre_png = f"{capture_name}__seg{k:04d}.png"
        ruta_png = img_dir / nombre_png
        generar_png(x, FS_HZ, fc, ruta_png)

        del x
        gc.collect()

        rows.append({
            "filename": f"all/{label}/{nombre_png}",
            "label": label,
            "drone_type": drone_type,
            "capture_id": capture_name,
            "fc": fc,
            "fs": FS_HZ,
            "environment": "outdoor" if drone_type != "interference" else "indoor",
        })

    return rows


# ---------------------------------------------------------------------------
# Argumentos de linea de comandos
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--max-segs",
        type=int,
        default=None,
        help=(
            "Maximo de segmentos a generar por captura. "
            "None = todos. Util para limitar el tiempo de proceso."
        ),
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Alias para --max-segs 30. Genera ~750 imagenes para el zero-shot.",
    )
    p.add_argument(
        "--skip-interference",
        action="store_true",
        help="Omitir capturas de interferencia (solo genera espectrogramas de dron).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DATASET,
        help="Directorio raiz del dataset de salida.",
    )
    p.add_argument(
        "--rawdata-dir",
        type=Path,
        default=ROOT_RAWDATA,
        help="Directorio raiz con las carpetas SigMF (samuel/, mavic_pro/, etc.).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    max_segs: int | None = args.max_segs
    if args.quick and max_segs is None:
        max_segs = 30

    output_dir = args.output_dir.resolve()
    rawdata_dir = args.rawdata_dir.resolve()
    csv_path = output_dir / "metadata" / "spectrograms_campo.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Output dir:  {output_dir}")
    print(f"Rawdata dir: {rawdata_dir}")
    print(f"Max segs:    {max_segs if max_segs is not None else 'todos'}")
    print()

    capturas = list(CAPTURES_DRONE)
    if not args.skip_interference:
        capturas += list(CAPTURES_INTERFERENCE)

    all_rows: list[dict] = []
    for sigmf_subdir, capture_name, drone_type, fc_nominal in capturas:
        label = "drone" if drone_type != "interference" else "interference"
        print(f"Procesando {capture_name} ({drone_type}, FC={fc_nominal/1e6:.0f} MHz)...")
        rows = procesar_captura(
            rawdata_dir,
            output_dir,
            sigmf_subdir,
            capture_name,
            drone_type,
            fc_nominal,
            label,
            max_segs,
        )
        all_rows.extend(rows)
        print(f"  -> {len(rows)} espectrogramas generados")

    df = pd.DataFrame(all_rows)

    # Si ya existe CSV previo, concatenar y deduplicar por filename
    if csv_path.exists():
        df_prev = pd.read_csv(csv_path)
        df = pd.concat([df_prev, df], ignore_index=True).drop_duplicates(
            subset="filename", keep="last"
        )

    df.to_csv(csv_path, index=False)

    print()
    print(f"Total espectrogramas en CSV: {len(df)}")
    resumen = df.groupby("drone_type").size().rename("n_imagenes")
    print(resumen.to_string())
    print(f"\nCSV guardado en: {csv_path}")
    print("\nSiguiente paso:")
    print("  uv run python scripts/infer_zeroshot_campo.py")


if __name__ == "__main__":
    main()
