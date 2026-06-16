import gc
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import spectrogram, windows


# ============================================================
# CONFIGURACION DE LA CAPTURA
# Aqui es donde cambias de archivo facilmente
# ============================================================

carpeta_entrada = Path("data_raw/sigmf/mavic_pro")

# Cambia SOLO esto para cambiar de captura
nombre_base = "mavic_10_captura1"

# Metadatos experimentales de esta captura
# ------------------------------------------------------------
# Variables a CAMBIAR por captura: label, wifi_level, distance_m, condition
# Variables FIJAS en toda la campaña v3: antenna, environment, gain, angle_deg
# ------------------------------------------------------------
capture_id  = nombre_base
label       = "mavic"     # "drone" o "interference"
wifi_level  = 1                # 1, 2 o 3 (nivel de saturacion WiFi)
distance_m  = 8               # metros (5, 10, 20, 40...) para drone; None para interference
condition   = "clean"            # "clean" para train/val/test limpio; "mixed" para test final
platform    = "solo"             # "ocusync" (drone), "wifi" (L1), "wifi+bt" (L2/L3)
antenna     = "directional"
angle_deg   = 0
environment = "outdoor"
gain        = 10

# Archivos SigMF derivados automaticamente
nombre_meta = carpeta_entrada / f"{nombre_base}.sigmf-meta"
nombre_datos = carpeta_entrada / f"{nombre_base}.sigmf-data"


# ============================================================
# CONFIGURACION DEL DATASET DE SALIDA
# ============================================================

dataset_root = Path("data/stage2_classifier_v1")

# Estructura compatible con Python:
# data/stage2_classifier_v1/all/drone
# data/stage2_classifier_v1/all/interference
carpeta_salida = dataset_root / "all" / label
metadata_dir = dataset_root / "metadata"
csv_salida = metadata_dir / "spectrograms_stage2.csv"

carpeta_salida.mkdir(parents=True, exist_ok=True)
metadata_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# PARAMETROS DE SEGMENTACION Y STFT
# ============================================================

duracion_segmento = 0.1  # segundos

NFFT = 1024

# Equivalente a MATLAB:
# win = hann(NFFT, 'periodic');
win = windows.hann(NFFT, sym=False)

overlap = round(0.75 * NFFT)
nfft = NFFT

# Escala de color
# "auto" = maxDb-80 a maxDb por cada imagen
# "fija" = usa caxis_fija_db para todas las imagenes
modo_escala = "auto"

rango_dinamico_db = 80
caxis_fija_db = (-120, -40)

# Tamaño visual de la figura exportada
# Antes: 1600 x 900, Resolution 150.
# Ahora bajamos algo para evitar consumo excesivo de RAM.
img_width_px = 1024
img_height_px = 576
export_resolution = 120


# ============================================================
# COLORMAP PARULA APROXIMADA
# ============================================================

def get_parula_cmap():
    """
    Aproximacion razonable a la paleta parula de MATLAB.
    No es pixel-perfect, pero visualmente se parece bastante.
    """
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


parula_cmap = get_parula_cmap()


# ============================================================
# FUNCION PARA LEER SOLO UN SEGMENTO IQ
# ============================================================

def leer_segmento_iq_float32_interleaved(path: Path, start_iq: int, num_iq: int):
    """
    Lee solo un segmento de un .sigmf-data con formato:
    I0, Q0, I1, Q1, ... en float32.

    Esto evita cargar toda la captura en memoria.
    """
    offset_bytes = start_iq * 2 * 4
    count_float32 = num_iq * 2

    raw = np.fromfile(
        path,
        dtype=np.float32,
        count=count_float32,
        offset=offset_bytes,
    )

    if raw.size < count_float32:
        return None

    i = raw[0::2]
    q = raw[1::2]

    x = i.astype(np.float32) + 1j * q.astype(np.float32)

    # Quitar componente DC solo del segmento actual
    x = x - np.mean(x)

    return x


# ============================================================
# 1. LEER METADATOS
# ============================================================

if not nombre_meta.exists():
    raise FileNotFoundError(f"No existe el archivo meta: {nombre_meta}")

with open(nombre_meta, "r", encoding="utf-8") as f:
    meta_info = json.load(f)

global_info = meta_info.get("global", {})
captures = meta_info.get("captures", [])

fs = global_info.get("core:sample_rate", global_info.get("core_sample_rate"))

if len(captures) == 0:
    raise ValueError("El .sigmf-meta no contiene campo captures")

fc = captures[0].get("core:frequency", captures[0].get("core_frequency"))

if fs is None or fc is None:
    raise ValueError("No se pudieron leer fs o fc del .sigmf-meta")

fs = float(fs)
fc = float(fc)

print("Metadatos cargados:")
print(f"  Archivo meta = {nombre_meta}")
print(f"  Frecuencia central = {fc / 1e6:.2f} MHz")
print(f"  Sample rate = {fs / 1e6:.2f} MHz")


# ============================================================
# 2. PREPARAR LECTURA RAW IQ SIN CARGAR TODO EN MEMORIA
# ============================================================

if not nombre_datos.exists():
    raise FileNotFoundError(f"No existe el archivo data: {nombre_datos}")

# Cada muestra IQ tiene 2 float32: I y Q.
# Cada float32 ocupa 4 bytes.
num_float32 = nombre_datos.stat().st_size // 4
num_muestras_iq = num_float32 // 2

duracion_total = num_muestras_iq / fs

print("\nDatos IQ detectados:")
print(f"  Archivo data = {nombre_datos}")
print(f"  Duracion total = {duracion_total:.3f} s")
print(f"  Numero total de muestras IQ = {num_muestras_iq}")


# ============================================================
# 3. DIVIDIR EN SEGMENTOS
# ============================================================

muestras_segmento = round(duracion_segmento * fs)
num_segmentos = num_muestras_iq // muestras_segmento

print("\nSegmentacion:")
print(f"  Duracion segmento = {duracion_segmento:.3f} s")
print(f"  Muestras por segmento = {muestras_segmento}")
print(f"  Segmentos completos = {num_segmentos}")


# ============================================================
# 4. PARAMETROS STFT MOSTRADOS POR PANTALLA
# ============================================================

df = fs / NFFT
dt = NFFT / fs
hop = (NFFT - overlap) / fs

print("\n--- Parametros STFT ---")
print(f"NFFT = {NFFT}")
print(f"Resolucion frecuencia df = {df:.2f} Hz ({df / 1e3:.2f} kHz)")
print(f"Duracion ventana dt = {dt:.6f} s ({dt * 1e3:.3f} ms)")
print(f"Hop temporal = {hop:.6f} s ({hop * 1e3:.3f} ms)")


# ============================================================
# 5. TABLA DE METADATOS PARA PYTHON
# ============================================================

metadata_rows = []


# ============================================================
# 6. GENERAR ESPECTROGRAMAS
# ============================================================

for k in range(num_segmentos):
    idx_inicio = k * muestras_segmento

    # Leer solo el segmento actual
    senal_seg = leer_segmento_iq_float32_interleaved(
        nombre_datos,
        idx_inicio,
        muestras_segmento,
    )

    if senal_seg is None:
        print(f"[WARN] Segmento {k:04d} incompleto, se omite.")
        continue

    # Equivalente a:
    # [S, F, T] = spectrogram(senal_seg, win, overlap, nfft, fs, 'centered');
    F, T, S = spectrogram(
        senal_seg,
        fs=fs,
        window=win,
        nperseg=NFFT,
        noverlap=overlap,
        nfft=nfft,
        detrend=False,
        return_onesided=False,
        scaling="density",
        mode="complex",
    )

    # Emular 'centered' de MATLAB
    F = np.fft.fftshift(F)
    S = np.fft.fftshift(S, axes=0)

    # Equivalente a:
    # Pdb = 10*log10(abs(S).^2 + eps);
    Pdb = 10 * np.log10(np.abs(S) ** 2 + np.finfo(float).eps)

    # Frecuencia absoluta en MHz
    F_abs = (F + fc) / 1e6

    dpi = export_resolution
    fig_w = img_width_px / dpi
    fig_h = img_height_px / dpi

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])

    # Similar a imagesc(T, F_abs, Pdb)
    im = ax.imshow(
        Pdb,
        aspect="auto",
        origin="lower",
        extent=[
            T[0] if len(T) > 0 else 0,
            T[-1] if len(T) > 0 else duracion_segmento,
            F_abs[0],
            F_abs[-1],
        ],
        cmap=parula_cmap,
        interpolation="nearest",
    )

    ax.set_ylim((fc - fs / 2) / 1e6, (fc + fs / 2) / 1e6)
    ax.axis("off")

    if modo_escala == "auto":
        maxDb = np.max(Pdb)
        im.set_clim(maxDb - rango_dinamico_db, maxDb)
    elif modo_escala == "fija":
        im.set_clim(*caxis_fija_db)
    else:
        raise ValueError('modo_escala debe ser "auto" o "fija"')

    # Nombre unico del PNG
    nombre_png = f"{capture_id}__seg{k:04d}.png"
    ruta_png = carpeta_salida / nombre_png

    fig.savefig(
        ruta_png,
        dpi=export_resolution,
        pad_inches=0,
    )

    plt.close(fig)

    # Ruta relativa compatible con Python
    filename_rel = f"all/{label}/{nombre_png}"

    metadata_rows.append(
        {
            "filename": filename_rel,
            "label": label,
            "wifi_level": wifi_level,
            "condition": condition,
            "antenna": antenna,
            "distance_m": distance_m,
            "angle": angle_deg,
            "environment": environment,
            "fc": fc,
            "fs": fs,
            "gain": gain,
            "capture_id": capture_id,
            "platform": platform,
        }
    )

    print(f"Guardado segmento {k:04d}: {ruta_png}")

    # Liberar memoria de este segmento
    del senal_seg, F, T, S, Pdb, F_abs, fig, ax, im
    gc.collect()


# ============================================================
# 7. GUARDAR CSV DE METADATOS
# ============================================================

metadata_rows_df = pd.DataFrame(metadata_rows)

if csv_salida.exists():
    metadata_antigua = pd.read_csv(csv_salida)
    metadata_total = pd.concat(
        [metadata_antigua, metadata_rows_df],
        ignore_index=True,
    )
else:
    metadata_total = metadata_rows_df

metadata_total.to_csv(csv_salida, index=False)

print("\nProceso terminado.")
print(f"PNGs generados en: {carpeta_salida}")
print(f"CSV actualizado en: {csv_salida}")
print(f"Filas añadidas en esta ejecución: {len(metadata_rows_df)}")
print(f"Filas totales en el CSV: {len(metadata_total)}")