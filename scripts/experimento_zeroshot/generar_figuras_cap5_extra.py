"""
Genera figuras adicionales del Capitulo 5:
  1. Distribución de ráfagas detectadas por imagen (n_drone_bboxes)
  2. Curvas de aprendizaje Stage 2 clasificador 6 clases
  3. Panel de matrices de confusion a distintos SNR (AWGN vs FHSS)

Salidas en: memoria_yolo/figuras/resultados/
"""

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT  = os.path.join(ROOT, "memoria_yolo", "figuras", "resultados")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

AZUL = "#2166ac"
ROJO = "#d6604d"
VERDE = "#4dac26"
GRIS  = "#636363"

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
PER_IMAGE_CSV = os.path.join(
    ROOT, "data", "mini3_detector_python_v3", "artifacts",
    "yolo", "run02_60ep_test", "presence", "per_image_predictions.csv")

STAGE2_METRICS = os.path.join(
    ROOT, "data", "stage2_classifier_v2_mavic_split", "artifacts",
    "resnet18_stage2_mavic_split", "metrics.csv")

CM_DIR = os.path.join(
    ROOT, "data", "stage2_classifier_v2_mavic_split", "artifacts",
    "snr_sweep", "confusion_matrices")

CLASES_6 = ["F450", "Hunter", "Mavic\nvideo", "Mavic\nno video", "Mini 3", "Interf."]


# ===========================================================================
# 1. Distribución de ráfagas por imagen
# ===========================================================================
def plot_ráfagas_por_imagen():
    df = pd.read_csv(PER_IMAGE_CSV)

    # Separar por tipo de captura
    drone    = df[df["capture_type"] == "drone"]
    nodrone  = df[df["capture_type"] == "nodrone"]

    # Sub-capturas drone
    d10 = drone[drone["capture_id"] == "mini3_drone_d10_w3_s02_cap01"]["n_drone_bboxes"]
    d5  = drone[drone["capture_id"] == "mini3_drone_d5_w1_s01_cap01"]["n_drone_bboxes"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- Histograma por captura drone ---
    ax = axes[0]
    bins = np.arange(0, max(d10.max(), d5.max()) + 5, 3)
    ax.hist(d5,  bins=bins, alpha=0.75, color=AZUL, label="Dron 5 m / WiFi W1",  edgecolor="white")
    ax.hist(d10, bins=bins, alpha=0.75, color=ROJO, label="Dron 10 m / WiFi W3", edgecolor="white")
    ax.axvline(3, color=GRIS, lw=1.2, linestyle=":", label="Umbral K=3")
    ax.set_xlabel("Ráfagas drone detectadas por ventana (0,1 s)")
    ax.set_ylabel("Número de imágenes")
    ax.set_title("Distribución de detecciones en capturas de dron")
    ax.legend()

    # --- Medias y percentiles como tabla visual ---
    ax2 = axes[1]
    datos = {
        "Dron 5m/W1": d5,
        "Dron 10m/W3": d10,
        "Sin dron W1": nodrone[nodrone["capture_id"].str.contains("w1")]["n_drone_bboxes"],
        "Sin dron W3": nodrone[nodrone["capture_id"].str.contains("w3")]["n_drone_bboxes"],
    }
    etiquetas = list(datos.keys())
    medias    = [v.mean() for v in datos.values()]
    p25       = [v.quantile(0.25) for v in datos.values()]
    p75       = [v.quantile(0.75) for v in datos.values()]
    colores_barras = [AZUL, ROJO, "#999999", "#999999"]

    x = np.arange(len(etiquetas))
    bars = ax2.bar(x, medias, color=colores_barras, edgecolor="white", width=0.6)
    err_low  = np.clip(np.array(medias) - np.array(p25), 0, None)
    err_high = np.clip(np.array(p75) - np.array(medias), 0, None)
    ax2.errorbar(x, medias,
                 yerr=[err_low, err_high],
                 fmt="none", color="black", capsize=5, lw=1.5)
    ax2.axhline(3, color=GRIS, lw=1.2, linestyle=":", label="Umbral K=3")
    ax2.set_xticks(x)
    ax2.set_xticklabels(etiquetas, fontsize=9.5)
    ax2.set_ylabel("Ráfagas drone por ventana (media ± IQR)")
    ax2.set_title("Media de detecciones por tipo de captura")
    ax2.legend()

    # Anotar media
    for bar, m in zip(bars, medias):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{m:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.suptitle("Ráfagas detectadas por ventana de 0,1 s — detector YOLO (conf ≥ 0,10)", y=1.01)
    fig.tight_layout()
    path = os.path.join(OUT, "yolo_rafagas_por_imagen.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Guardado: {path}")


# ===========================================================================
# 2. Curvas de aprendizaje Stage 2 (6 clases)
# ===========================================================================
def plot_stage2_learning_curves():
    df = pd.read_csv(STAGE2_METRICS)
    train = df[df["phase"] == "train"].set_index("epoch")
    val   = df[df["phase"] == "val"].set_index("epoch")
    epochs = train.index.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # --- Loss ---
    ax = axes[0]
    ax.plot(epochs, train["loss"], color=AZUL, lw=1.8, label="Entrenamiento")
    ax.plot(epochs, val["loss"],   color=ROJO, lw=1.8, linestyle="--", label="Validación")
    ax.axvline(10, color=GRIS, lw=1.0, linestyle=":", label="Mejor época (10)")
    ax.set_xlabel("Época")
    ax.set_ylabel("Pérdida (cross-entropy)")
    ax.set_title("Pérdida")
    ax.legend()
    ax.set_xlim(1, max(epochs))

    # --- F1 macro ---
    ax = axes[1]
    ax.plot(epochs, train["f1_macro"], color=AZUL, lw=1.8, label="Entrenamiento")
    ax.plot(epochs, val["f1_macro"],   color=ROJO, lw=1.8, linestyle="--", label="Validación")
    ax.axvline(10, color=GRIS, lw=1.0, linestyle=":", label="Mejor época (10)")
    ax.set_xlabel("Época")
    ax.set_ylabel("F1 macro")
    ax.set_title("F1 macro")
    ax.legend()
    ax.set_xlim(1, max(epochs))
    ax.set_ylim(0.75, 1.01)

    fig.suptitle("ResNet18 Stage 2 (6 clases) — curvas de aprendizaje", y=1.02)
    fig.tight_layout()
    path = os.path.join(OUT, "stage2_curvas_aprendizaje.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Guardado: {path}")


# ===========================================================================
# 3. Panel de matrices de confusion a SNR clave
# ===========================================================================
def _load_cm(noise, snr_str):
    """Carga una matriz de confusion desde CSV."""
    fname = f"cm_{noise}_{snr_str}.csv"
    path  = os.path.join(CM_DIR, fname)
    df = pd.read_csv(path, index_col=0)
    cm = df.values.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)
    return cm_norm


def _draw_cm(ax, cm_norm, title, etiquetas, fontsize=7.5):
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(etiquetas)))
    ax.set_xticklabels(etiquetas, fontsize=fontsize)
    ax.set_yticks(range(len(etiquetas)))
    ax.set_yticklabels(etiquetas, fontsize=fontsize)
    ax.set_title(title, fontsize=9.5, pad=4)
    for i in range(len(etiquetas)):
        for j in range(len(etiquetas)):
            v = cm_norm[i, j]
            color = "white" if v > 0.55 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=fontsize - 0.5, color=color)
    return im


def plot_snr_confusion_panel():
    etiquetas = ["F450", "Hunt.", "M.vid", "M.nov", "Mini3", "Int."]

    # AWGN: inf, +5, 0, -5 dB
    # FHSS: inf, -5, -10, -20 dB
    configs = [
        ("AWGN",     [("inf", "sin ruido"), ("+5",  "+5 dB"), ("+0",  "0 dB"),  ("-5",  "−5 dB")]),
        ("FHSS",     [("inf", "sin ruido"), ("-5",  "−5 dB"), ("-10", "−10 dB"), ("-20", "−20 dB")]),
    ]

    noise_labels = {"AWGN": "AWGN", "FHSS": "FHSS-like"}
    noise_keys   = {"AWGN": "awgn", "FHSS": "fhss"}

    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))

    for row, (noise, snr_list) in enumerate(configs):
        nkey = noise_keys[noise]
        for col, (snr_str, snr_label) in enumerate(snr_list):
            ax = axes[row, col]
            try:
                cm_norm = _load_cm(nkey, snr_str)
            except FileNotFoundError:
                ax.set_visible(False)
                continue
            title = f"{noise_labels[noise]} — {snr_label}"
            im = _draw_cm(ax, cm_norm, title, etiquetas)
            if col == 0:
                ax.set_ylabel("Clase real", fontsize=9)
            if row == 1:
                ax.set_xlabel("Clase predicha", fontsize=9)

    # Colorbar comun
    fig.subplots_adjust(right=0.88, hspace=0.45, wspace=0.35)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax).set_label("Fracción de muestras", fontsize=9)

    fig.suptitle(
        "Evolución de la matriz de confusión con el SNR\n"
        "Fila superior: AWGN — Fila inferior: FHSS-like",
        fontsize=11, y=1.01)

    path = os.path.join(OUT, "snr_confusion_panel.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Guardado: {path}")


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    print("Generando figuras adicionales del Capitulo 5...")
    plot_ráfagas_por_imagen()
    plot_stage2_learning_curves()
    plot_snr_confusion_panel()
    print("Listo.")
