"""
Genera las figuras del Capitulo 5 que no existen todavia:
  1. Curvas de aprendizaje ResNet18 binario (loss y F1 train/val vs epoch)
  2. Heatmap de recall del sweep de presencia (conf x K)
  3. Barras de deteccion por captura en el punto operativo recomendado
  4. Matriz de confusion normalizada del clasificador Stage 2 (6 clases)

Salidas en: memoria_yolo/figuras/resultados/
"""

import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "memoria_yolo", "figuras", "resultados")
os.makedirs(OUT, exist_ok=True)

METRICS_CSV   = os.path.join(ROOT, "data", "mini3_detector_python_v3", "artifacts", "metrics.csv")
PRESENCE_JSON = os.path.join(ROOT, "data", "mini3_detector_python_v3", "artifacts",
                             "yolo", "run02_60ep_test", "presence", "presence_metrics.json")
CM_CSV        = os.path.join(ROOT, "data", "stage2_classifier_v2_mavic_split", "artifacts",
                             "resnet18_stage2_mavic_split", "confusion_matrix_test.csv")

# Estilo comun
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

AZUL  = "#2166ac"
ROJO  = "#d6604d"
GRIS  = "#636363"


# ===========================================================================
# 1. Curvas de aprendizaje ResNet18 binario
# ===========================================================================
def plot_learning_curves():
    df = pd.read_csv(METRICS_CSV)
    train = df[df["phase"] == "train"].set_index("epoch")
    val   = df[df["phase"] == "val"].set_index("epoch")

    epochs = train.index.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # --- Loss ---
    ax = axes[0]
    ax.plot(epochs, train["loss"], color=AZUL,  lw=1.8, label="Entrenamiento")
    ax.plot(epochs, val["loss"],   color=ROJO,  lw=1.8, linestyle="--", label="Validacion")
    ax.axvline(5, color=GRIS, lw=1.0, linestyle=":", label="Mejor epoca (5)")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Perdida (cross-entropy)")
    ax.set_title("Perdida")
    ax.legend()
    ax.set_xlim(1, max(epochs))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    # --- F1 ---
    ax = axes[1]
    ax.plot(epochs, train["f1"], color=AZUL, lw=1.8, label="Entrenamiento")
    ax.plot(epochs, val["f1"],   color=ROJO, lw=1.8, linestyle="--", label="Validacion")
    ax.axvline(5, color=GRIS, lw=1.0, linestyle=":", label="Mejor epoca (5)")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("F1")
    ax.set_title("F1")
    ax.legend()
    ax.set_xlim(1, max(epochs))
    ax.set_ylim(0.97, 1.002)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    fig.suptitle("ResNet18 binario — curvas de aprendizaje", y=1.02)
    fig.tight_layout()
    path = os.path.join(OUT, "resnet18_binary_curvas.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Guardado: {path}")


# ===========================================================================
# 2. Heatmap recall del sweep de presencia (conf x K)
# ===========================================================================
def plot_presencia_heatmap():
    with open(PRESENCE_JSON) as f:
        data = json.load(f)

    confs = data["sweep_conf"]
    ks    = data["sweep_min_drone_bursts"]

    # Construir matriz de recall (filas=conf, cols=K)
    recall = np.zeros((len(confs), len(ks)))
    for entry in data["metrics"]:
        ci = confs.index(entry["conf_threshold"])
        ki = ks.index(entry["min_drone_bursts"])
        recall[ci, ki] = entry["recall"] * 100  # en %

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(recall, cmap="RdYlGn", vmin=60, vmax=100, aspect="auto")

    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([f"K={k}" for k in ks])
    ax.set_yticks(range(len(confs)))
    ax.set_yticklabels([f"conf={c}" for c in confs])
    ax.set_xlabel("Minimo de rafagas requeridas (K)")
    ax.set_ylabel("Umbral de confianza")
    ax.set_title("Recall de presencia (%) — Precision y Especificidad = 1,00 en todas las celdas")

    # Anotar valores
    for i in range(len(confs)):
        for j in range(len(ks)):
            val = recall[i, j]
            color = "black" if val > 75 else "white"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=10, color=color, fontweight="bold")

    # Marcar punto operativo recomendado (conf=0.10, K=3)
    ci_rec = confs.index(0.1)
    ki_rec = ks.index(3)
    ax.add_patch(plt.Rectangle((ki_rec - 0.5, ci_rec - 0.5), 1, 1,
                                fill=False, edgecolor="black", lw=2.5))

    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Recall (%)")

    fig.tight_layout()
    path = os.path.join(OUT, "presencia_heatmap.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Guardado: {path}")


# ===========================================================================
# 3. Barras por captura en el punto operativo recomendado (conf=0.10, K=3)
# ===========================================================================
def plot_presencia_por_captura():
    with open(PRESENCE_JSON) as f:
        data = json.load(f)

    # Buscar la entrada conf=0.10, K=3
    entry = next(e for e in data["metrics"]
                 if e["conf_threshold"] == 0.1 and e["min_drone_bursts"] == 3)

    capturas = []
    porcentajes = []
    colores = []
    etiquetas = []

    labels_map = {
        "mini3_drone_d10_w3_s02_cap01": "Dron 10 m\nWiFi W3",
        "mini3_drone_d5_w1_s01_cap01":  "Dron 5 m\nWiFi W1",
        "mini3_nodrone_w1_s01_cap03":   "Sin dron\nWiFi W1",
        "mini3_nodrone_w3_s01_cap03":   "Sin dron\nWiFi W3",
    }

    for cap_id, info in entry["per_capture"].items():
        capturas.append(labels_map.get(cap_id, cap_id))
        porcentajes.append(info["pct_pred_present"])
        if info["capture_type"] == "drone":
            colores.append(AZUL)
        else:
            colores.append(ROJO)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(capturas, porcentajes, color=colores, edgecolor="white", width=0.55)

    # Linea de referencia al 50%
    ax.axhline(50, color=GRIS, lw=1.0, linestyle=":", label="50 %")

    # Etiquetar barras
    for bar, pct in zip(bars, porcentajes):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{pct:.0f}%",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Ventanas con dron detectado (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Deteccion de presencia por captura\n(conf = 0,10 ; K = 3 rafagas minimas)")
    ax.set_yticks([0, 25, 50, 75, 100])

    # Leyenda manual
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=AZUL, label="Captura con dron"),
        Patch(facecolor=ROJO, label="Captura sin dron"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    fig.tight_layout()
    path = os.path.join(OUT, "presencia_por_captura.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Guardado: {path}")


# ===========================================================================
# 4. Matriz de confusion normalizada Stage 2 (6 clases)
# ===========================================================================
def plot_confusion_matrix_stage2():
    df = pd.read_csv(CM_CSV, index_col=0)

    clases = list(df.index)
    cm_abs = df.values.astype(float)

    # Normalizar por filas (recall por clase)
    row_sums = cm_abs.sum(axis=1, keepdims=True)
    cm_norm  = np.divide(cm_abs, row_sums, where=row_sums != 0)

    # Etiquetas mas legibles
    etiquetas = ["F450", "Hunter", "Mavic\nvideo", "Mavic\nno video", "Mini 3", "Interf."]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(len(etiquetas)))
    ax.set_xticklabels(etiquetas, fontsize=10)
    ax.set_yticks(range(len(etiquetas)))
    ax.set_yticklabels(etiquetas, fontsize=10)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.set_title("Matriz de confusion normalizada — Clasificador Stage 2 (6 clases)")

    for i in range(len(etiquetas)):
        for j in range(len(etiquetas)):
            val = cm_norm[i, j]
            color = "white" if val > 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9.5, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Fraccion de muestras")

    fig.tight_layout()
    path = os.path.join(OUT, "stage2_confusion_matrix_norm.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Guardado: {path}")


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    print("Generando figuras del Capitulo 5...")
    plot_learning_curves()
    plot_presencia_heatmap()
    plot_presencia_por_captura()
    plot_confusion_matrix_stage2()
    print("Listo.")
