"""
Figura limpia de la evolucion del entrenamiento YOLOv8 para la memoria.

Reduce la rejilla de 10 paneles que genera Ultralytics (results.png) a tres
paneles que sostienen directamente el texto del capitulo:
  1. Perdidas de entrenamiento (box, cls, dfl): descienden de forma continua.
  2. Precision y Recall de validacion: plateau aproximado entre epocas 20 y 30.
  3. mAP@50 y mAP@50-95 de validacion: plateau con leve mejora del mAP@50-95.

Uso:
    uv run python scripts/plot_yolo_results_clean.py
    uv run python scripts/plot_yolo_results_clean.py --results <ruta>/results.csv

Salida:
    <output-dir>/yolo_run02_curvas.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_DEFAULT_RESULTS = Path(
    "data/mini3_detector_python_v3/artifacts/yolo/run02_60ep/results.csv"
)


def parse_args():
    p = argparse.ArgumentParser(description="Figura limpia de curvas YOLOv8")
    p.add_argument("--results", type=Path, default=_DEFAULT_RESULTS)
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Por defecto, la carpeta del results.csv")
    p.add_argument("--plateau", type=int, nargs=2, default=(20, 30),
                   help="Rango de epocas a sombrear como plateau")
    return p.parse_args()


def load_results(path):
    """Lee el results.csv de Ultralytics en columnas (claves sin espacios)."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        cols = {k.strip(): [] for k in reader.fieldnames}
        for row in reader:
            for k in reader.fieldnames:
                cols[k.strip()].append(float(row[k]))
    return {k: np.asarray(v) for k, v in cols.items()}


def smooth(y, window=5):
    """
    Media movil centrada que respeta los extremos.

    En cada punto promedia solo las muestras disponibles dentro de la ventana,
    evitando el artefacto de caida en los bordes de np.convolve(mode='same').
    """
    if len(y) < 3:
        return y
    half = window // 2
    out = np.empty(len(y), dtype=float)
    for i in range(len(y)):
        a = max(0, i - half)
        b = min(len(y), i + half + 1)
        out[i] = y[a:b].mean()
    return out


def plot_series(ax, x, y, color, label):
    """Dibuja la serie cruda tenue y la tendencia suavizada encima."""
    ax.plot(x, y, color=color, alpha=0.25, linewidth=1)
    ax.plot(x, smooth(y), color=color, linewidth=2, label=label)


def main():
    args = parse_args()
    d = load_results(args.results)
    out_dir = args.output_dir or args.results.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    ep = d["epoch"]
    lo, hi = args.plateau

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))

    ax = axes[0]
    plot_series(ax, ep, d["train/box_loss"], "#1f77b4", "box")
    plot_series(ax, ep, d["train/cls_loss"], "#ff7f0e", "cls")
    plot_series(ax, ep, d["train/dfl_loss"], "#2ca02c", "dfl")
    ax.set_title("Perdidas de entrenamiento", fontsize=10)
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Perdida")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    plot_series(ax, ep, d["metrics/precision(B)"], "#1f77b4", "Precision")
    plot_series(ax, ep, d["metrics/recall(B)"], "#d62728", "Recall")
    ax.set_title("Precision y Recall (validacion)", fontsize=10)
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Valor")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, frameon=False, loc="center right")

    ax = axes[2]
    plot_series(ax, ep, d["metrics/mAP50(B)"], "#9467bd", "mAP@50")
    plot_series(ax, ep, d["metrics/mAP50-95(B)"], "#8c564b", "mAP@50-95")
    ax.set_title("mAP (validacion)", fontsize=10)
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Valor")
    ax.legend(fontsize=8, frameon=False, loc="center right")

    for ax in axes:
        ax.axvspan(lo, hi, color="gray", alpha=0.10)
        ax.grid(True, alpha=0.25)
        ax.margins(x=0.01)

    etiqueta = "plateau (~" + str(lo) + "-" + str(hi) + ")"
    axes[1].annotate(etiqueta, xy=((lo + hi) / 2.0, 0.12),
                     ha="center", fontsize=8, color="gray")

    fig.tight_layout()
    out_path = out_dir / "yolo_run02_curvas.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Guardado:", out_path)


if __name__ == "__main__":
    main()
