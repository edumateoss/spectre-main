"""
Reorganiza los montajes 1x6 del barrido SNR (snr_examples.py) en rejillas 2x3
para que cada espectrograma se vea grande en la memoria.

Recorta los 6 paneles del montaje original y los recoloca con titulos propios.
Salida (sobrescribe):
  memoria_yolo/figuras/resultados/snr_ejemplo_mini3_awgn.png
  memoria_yolo/figuras/resultados/snr_ejemplo_mini3_fhss.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "memoria_yolo", "figuras", "resultados")
TITULOS = ["limpio", "SNR = +20 dB", "SNR = +10 dB",
           "SNR = 0 dB", "SNR = -10 dB", "SNR = -20 dB"]

plt.rcParams.update({"font.family": "serif", "font.size": 12})


def detectar_paneles(arr):
    white = (arr > 245).all(axis=2)
    # filas con contenido (espectrograma)
    rows = np.where(white.mean(axis=1) < 0.6)[0]
    r0, r1 = rows.min(), rows.max()
    # columnas: bloques no blancos
    gap = white.mean(axis=0) > 0.85
    idx = np.where(~gap)[0]
    bloques = []
    start = prev = idx[0]
    for x in idx[1:]:
        if x - prev > 8:
            bloques.append((start, prev)); start = x
        prev = x
    bloques.append((start, prev))
    return r0, r1, bloques


def rehacer(nombre, suptitulo):
    path = os.path.join(FIG, nombre)
    arr = np.array(Image.open(path).convert("RGB"))
    r0, r1, bloques = detectar_paneles(arr)
    paneles = [arr[r0:r1, a:b] for (a, b) in bloques]

    fig, axes = plt.subplots(2, 3, figsize=(12, 5.2))
    for k, ax in enumerate(axes.flat):
        ax.imshow(paneles[k], aspect="auto")
        ax.set_title(TITULOS[k], fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(suptitulo, fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Reescrito:", path, "->", len(paneles), "paneles 2x3")


rehacer("snr_ejemplo_mini3_awgn.png", "DJI Mini 3 — ruido AWGN")
rehacer("snr_ejemplo_mini3_fhss.png", "DJI Mini 3 — interferencia FHSS sintetica")
