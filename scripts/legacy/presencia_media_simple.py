"""
Figura simple para el criterio de presencia de dron (P3).

Idea sencilla: contar cuantas rafagas drone detecta el modelo en cada ventana
de 0.1 s y mirar la media. Las ventanas con dron promedian ~118 rafagas; las
de solo interferencia, ~0,2. Se fija un umbral minimo redondo y arbitrario
(K=10), muy por debajo de la media del dron y por encima del ruido residual.

Salida: memoria_yolo/figuras/resultados/presencia_media_rafagas.png
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "data", "mini3_detector_python_v3", "artifacts",
                   "yolo", "run02_60ep_test", "presence",
                   "per_image_predictions.csv")
OUT = os.path.join(ROOT, "memoria_yolo", "figuras", "resultados")
os.makedirs(OUT, exist_ok=True)

AZUL = "#2166ac"
ROJO = "#d6604d"
GRIS = "#555555"
K = 10

plt.rcParams.update({"font.family": "serif", "font.size": 12})


def main():
    rows = list(csv.DictReader(open(CSV)))
    nd = np.array([int(r["n_drone_bboxes"]) for r in rows])
    gt = np.array([int(r["gt_presence"]) for r in rows])
    media_dron = nd[gt == 1].mean()
    media_nodron = nd[gt == 0].mean()

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    barras = ax.bar(["Con dron", "Sin dron"], [media_dron, media_nodron],
                    color=[AZUL, ROJO], width=0.55, edgecolor="white")
    # valor encima de cada barra
    for b, v in zip(barras, [media_dron, media_nodron]):
        ax.text(b.get_x() + b.get_width() / 2, v + 3,
                f"{v:.1f}".replace(".", ","), ha="center", va="bottom",
                fontsize=12, fontweight="bold")
    # umbral
    ax.axhline(K, color=GRIS, ls="--", lw=1.6)
    ax.text(1.45, K + 3, f"Umbral K = {K}", color=GRIS, ha="right", fontsize=11)

    ax.set_ylabel("Ráfagas drone detectadas por ventana (media)")
    ax.set_title("Media de ráfagas por ventana de 0,1 s")
    ax.set_ylim(0, media_dron * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = os.path.join(OUT, "presencia_media_rafagas.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"media dron={media_dron:.1f}  media no-dron={media_nodron:.2f}")
    print("Guardado:", out)


if __name__ == "__main__":
    main()
