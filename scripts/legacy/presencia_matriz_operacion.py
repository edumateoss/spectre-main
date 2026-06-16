"""
Matriz de confusion del clasificador de presencia en el punto de operacion
recomendado (thr=0.10, K=3), a nivel de imagen.

Conteos tomados de presence_metrics.json: TP=273, FP=0, FN=27, TN=300.

Salida: memoria_yolo/figuras/resultados/presencia_matriz_operacion.png
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON = os.path.join(ROOT, "data", "mini3_detector_python_v3", "artifacts",
                    "yolo", "run02_60ep_test", "presence", "presence_metrics.json")
OUT = os.path.join(ROOT, "memoria_yolo", "figuras", "resultados")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.size": 12})


def conteos(thr=0.1, k=3):
    m = json.load(open(JSON))["metrics"]
    for e in m:
        if abs(e["conf_threshold"] - thr) < 1e-9 and e["min_drone_bursts"] == k:
            c = e["counts"]
            return c["TP"], c["FP"], c["FN"], c["TN"]
    raise ValueError("config no encontrada")


def main():
    tp, fp, fn, tn = conteos()
    # filas = real (Dron, Sin dron); columnas = predicho (Dron, Sin dron)
    M = np.array([[tp, fn], [fp, tn]])
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=M.max())
    ax.set_xticks([0, 1], ["Dron", "Sin dron"])
    ax.set_yticks([0, 1], ["Dron", "Sin dron"])
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    etiq = [["VP", "FN"], ["FP", "VN"]]
    for i in range(2):
        for j in range(2):
            color = "white" if M[i, j] > M.max() * 0.5 else "black"
            ax.text(j, i, f"{M[i, j]}\n({etiq[i][j]})", ha="center", va="center",
                    color=color, fontsize=14, fontweight="bold")
    ax.set_title("Matriz de confusion de presencia\n(thr = 0,10,  K = 3)")
    fig.tight_layout()
    out = os.path.join(OUT, "presencia_matriz_operacion.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
    print("Guardado:", out)


if __name__ == "__main__":
    main()
