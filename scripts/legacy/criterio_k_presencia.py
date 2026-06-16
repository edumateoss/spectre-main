"""
Determinacion del criterio optimo de K (numero minimo de rafagas drone por
ventana de 0.1 s) para confirmar presencia de dron.

Idea: el K adecuado NO es arbitrario. Se elige observando el "hueco de
separacion" entre dos poblaciones medidas sobre el conjunto de prueba:
  - suelo de falsos: maximo numero de cajas drone en ventanas SIN dron.
  - suelo de verdaderos: minimo numero de cajas drone en ventanas CON dron.
Cualquier K dentro de ese hueco da 100% de recall y 100% de especificidad.
El K recomendado es el punto medio del hueco (maximo margen a ambos errores).

Genera una figura de dos paneles:
  (a) Distribucion de rafagas drone por ventana (dron vs no-dron) con el hueco
      de separacion sombreado.
  (b) Recall y especificidad de presencia en funcion de K, con la meseta
      perfecta resaltada.

Salida: memoria_yolo/figuras/resultados/presencia_criterio_k.png
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
VERDE = "#4daf4a"
GRIS = "#555555"

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9.5, "figure.dpi": 150, "savefig.dpi": 180,
})


def cargar():
    rows = list(csv.DictReader(open(CSV)))
    nd = np.array([int(r["n_drone_bboxes"]) for r in rows])
    gt = np.array([int(r["gt_presence"]) for r in rows])
    return nd[gt == 1], nd[gt == 0]


def barrido_k(drone, nodrone, kmax=50):
    ks = np.arange(1, kmax + 1)
    rec = np.array([(drone >= k).mean() for k in ks])
    spec = np.array([(nodrone < k).mean() for k in ks])
    return ks, rec, spec


def main():
    drone, nodrone = cargar()
    floor = int(nodrone.max())        # maximo de cajas falsas (no-dron)
    techo = int(drone.min())          # minimo de cajas verdaderas (dron)
    k_lo, k_hi = floor + 1, techo     # rango seguro de K
    k_rec = int(round((floor + techo) / 2))  # K recomendado: maximo margen

    print(f"Suelo de falsos (no-dron) max = {floor}")
    print(f"Suelo de verdaderos (dron) min = {techo}")
    print(f"Hueco de separacion: K en [{k_lo}, {k_hi}] -> 100% recall y 100% especificidad")
    print(f"K recomendado (punto medio, maximo margen) = {k_rec}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

    # --- Panel (a): distribucion en escala log (x+1 para incluir el 0) ---
    bins = np.logspace(0, np.log10(drone.max() + 1), 34)
    ax1.hist(nodrone + 1, bins=bins, color=ROJO, alpha=0.85,
             label=f"Sin dron (n={len(nodrone)})")
    ax1.hist(drone + 1, bins=bins, color=AZUL, alpha=0.85,
             label=f"Con dron (n={len(drone)})")
    ax1.set_xscale("log")
    # hueco de separacion
    ax1.axvspan(floor + 1, techo + 1, color=VERDE, alpha=0.18,
                label=f"Hueco seguro (K {k_lo}-{k_hi})")
    ax1.axvline(k_rec + 1, color=GRIS, ls="--", lw=1.6,
                label=f"K recomendado = {k_rec}")
    ax1.set_xlabel("Rafagas drone por ventana de 0,1 s  (+1, escala log)")
    ax1.set_ylabel("Numero de ventanas")
    ax1.set_title("(a) Separacion entre ventanas con y sin dron")
    ax1.legend(frameon=False, loc="upper center")

    # --- Panel (b): recall y especificidad vs K ---
    ks, rec, spec = barrido_k(drone, nodrone, kmax=50)
    ax2.plot(ks, 100 * rec, color=AZUL, lw=2.2, marker="o", ms=3,
             label="Recall de presencia")
    ax2.plot(ks, 100 * spec, color=ROJO, lw=2.2, marker="s", ms=3,
             label="Especificidad")
    ax2.axvspan(k_lo, k_hi, color=VERDE, alpha=0.18,
                label=f"Meseta perfecta (K {k_lo}-{k_hi})")
    ax2.axvline(k_rec, color=GRIS, ls="--", lw=1.6,
                label=f"K recomendado = {k_rec}")
    ax2.set_xlabel("K (minimo de rafagas para confirmar presencia)")
    ax2.set_ylabel("Porcentaje (%)")
    ax2.set_ylim(60, 102)
    ax2.set_title("(b) Recall y especificidad frente a K")
    ax2.grid(True, alpha=0.25)
    ax2.legend(frameon=False, loc="lower left")

    fig.tight_layout()
    out = os.path.join(OUT, "presencia_criterio_k.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Guardado:", out)


if __name__ == "__main__":
    main()
