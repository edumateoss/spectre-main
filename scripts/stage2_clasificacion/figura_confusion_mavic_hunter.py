"""
Figura comparativa para P4: por que mavic_novideo se confunde con hunter.

Tres espectrogramas representativos del conjunto de prueba:
  (a) mavic_video    -> bandas anchas de video (OFDM) + control FHSS: distintivo.
  (b) mavic_novideo  -> solo rafagas de control FHSS dispersas.
  (c) hunter         -> rafagas dispersas, morfologicamente similares a (b).

La similitud entre (b) y (c) explica el colapso de mavic_novideo hacia hunter
en la matriz de confusion del Stage 2.

Salida: memoria_yolo/figuras/resultados/stage2_morfologia_mavic_hunter.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALL = os.path.join(ROOT, "data", "stage2_classifier_v1", "all")
OUT = os.path.join(ROOT, "memoria_yolo", "figuras", "resultados")
os.makedirs(OUT, exist_ok=True)

paneles = [
    (os.path.join(ALL, "mavic", "mavic_10_+10__seg0005.png"),
     "(a) Mavic con vídeo"),
    (os.path.join(ALL, "mavic", "mavic_6__seg0005.png"),
     "(b) Mavic sin vídeo"),
    (os.path.join(ALL, "hunter", "hunter_2__seg0005.png"),
     "(c) Hunter"),
]

plt.rcParams.update({"font.family": "serif", "font.size": 11})

fig, axes = plt.subplots(1, 3, figsize=(13, 2.9))
for ax, (path, titulo) in zip(axes, paneles):
    ax.imshow(mpimg.imread(path), aspect="auto")
    ax.set_title(titulo, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
out = os.path.join(OUT, "stage2_morfologia_mavic_hunter.png")
fig.savefig(out, dpi=180, bbox_inches="tight")
plt.close(fig)
print("Guardado:", out)
