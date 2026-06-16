"""
analizar_zeroshot_detalle.py
============================

Analisis detallado del experimento zero-shot a partir del report CSV ya
generado por infer_zeroshot_campo.py. No requiere correr el modelo YOLO.

Genera en figuras_memoria/:
  zeroshot_sweep_k.png        Recall vs K (K=1..5) para F450, Mavic y referencia Mini3
  zeroshot_por_captura.png    Recall por captura individual (analoga a presencia_por_captura)
  zeroshot_hunter_analisis.png Distribucion de cajas drone vs interference en Hunter

Exporta ademas:
  zeroshot_sweep_k.json       Tabla numerica del sweep K para la memoria

Uso:
  uv run python scripts/analizar_zeroshot_detalle.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
DIR_CAMPO  = Path("data/stage2_classifier_v1")
REPORT_CSV = DIR_CAMPO / "artifacts" / "zeroshot" / "zeroshot_report.csv"
FIGURAS    = DIR_CAMPO / "artifacts" / "zeroshot" / "figuras_memoria"

DPI  = 180
SEED = 42

COLOR = {
    "f450":   "#4CAF50",
    "mavic":  "#FF9800",
    "hunter": "#9E9E9E",
    "mini3":  "#2196F3",
}

# Recall de referencia Mini3 en v3 (presencia_analysis.py, conf=0.10, K=k)
# K=1:99.0  K=2:95.3  K=3:91.0  K=4:89.0  K=5: estimado ~87
RECALL_MINI3 = {1: 99.0, 2: 95.3, 3: 91.0, 4: 89.0, 5: 87.0}

# Nombres legibles de captura para el eje X
ALIAS_CAPTURA = {
    "f450_1":          "f450-c1",
    "f450_2":          "f450-c2",
    "f450_3":          "f450-c3",
    "f450_4":          "f450-c4",
    "f450_5":          "f450-c5",
    "f450_6":          "f450-c6",
    "f450_7":          "f450-c7",
    "f450_8":          "f450-c8",
    "mavic_4":         "mvc-c4",
    "mavic_5":         "mvc-c5",
    "mavic_6":         "mvc-c6",
    "mavic_10":        "mvc-10",
    "mavic_10_+10":    "mvc-10+",
    "mavic_10_+10_1":  "mvc-10+1",
    "mavic_10_-10_1":  "mvc-10-1",
    "mavic_10_-10_2":  "mvc-10-2",
    "mavic_10_captura1": "mvc-lim",
}


# ---------------------------------------------------------------------------
# Figura 1: Sweep K — recall por K para F450, Mavic y referencia Mini3
# ---------------------------------------------------------------------------
def fig_sweep_k(df: pd.DataFrame, output_path: Path) -> dict:
    ks = [1, 2, 3, 4, 5]
    tipos = ["f450", "mavic", "hunter"]
    etiquetas = {"f450": "F450 (Samuel)", "mavic": "Mavic Pro (Alvaro)", "hunter": "Hunter"}

    resultados: dict[str, dict] = {"mini3": {}, "f450": {}, "mavic": {}, "hunter": {}}

    for tipo in tipos:
        sub = df[df["drone_type"] == tipo]
        for k in ks:
            recall = 100.0 * (sub["n_boxes_drone"] >= k).sum() / len(sub)
            resultados[tipo][k] = round(recall, 1)

    for k in ks:
        resultados["mini3"][k] = RECALL_MINI3[k]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Mini3 como linea de referencia punteada
    ax.plot(ks, [resultados["mini3"][k] for k in ks],
            color=COLOR["mini3"], linestyle="--", linewidth=1.5,
            marker="s", markersize=5, label="Mini3 (referencia v3)", zorder=3)

    for tipo in tipos:
        vals = [resultados[tipo][k] for k in ks]
        ax.plot(ks, vals,
                color=COLOR[tipo], linewidth=2, marker="o", markersize=6,
                label=etiquetas[tipo], zorder=4)

    ax.set_xlabel("K — minimo de rafagas drone para declarar presencia", fontsize=10)
    ax.set_ylabel("Recall de presencia (%)", fontsize=10)
    ax.set_title("Recall de presencia zero-shot vs K\n(conf = 0,10)", fontsize=11)
    ax.set_xticks(ks)
    ax.set_ylim(-5, 108)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)
    print(f"  Guardada: {output_path.name}")

    return resultados


# ---------------------------------------------------------------------------
# Figura 2: Recall por captura individual (F450 + Mavic)
# ---------------------------------------------------------------------------
def fig_por_captura(df: pd.DataFrame, output_path: Path) -> None:
    sub = df[df["drone_type"].isin(["f450", "mavic"])].copy()

    resumen = (
        sub.groupby(["capture_id", "drone_type"])
        .apply(lambda g: pd.Series({
            "recall": 100.0 * g["detectado"].sum() / len(g),
            "n": len(g),
        }), include_groups=False)
        .reset_index()
    )

    # Orden: primero F450, luego Mavic; dentro de cada tipo por recall desc
    orden_tipo = {"f450": 0, "mavic": 1}
    resumen["_orden"] = resumen["drone_type"].map(orden_tipo)
    resumen = resumen.sort_values(["_orden", "recall"], ascending=[True, False])

    etiquetas = [ALIAS_CAPTURA.get(c, c) for c in resumen["capture_id"]]
    colores   = [COLOR[t] for t in resumen["drone_type"]]
    recalls   = resumen["recall"].values

    fig, ax = plt.subplots(figsize=(11, 4))
    bars = ax.bar(range(len(resumen)), recalls, color=colores, width=0.7, zorder=3)

    # Valor encima de cada barra
    for i, (bar, val) in enumerate(zip(bars, recalls)):
        y = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y + 1.0,
            f"{val:.0f}%",
            ha="center", va="bottom", fontsize=7.5, fontweight="bold",
        )

    # Separador visual entre F450 y Mavic
    n_f450 = (resumen["drone_type"] == "f450").sum()
    ax.axvline(n_f450 - 0.5, color="#666", linestyle=":", linewidth=1.2)
    ax.text(n_f450 - 0.5 - 0.1, 102, "F450", ha="right", fontsize=8.5, color="#444")
    ax.text(n_f450 - 0.5 + 0.1, 102, "Mavic Pro", ha="left", fontsize=8.5, color="#444")

    ax.set_xticks(range(len(resumen)))
    ax.set_xticklabels(etiquetas, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Recall de presencia (%)", fontsize=10)
    ax.set_ylim(0, 112)
    ax.set_title(
        "Recall de presencia zero-shot por captura individual (conf = 0,10, K = 1)",
        fontsize=11,
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    parche_f450  = mpatches.Patch(color=COLOR["f450"],  label="F450")
    parche_mavic = mpatches.Patch(color=COLOR["mavic"], label="Mavic Pro")
    ax.legend(handles=[parche_f450, parche_mavic], fontsize=9, loc="lower right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)
    print(f"  Guardada: {output_path.name}")


# ---------------------------------------------------------------------------
# Figura 3: Hunter — distribucion de cajas drone vs interference
# ---------------------------------------------------------------------------
def fig_hunter_analisis(df: pd.DataFrame, output_path: Path) -> None:
    hunter = df[df["drone_type"] == "hunter"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel izquierdo: histograma de cajas interference por imagen
    ax = axes[0]
    bins = range(0, int(hunter["n_boxes_interf"].max()) + 2)
    ax.hist(hunter["n_boxes_interf"], bins=bins, color="#9E9E9E", edgecolor="white",
            linewidth=0.4, zorder=3)
    ax.set_xlabel("N.° de cajas 'interference' por espectrograma", fontsize=10)
    ax.set_ylabel("N.° de imagenes", fontsize=10)
    ax.set_title("Hunter: detecciones clasificadas\ncomo 'interference'", fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    media_interf = hunter["n_boxes_interf"].mean()
    ax.axvline(media_interf, color="#333", linestyle="--", linewidth=1.3,
               label=f"Media = {media_interf:.1f}")
    ax.legend(fontsize=9)

    # Panel derecho: comparativa drone vs interference cajas (barras apiladas por imagen, sample)
    ax2 = axes[1]
    rng = np.random.default_rng(SEED)
    muestra = hunter.sample(n=min(80, len(hunter)), random_state=SEED).sort_values(
        "n_boxes_interf", ascending=False
    ).reset_index(drop=True)

    xs = np.arange(len(muestra))
    ax2.bar(xs, muestra["n_boxes_interf"], color="#9E9E9E", label="interference", zorder=3)
    ax2.bar(xs, muestra["n_boxes_drone"],  color="#E53935", bottom=muestra["n_boxes_interf"],
            label="drone", zorder=3)

    ax2.set_xlabel("Imagen (muestra de 80, ordenada por cajas interference)", fontsize=9)
    ax2.set_ylabel("N.° de cajas predichas", fontsize=10)
    ax2.set_title("Hunter: composicion de predicciones\npor imagen (muestra)", fontsize=11)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=9)
    ax2.set_xticks([])

    fig.suptitle(
        "Hunter: el detector responde a la senal pero la clasifica como 'interference'",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardada: {output_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    FIGURAS.mkdir(parents=True, exist_ok=True)

    if not REPORT_CSV.exists():
        raise FileNotFoundError(
            f"No existe {REPORT_CSV}.\n"
            "Ejecuta primero: uv run python scripts/infer_zeroshot_campo.py"
        )

    df = pd.read_csv(REPORT_CSV)
    print(f"Cargadas {len(df)} filas de {REPORT_CSV}")
    print()

    print("Generando figuras...")

    resultados_sweep = fig_sweep_k(df, FIGURAS / "zeroshot_sweep_k.png")
    fig_por_captura(df, FIGURAS / "zeroshot_por_captura.png")
    fig_hunter_analisis(df, FIGURAS / "zeroshot_hunter_analisis.png")

    # Exportar tabla del sweep K a JSON para la memoria
    sweep_path = DIR_CAMPO / "artifacts" / "zeroshot" / "zeroshot_sweep_k.json"
    with open(sweep_path, "w", encoding="utf-8") as f:
        json.dump(resultados_sweep, f, indent=2)
    print(f"  Tabla sweep K: {sweep_path}")

    # Imprimir tabla en consola
    print()
    print("=" * 55)
    print("SWEEP K  (conf=0.10)  —  Recall de presencia (%)")
    print("=" * 55)
    header = f"{'Tipo':<12}" + "".join(f"  K={k}" for k in [1, 2, 3, 4, 5])
    print(header)
    print("-" * 55)
    for tipo, label in [
        ("mini3",  "Mini3 (ref)"),
        ("f450",   "F450"),
        ("mavic",  "Mavic Pro"),
        ("hunter", "Hunter"),
    ]:
        fila = f"{label:<12}" + "".join(
            f"  {resultados_sweep[tipo][k]:>4.1f}" for k in [1, 2, 3, 4, 5]
        )
        print(fila)
    print("=" * 55)

    print()
    print("Capturas Mavic con recall 0% (senhal no visible):")
    mavic = df[df["drone_type"] == "mavic"]
    for cap, grp in mavic.groupby("capture_id"):
        r = 100.0 * grp["detectado"].sum() / len(grp)
        if r < 5:
            print(f"  {cap}: {r:.1f}%  (n={len(grp)})")

    print()
    print("Hunter — estadisticas de cajas:")
    h = df[df["drone_type"] == "hunter"]
    print(f"  Cajas interference: media={h['n_boxes_interf'].mean():.1f}, "
          f"mediana={h['n_boxes_interf'].median():.0f}, "
          f"max={h['n_boxes_interf'].max()}")
    print(f"  Cajas drone:        media={h['n_boxes_drone'].mean():.2f}, "
          f"max={h['n_boxes_drone'].max()}")
    print(f"  Imagenes con >=1 caja interference: "
          f"{(h['n_boxes_interf'] >= 1).sum()}/{len(h)}")

    print()
    print(f"Figuras guardadas en: {FIGURAS.resolve()}")


if __name__ == "__main__":
    main()
