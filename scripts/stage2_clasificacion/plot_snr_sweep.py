"""
plot_snr_sweep.py
=================

Genera las graficas y matrices de confusion para incluir en la memoria del
TFG, a partir de los CSVs producidos por snr_sweep_stage2.py.

Salidas en --output-dir:
    - accuracy_vs_snr.png      (dos curvas: AWGN, FHSS)
    - f1_macro_vs_snr.png      (dos curvas)
    - f1_per_class_vs_snr.png  (subplot por clase, dos curvas cada uno)
    - confusion_matrices/cm_{noise}_{snr}_norm.png para SNR clave

Uso:
    uv run python scripts/plot_snr_sweep.py
    uv run python scripts/plot_snr_sweep.py --key-snrs inf 10 0 -10
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reusar lista de clases del entrenamiento
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_stage2_classifier_mavic_split import CLASS_NAMES  # noqa: E402


DEFAULT_INPUT_DIR = Path("data/stage2_classifier_v2_mavic_split/artifacts/snr_sweep")
DEFAULT_KEY_SNRS = ["inf", "10", "0", "-10"]

# Colores estables por tipo de ruido
NOISE_COLORS = {
    "awgn": "#1f77b4",   # azul: ruido canonico
    "fhss": "#d62728",   # rojo: interferencia BT-like (mas dura)
}
NOISE_LABELS = {
    "awgn": "AWGN",
    "fhss": "FHSS-like (BT sint.)",
}


def _snr_axis_value(snr_str: str) -> float:
    """Para graficar: 'inf' lo colocamos a la derecha del SNR maximo + 5."""
    if snr_str == "inf":
        return float("inf")
    return float(snr_str)


def _prepare_axis(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Convierte los SNR a posiciones numericas en eje X, con 'inf' al final."""
    snr_strs = sorted(df["snr_db"].unique().tolist(), key=_snr_axis_value)
    # Asegurar que 'inf' va al final
    if "inf" in snr_strs:
        snr_strs.remove("inf")
        snr_strs.append("inf")
    # Eje X numerico: float(snr) para finitos, max+5 para inf
    finite = [float(s) for s in snr_strs if s != "inf"]
    inf_pos = (max(finite) + 5.0) if finite else 25.0
    x = np.array([inf_pos if s == "inf" else float(s) for s in snr_strs])
    return x, snr_strs


def plot_metric_vs_snr(
    sweep_df: pd.DataFrame,
    metric: str,
    output_path: Path,
    ylabel: str,
    title: str,
) -> None:
    x, snr_strs = _prepare_axis(sweep_df)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for noise_type in sweep_df["noise_type"].unique():
        sub = sweep_df[sweep_df["noise_type"] == noise_type].copy()
        # Reindexar para que las x salgan en el mismo orden
        sub = sub.set_index("snr_db").reindex(snr_strs).reset_index()
        y = sub[metric].to_numpy()
        ax.plot(
            x, y,
            marker="o",
            linewidth=2,
            color=NOISE_COLORS.get(noise_type, "gray"),
            label=NOISE_LABELS.get(noise_type, noise_type),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(snr_strs)
    ax.set_xlabel("SNR (dB)  -  'inf' = sin ruido")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"  guardado {output_path}")


def plot_f1_per_class(sweep_df: pd.DataFrame, output_path: Path) -> None:
    x, snr_strs = _prepare_axis(sweep_df)
    n_classes = len(CLASS_NAMES)
    ncols = 3
    nrows = int(math.ceil(n_classes / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.0 * nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)

    for i, cls in enumerate(CLASS_NAMES):
        ax = axes[i]
        col = f"f1_{cls}"
        for noise_type in sweep_df["noise_type"].unique():
            sub = sweep_df[sweep_df["noise_type"] == noise_type].copy()
            sub = sub.set_index("snr_db").reindex(snr_strs).reset_index()
            y = sub[col].to_numpy()
            ax.plot(
                x, y, marker="o", linewidth=1.6,
                color=NOISE_COLORS.get(noise_type, "gray"),
                label=NOISE_LABELS.get(noise_type, noise_type),
            )
        ax.set_title(cls)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        if i % ncols == 0:
            ax.set_ylabel("F1")
        if i // ncols == nrows - 1:
            ax.set_xticks(x)
            ax.set_xticklabels(snr_strs, rotation=0)
            ax.set_xlabel("SNR (dB)")

    # Apagar axes vacios sobrantes
    for j in range(n_classes, len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  guardado {output_path}")


def plot_confusion(cm_csv: Path, output_png: Path, title: str) -> None:
    cm_df = pd.read_csv(cm_csv, index_col=0)
    cm = cm_df.to_numpy()
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm_norm, interpolation="nearest", vmin=0, vmax=1, cmap="viridis")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(CLASS_NAMES)))
    ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(title)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm_norm[i, j]:.2f}\n({cm[i, j]})"
            color = "white" if cm_norm[i, j] < 0.5 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)

    fig.tight_layout()
    fig.savefig(output_png, dpi=200)
    plt.close(fig)
    print(f"  guardado {output_png}")


def main() -> None:
    p = argparse.ArgumentParser(description="Graficas del barrido SNR Stage 2")
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    p.add_argument("--output-dir", type=Path, default=None, help="Por defecto, igual a --input-dir.")
    p.add_argument("--key-snrs", nargs="+", default=DEFAULT_KEY_SNRS,
                   help="SNR a usar para graficar matrices de confusion.")
    args = p.parse_args()

    in_dir = args.input_dir
    out_dir = args.output_dir or in_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "confusion_matrices").mkdir(parents=True, exist_ok=True)

    sweep_csv = in_dir / "snr_sweep_results.csv"
    sweep_df = pd.read_csv(sweep_csv)
    sweep_df["snr_db"] = sweep_df["snr_db"].astype(str)
    print(f"Cargado {sweep_csv}: {len(sweep_df)} filas")

    print("\nGenerando graficas resumen:")
    plot_metric_vs_snr(
        sweep_df, "accuracy",
        out_dir / "accuracy_vs_snr.png",
        ylabel="Accuracy",
        title="Accuracy vs SNR - Stage 2 (6 clases)",
    )
    plot_metric_vs_snr(
        sweep_df, "f1_macro",
        out_dir / "f1_macro_vs_snr.png",
        ylabel="F1 macro",
        title="Macro-F1 vs SNR - Stage 2 (6 clases)",
    )
    plot_f1_per_class(
        sweep_df,
        out_dir / "f1_per_class_vs_snr.png",
    )

    print("\nMatrices de confusion en SNRs clave:")
    for noise_type in sweep_df["noise_type"].unique():
        for snr_str in args.key_snrs:
            cm_csv = in_dir / "confusion_matrices" / f"cm_{noise_type}_{snr_str}.csv"
            if not cm_csv.exists():
                print(f"  [skip] no existe {cm_csv}")
                continue
            out_png = out_dir / "confusion_matrices" / f"cm_{noise_type}_{snr_str}_norm.png"
            title = f"Matriz de confusion - {NOISE_LABELS.get(noise_type, noise_type)} - SNR={snr_str} dB"
            plot_confusion(cm_csv, out_png, title)

    print(f"\nListo. Outputs en: {out_dir}")


if __name__ == "__main__":
    main()
