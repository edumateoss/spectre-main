"""
Evaluacion detallada del detector binario drone/interference.

Uso:
    uv run python scripts/evaluate_detector.py
    uv run python scripts/evaluate_detector.py --split val
    uv run python scripts/evaluate_detector.py --checkpoint ruta/al/checkpoint.pth

Salidas (en --artifacts-dir/eval_<split>/):
    predictions.csv          prediccion y probabilidad por muestra
    evaluation_report.json   metricas globales + desglose por subgrupo
    confusion_matrix.png     matriz de confusion normalizada por fila
    roc_curve.png            curva ROC con AUC
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # sin display interactivo en scripts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torchvision import models
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from common import CLASS_TO_IDX, IDX_TO_CLASS, make_loader

_DATA_DIR     = Path("data/mini3_detector_python_v1")
_METADATA_DIR = _DATA_DIR / "metadata"
_ARTIFACTS    = _DATA_DIR / "artifacts"

_CLASS_NAMES = [IDX_TO_CLASS[0], IDX_TO_CLASS[1]]  # ["interference", "drone"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluacion del detector de drones")
    p.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Particion a evaluar",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=_ARTIFACTS / "best_model.pth",
        help="Ruta al checkpoint .pth",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--no-freq-normalize",
        action="store_true",
        default=False,
        help="Desactivar FrequencyNormalize (usar con modelos entrenados sin ella)",
    )
    p.add_argument("--metadata-dir", type=Path, default=_METADATA_DIR)
    p.add_argument("--data-root", type=Path, default=_DATA_DIR)
    p.add_argument("--artifacts-dir", type=Path, default=_ARTIFACTS)
    return p.parse_args()


def load_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Reconstruye ResNet18 binario y carga los pesos del checkpoint."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    epoch = checkpoint.get("epoch", "?")
    val_f1 = checkpoint.get("val_f1", "?")
    print(f"Checkpoint cargado: epoch={epoch}, val_f1={val_f1}")
    return model


def run_inference(
    model: nn.Module,
    csv_path: Path,
    data_root: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    freq_normalize: bool = True,
) -> pd.DataFrame:
    """
    Ejecuta inferencia sobre todo el CSV y devuelve un DataFrame con
    metadatos originales + columnas: y_true, y_pred, prob_drone.
    """
    loader = make_loader(
        csv_path, data_root, train=False,
        batch_size=batch_size, num_workers=num_workers,
        freq_normalize=freq_normalize,
    )
    meta_df = pd.read_csv(csv_path).reset_index(drop=True)

    all_preds: list[int]   = []
    all_probs: list[float] = []

    with torch.no_grad():
        for images, _ in tqdm(loader, desc="inferencia"):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs  = torch.softmax(logits, dim=1)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_probs.extend(probs[:, CLASS_TO_IDX["drone"]].cpu().numpy())

    meta_df["y_true"]    = meta_df["label"].map(CLASS_TO_IDX)
    meta_df["y_pred"]    = all_preds
    meta_df["prob_drone"] = all_probs
    meta_df["correct"]   = meta_df["y_true"] == meta_df["y_pred"]
    meta_df["pred_label"] = meta_df["y_pred"].map(IDX_TO_CLASS)
    return meta_df


def _metrics_dict(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Calcula el conjunto completo de metricas de clasificacion binaria."""
    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None
    return {
        "n":         int(len(y_true)),
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "f1":        round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "auc_roc":   round(auc, 4) if auc is not None else None,
    }


def compute_breakdown(df: pd.DataFrame, group_col: str) -> list[dict]:
    """Metricas por cada valor unico de group_col, ordenadas por el valor."""
    rows = []
    for val, grp in df.groupby(group_col, sort=True):
        m = _metrics_dict(
            grp["y_true"].values,
            grp["y_pred"].values,
            grp["prob_drone"].values,
        )
        rows.append({group_col: str(val), **m})
    return rows


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    """Matriz de confusion normalizada por fila (recall por clase)."""
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    cm_abs = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=False,
        fmt=".2f",
        cmap="Blues",
        xticklabels=_CLASS_NAMES,
        yticklabels=_CLASS_NAMES,
        vmin=0.0,
        vmax=1.0,
        ax=ax,
    )
    # Anotaciones con porcentaje y conteo absoluto
    for i in range(len(_CLASS_NAMES)):
        for j in range(len(_CLASS_NAMES)):
            ax.text(
                j + 0.5, i + 0.5,
                f"{cm[i, j]:.2f}\n({cm_abs[i, j]})",
                ha="center", va="center",
                fontsize=10,
                color="white" if cm[i, j] > 0.6 else "black",
            )

    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Verdadero")
    ax.set_title("Matriz de confusion (normalizada por fila)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix guardada: {output_path}")


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path,
) -> None:
    """Curva ROC con AUC para la clase positiva (drone=1)."""
    fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=1)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.01])
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Curva ROC — detector drone vs interference")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Curva ROC guardada: {output_path}")


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    csv_path = args.metadata_dir / f"{args.split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No existe {csv_path}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"No existe checkpoint {args.checkpoint}")

    out_dir = args.artifacts_dir / f"eval_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Inferencia
    freq_normalize = not args.no_freq_normalize
    print(f"FrequencyNormalize: {'activado' if freq_normalize else 'desactivado'}")

    model = load_model(args.checkpoint, device)
    df = run_inference(
        model, csv_path, args.data_root,
        args.batch_size, args.num_workers, device,
        freq_normalize=freq_normalize,
    )

    y_true = df["y_true"].values
    y_pred = df["y_pred"].values
    y_prob = df["prob_drone"].values

    # Metricas globales
    overall = _metrics_dict(y_true, y_pred, y_prob)
    print(f"\nMetricas globales ({args.split}):")
    for k, v in overall.items():
        print(f"  {k}: {v}")

    # Desgloses
    breakdown_cols = ["label", "distance", "antenna", "environment"]
    breakdowns: dict[str, list[dict]] = {}
    for col in breakdown_cols:
        if col in df.columns and df[col].nunique() > 1:
            breakdowns[f"por_{col}"] = compute_breakdown(df, col)

    # Desglose label x distance (muestra la correlacion experimental)
    if "distance" in df.columns:
        rows = []
        for (lbl, dist), grp in df.groupby(["label", "distance"], sort=True):
            m = _metrics_dict(
                grp["y_true"].values,
                grp["y_pred"].values,
                grp["prob_drone"].values,
            )
            rows.append({"label": lbl, "distance": int(dist), **m})
        breakdowns["por_label_distance"] = rows

    # Desglose por capture_id (precision por captura)
    cap_rows = []
    for cap_id, grp in df.groupby("capture_id", sort=True):
        lbl = grp["label"].iloc[0]
        n_correct = int(grp["correct"].sum())
        n_total   = len(grp)
        cap_rows.append({
            "capture_id": cap_id,
            "label": lbl,
            "n": n_total,
            "n_correct": n_correct,
            "accuracy": round(n_correct / n_total, 4),
        })
    breakdowns["por_capture_id"] = cap_rows

    # Guardar reporte
    report = {"split": args.split, "overall": overall, **breakdowns}
    report_path = out_dir / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReporte guardado: {report_path}")

    # Guardar predicciones por muestra
    preds_path = out_dir / "predictions.csv"
    df.to_csv(preds_path, index=False)
    print(f"Predicciones guardadas: {preds_path}")

    # Graficas
    plot_confusion_matrix(y_true, y_pred, out_dir / "confusion_matrix.png")

    if len(np.unique(y_true)) > 1:
        plot_roc_curve(y_true, y_prob, out_dir / "roc_curve.png")
    else:
        print("[WARN] Solo una clase en la particion, ROC no disponible.")

    # Imprimir desgloses relevantes
    if "por_label_distance" in breakdowns:
        print("\nDesglose por label x distance:")
        for row in breakdowns["por_label_distance"]:
            print(f"  {row['label']:15s} {row['distance']}m  "
                  f"n={row['n']:4d}  acc={row['accuracy']:.4f}  f1={row['f1']:.4f}")

    print(f"\nArtifactos de evaluacion en: {out_dir}")


if __name__ == "__main__":
    main()
