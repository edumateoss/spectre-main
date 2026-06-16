"""
evaluate_rfuav.py
=================

Evalua el clasificador entrenado por train_rfuav_classifier.py sobre el split
de validacion de RFUAV. Genera:

  - confusion_matrix.png        (matriz 37x37 normalizada por fila)
  - confusion_matrix.csv        (matriz cruda, conteos absolutos)
  - per_class_metrics.csv       (precision, recall, F1, soporte por clase)
  - predictions.csv             (path, true_class, pred_class, max_softmax, top5)
  - evaluation_report.json      (top1, top5, macro-F1, weighted-F1)

Pone especial atencion en el cluster de drones DJI (MINI3, MINI4, AVATA2,
MAVIC3, FPV COMBO) que es el grupo mas relevante para nuestro TFG.

Uso:
    uv run python scripts/evaluate_rfuav.py \\
        --data-root "D:/datasets/rfuav/ImageSet-AllDrones-MatlabPipeline" \\
        --checkpoint data/rfuav_classifier/artifacts/best_model.pth
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import FrequencyNormalize, set_seed

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

DJI_DRONE_CLASSES = ["DJI AVATA2", "DJI FPV COMBO", "DJI MAVIC3 PRO", "DJI MINI3", "DJI MINI4 PRO"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=Path(r"D:/datasets/rfuav/ImageSet-AllDrones-MatlabPipeline"))
    p.add_argument("--checkpoint", type=Path, default=Path("data/rfuav_classifier/artifacts/best_model.pth"))
    p.add_argument("--out-dir", type=Path, default=Path("data/rfuav_classifier/artifacts/eval"))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_transforms(image_size: int, freq_normalize: bool) -> transforms.Compose:
    steps = []
    if freq_normalize:
        steps.append(FrequencyNormalize(window_frac=0.5))
    steps += [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ]
    return transforms.Compose(steps)


def build_model_from_checkpoint(checkpoint: dict, device: torch.device) -> torch.nn.Module:
    num_classes = checkpoint["num_classes"]
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], out_path: Path) -> None:
    """
    Matriz 37x37 normalizada por fila (porcentaje de aciertos por clase real).
    """
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, row_sums, where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Clase real")
    ax.set_title("Matriz de confusion RFUAV (normalizada por fila)")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_dji_subset_cm(cm: np.ndarray, class_names: list[str], out_path: Path) -> None:
    """
    Submatriz solo con las clases DJI, util para ver separabilidad entre modelos
    DJI (que comparten OcuSync con variantes).
    """
    indices = [class_names.index(c) for c in DJI_DRONE_CLASSES if c in class_names]
    sub = cm[np.ix_(indices, indices)].astype(float)
    sub_norm = sub / np.maximum(sub.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(sub_norm, cmap="viridis", vmin=0, vmax=1)
    labels = [class_names[i] for i in indices]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Clase real")
    ax.set_title("Matriz de confusion: solo familia DJI")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{sub_norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if sub_norm[i, j] < 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cargando checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model_from_checkpoint(checkpoint, device)

    image_size = checkpoint.get("image_size", 224)
    freq_normalize = checkpoint.get("freq_normalize", False)
    print(f"image_size={image_size}  freq_normalize={freq_normalize}")

    val_dir = args.data_root / "valid"
    if not val_dir.exists():
        raise FileNotFoundError(val_dir)

    val_ds = ImageFolder(str(val_dir), transform=build_transforms(image_size, freq_normalize))
    class_names = val_ds.classes
    print(f"Clases: {len(class_names)}")
    print(f"Imagenes de validacion: {len(val_ds)}")

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    # ---------------------------------------------------------------------
    # Inferencia sobre todo el split de validacion
    # ---------------------------------------------------------------------
    all_paths = [p for p, _ in val_ds.samples]
    all_true: list[int] = []
    all_pred: list[int] = []
    all_max_softmax: list[float] = []
    all_top5: list[list[int]] = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Inferencia"):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            max_p, _ = probs.max(dim=1)
            preds_top1 = logits.argmax(dim=1)
            preds_top5 = logits.topk(min(5, logits.size(1)), dim=1).indices

            all_true.extend(labels.cpu().numpy().tolist())
            all_pred.extend(preds_top1.cpu().numpy().tolist())
            all_max_softmax.extend(max_p.cpu().numpy().tolist())
            all_top5.extend(preds_top5.cpu().numpy().tolist())

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)

    # ---------------------------------------------------------------------
    # Metricas agregadas
    # ---------------------------------------------------------------------
    top1 = float((y_true == y_pred).mean())
    top5 = float(np.mean([t in top5 for t, top5 in zip(y_true, all_top5)]))
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\nResultados globales sobre {len(y_true)} imagenes de validacion:")
    print(f"  Top-1:        {top1:.4f}")
    print(f"  Top-5:        {top5:.4f}")
    print(f"  Macro F1:     {f1_macro:.4f}")
    print(f"  Weighted F1:  {f1_weighted:.4f}")

    # ---------------------------------------------------------------------
    # Matriz de confusion + version DJI
    # ---------------------------------------------------------------------
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(args.out_dir / "confusion_matrix.csv")
    plot_confusion_matrix(cm, class_names, args.out_dir / "confusion_matrix.png")
    plot_dji_subset_cm(cm, class_names, args.out_dir / "confusion_matrix_dji.png")

    # ---------------------------------------------------------------------
    # Metricas por clase
    # ---------------------------------------------------------------------
    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True
    )
    rows = []
    for cls in class_names:
        r = report[cls]
        rows.append({
            "class": cls,
            "precision": round(r["precision"], 4),
            "recall": round(r["recall"], 4),
            "f1": round(r["f1-score"], 4),
            "support": int(r["support"]),
        })
    pd.DataFrame(rows).to_csv(args.out_dir / "per_class_metrics.csv", index=False)

    # Foco en DJI: imprimir tabla en pantalla
    print("\nMetricas por clase para la familia DJI:")
    print(f"  {'class':<20s}  prec   rec    f1     support")
    for cls in DJI_DRONE_CLASSES:
        if cls not in class_names:
            continue
        r = report[cls]
        print(f"  {cls:<20s}  {r['precision']:.3f}  {r['recall']:.3f}  {r['f1-score']:.3f}  {int(r['support'])}")

    # ---------------------------------------------------------------------
    # Predicciones detalladas
    # ---------------------------------------------------------------------
    pred_rows = []
    for path, t, p, mxp, top5 in zip(all_paths, y_true, y_pred, all_max_softmax, all_top5):
        pred_rows.append({
            "path": path,
            "true_class": class_names[t],
            "pred_class": class_names[p],
            "correct": int(t == p),
            "max_softmax": round(mxp, 4),
            "top5": ",".join(class_names[i] for i in top5),
        })
    pd.DataFrame(pred_rows).to_csv(args.out_dir / "predictions.csv", index=False)

    # ---------------------------------------------------------------------
    # Reporte final
    # ---------------------------------------------------------------------
    eval_report = {
        "n_samples": int(len(y_true)),
        "num_classes": len(class_names),
        "top1": round(top1, 6),
        "top5": round(top5, 6),
        "f1_macro": round(f1_macro, 6),
        "f1_weighted": round(f1_weighted, 6),
        "checkpoint": str(args.checkpoint),
    }
    with open(args.out_dir / "evaluation_report.json", "w") as f:
        json.dump(eval_report, f, indent=2)

    print(f"\nResultados guardados en: {args.out_dir}")


if __name__ == "__main__":
    main()
