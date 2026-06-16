"""
Entrena un detector binario drone/interference basado en ResNet18.

Uso:
    uv run python scripts/train_detector.py
    uv run python scripts/train_detector.py --epochs 30 --freeze-backbone 2

Salidas (en --artifacts-dir):
    best_model.pth    checkpoint con mejor val_f1
    last_model.pth    checkpoint del ultimo epoch
    metrics.csv       metricas por epoch (train y val)
    run_config.json   hiperparametros del experimento
    results.json      resumen final con metricas en test
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from common import CLASS_TO_IDX, make_loader, set_seed

_DATA_DIR     = Path("data/mini3_detector_python_v1")
_METADATA_DIR = _DATA_DIR / "metadata"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrenamiento detector binario drones")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument(
        "--freeze-backbone",
        type=int,
        default=3,
        help="Numero de grupos del backbone a congelar: stem, layer1, layer2, layer3, layer4 (0-5)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Workers del DataLoader. 0 = proceso principal (recomendado en Windows)",
    )
    p.add_argument("--metadata-dir", type=Path, default=_METADATA_DIR)
    p.add_argument("--data-root", type=Path, default=_DATA_DIR)
    p.add_argument("--artifacts-dir", type=Path, default=_DATA_DIR / "artifacts")
    p.add_argument(
        "--no-freq-normalize",
        action="store_true",
        default=False,
        help="Desactivar FrequencyNormalize (baseline sin normalizacion de frecuencia)",
    )

    p.add_argument(
        "--power-ablation",
        action="store_true",
        default=False,
        help="Activar PowerAblation: elimina/reduce informacion de potencia para estudiar si el modelo depende de la intensidad.",
    )
    p.add_argument(
        "--no-eval-test",
        action="store_true",
        default=False,
        help="Omitir evaluacion en test al finalizar",
    )
    return p.parse_args()


def build_model(freeze_backbone: int) -> nn.Module:
    """
    ResNet18 preentrenado en ImageNet con cabeza binaria (interference=0, drone=1).
    Congela los primeros freeze_backbone grupos del backbone.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)

    # Grupos de menor a mayor nivel semantico
    backbone_groups = [
        [model.conv1, model.bn1],
        [model.layer1],
        [model.layer2],
        [model.layer3],
        [model.layer4],
    ]
    for group in backbone_groups[:freeze_backbone]:
        for module in group:
            for param in module.parameters():
                param.requires_grad = False

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(
        f"Parametros entrenables: {n_trainable:,} / {n_total:,} "
        f"({100 * n_trainable / n_total:.1f}%)"
    )
    return model


def compute_class_weights(train_csv: Path) -> torch.Tensor:
    """Pesos inversamente proporcionales a la frecuencia de clase en train."""
    df = pd.read_csv(train_csv)
    counts = df["label"].map(CLASS_TO_IDX).value_counts().sort_index()
    weights = len(df) / (len(CLASS_TO_IDX) * counts.values)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: Optional[AdamW],
    device: torch.device,
    train: bool,
) -> dict[str, float]:
    """Ejecuta un epoch. Devuelve loss, accuracy, f1, precision y recall."""
    model.train(train)

    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    desc = "train" if train else "  val"
    with torch.set_grad_enabled(train):
        for images, labels in tqdm(loader, leave=False, desc=desc):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    return {
        "loss":      total_loss / len(y_true),
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "f1":        float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_csv = args.metadata_dir / "train.csv"
    val_csv   = args.metadata_dir / "val.csv"
    test_csv  = args.metadata_dir / "test.csv"

    for p in [train_csv, val_csv, test_csv]:
        if not p.exists():
            raise FileNotFoundError(
                f"No existe {p}. Ejecuta split_capture_level.py primero."
            )

    freq_normalize = not args.no_freq_normalize
    print(f"FrequencyNormalize: {'activado' if freq_normalize else 'desactivado'}")

    train_loader = make_loader(
        train_csv, args.data_root, train=True,
        batch_size=args.batch_size, num_workers=args.num_workers,
        freq_normalize=freq_normalize,
    )
    val_loader = make_loader(
        val_csv, args.data_root, train=False,
        batch_size=args.batch_size, num_workers=args.num_workers,
        freq_normalize=freq_normalize,
    )
    print(
        f"\nTrain: {len(train_loader.dataset)} muestras | "
        f"Val: {len(val_loader.dataset)} muestras"
    )

    model = build_model(args.freeze_backbone).to(device)

    class_weights = compute_class_weights(train_csv).to(device)
    print(
        f"Pesos de clase: interference={class_weights[0]:.3f}, "
        f"drone={class_weights[1]:.3f}"
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Guardar configuracion del experimento
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    with open(args.artifacts_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    metrics_csv    = args.artifacts_dir / "metrics.csv"
    metrics_fields = ["epoch", "phase", "loss", "accuracy", "f1", "precision", "recall", "lr"]
    with open(metrics_csv, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=metrics_fields).writeheader()

    best_val_f1 = -1.0
    best_epoch  = -1

    print(
        f"\nIniciando entrenamiento: {args.epochs} epochs | "
        f"freeze_backbone={args.freeze_backbone} | lr={args.lr}\n"
    )

    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        train_m = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_m   = run_epoch(model, val_loader,   criterion, None,      device, train=False)

        scheduler.step()

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train  loss={train_m['loss']:.4f} acc={train_m['accuracy']:.4f} f1={train_m['f1']:.4f} | "
            f"val    loss={val_m['loss']:.4f} acc={val_m['accuracy']:.4f} f1={val_m['f1']:.4f}"
        )

        with open(metrics_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics_fields)
            for phase, m in [("train", train_m), ("val", val_m)]:
                writer.writerow({
                    "epoch": epoch,
                    "phase": phase,
                    "lr":    round(current_lr, 8),
                    **{k: round(v, 6) for k, v in m.items()},
                })

        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            best_epoch  = epoch
            torch.save(
                {
                    "epoch":                epoch,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1":               best_val_f1,
                    "val_accuracy":         val_m["accuracy"],
                },
                args.artifacts_dir / "best_model.pth",
            )
            print(f"  -> Nuevo mejor modelo (val_f1={best_val_f1:.4f})")

    torch.save(
        {
            "epoch":                args.epochs,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        args.artifacts_dir / "last_model.pth",
    )

    print(f"\nEntrenamiento completado. Mejor epoch: {best_epoch} (val_f1={best_val_f1:.4f})")

    results: dict = {
        "best_epoch":   best_epoch,
        "best_val_f1":  round(best_val_f1, 6),
        "config":       config,
    }

    if not args.no_eval_test:
        checkpoint = torch.load(
            args.artifacts_dir / "best_model.pth", map_location=device
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        test_loader = make_loader(
            test_csv, args.data_root, train=False,
            batch_size=args.batch_size, num_workers=args.num_workers,
            freq_normalize=freq_normalize,
        )
        test_m = run_epoch(model, test_loader, criterion, None, device, train=False)

        print(f"\nTest (mejor modelo, epoch {best_epoch}):")
        for k, v in test_m.items():
            print(f"  {k}: {v:.4f}")

        results["test"] = {k: round(v, 6) for k, v in test_m.items()}

    with open(args.artifacts_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nArtifactos guardados en: {args.artifacts_dir}")


if __name__ == "__main__":
    main()
