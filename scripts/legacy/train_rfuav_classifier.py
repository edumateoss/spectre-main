"""
train_rfuav_classifier.py
=========================

Entrena un clasificador multi-clase (37 modelos de dron/emisora) sobre el
dataset RFUAV (ImageSet-AllDrones-MatlabPipeline).

Estrategia del TFG:
  - Aprovechar el benchmark publicado RFUAV como modelo de referencia.
  - El clasificador aprende la firma RF de cada modelo de dron sobre datos
    perfectamente etiquetados (RFUAV ya viene organizado en carpetas por clase).
  - El modelo resultante se aplicara despues a NUESTRAS capturas de DJI MINI3
    (cross-dataset validation) y a las capturas de interferencia (outlier
    detection para resolver el problema dron-vs-interferencia).

Estructura esperada del dataset (D:\\datasets\\rfuav\\ImageSet-AllDrones-MatlabPipeline):
    train/
        DJI MINI3/         (470 imagenes)
        DJI MAVIC3 PRO/    (335 imagenes)
        ... 37 clases en total
    valid/
        DJI MINI3/         (1109 imagenes)
        DJI MAVIC3 PRO/    (791 imagenes)
        ... mismas 37 clases

Salidas (en --artifacts-dir):
    best_model.pth         checkpoint con mejor val_top1
    last_model.pth         checkpoint del ultimo epoch
    metrics.csv            metricas por epoch (train y val)
    run_config.json        hiperparametros del experimento
    results.json           resumen final
    class_mapping.json     mapeo de indice -> nombre de clase (ordenado por ImageFolder)

Uso tipico (Windows, GPU RTX 4070):
    uv run python scripts/train_rfuav_classifier.py \\
        --data-root "D:/datasets/rfuav/ImageSet-AllDrones-MatlabPipeline" \\
        --epochs 20 --batch-size 64 --freeze-backbone 3
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from common import set_seed, FrequencyNormalize

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_DEFAULT_ROOT = Path(r"D:/datasets/rfuav/ImageSet-AllDrones-MatlabPipeline")
_DEFAULT_ARTIFACTS = Path("data/rfuav_classifier/artifacts")


# ---------------------------------------------------------------------------
# Argumentos
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=_DEFAULT_ROOT,
                   help="Raiz del dataset RFUAV (con subcarpetas train/ y valid/).")
    p.add_argument("--artifacts-dir", type=Path, default=_DEFAULT_ARTIFACTS)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0,
                   help="Workers del DataLoader. 0 = principal (estable en Windows).")
    p.add_argument("--image-size", type=int, default=224,
                   help="Tamano al que se reescalan las imagenes antes de la red.")
    p.add_argument(
        "--freeze-backbone",
        type=int,
        default=3,
        help="Cuantos grupos del backbone se congelan (0-5).",
    )
    p.add_argument(
        "--freq-normalize",
        action="store_true",
        default=False,
        help=(
            "Aplicar FrequencyNormalize antes del Resize. Recorta la banda "
            "de mayor energia para reducir dependencia de la Fc absoluta."
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Transformaciones
# ---------------------------------------------------------------------------
def build_transforms(image_size: int, train: bool, freq_normalize: bool) -> transforms.Compose:
    """
    Pipeline de transformaciones.

    Sin flips ni rotaciones (los ejes tiempo y frecuencia tienen significado
    fisico). Augmentations muy suaves en train para no romper la firma.
    """
    steps = []
    if freq_normalize:
        steps.append(FrequencyNormalize(window_frac=0.5))
    steps.append(transforms.Resize((image_size, image_size)))
    if train:
        steps += [
            transforms.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.95, 1.05)),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        ]
    steps += [
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ]
    return transforms.Compose(steps)


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
def build_model(num_classes: int, freeze_backbone: int) -> nn.Module:
    """
    ResNet18 preentrenado en ImageNet con cabeza de num_classes outputs.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

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
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"Parametros entrenables: {n_trainable:,} / {n_total:,} "
        f"({100 * n_trainable / n_total:.1f}%)"
    )
    return model


# ---------------------------------------------------------------------------
# Bucle de epoch
# ---------------------------------------------------------------------------
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[AdamW],
    device: torch.device,
    train: bool,
) -> dict[str, float]:
    """
    Ejecuta un epoch y devuelve loss, top1, top5 y macro F1.
    """
    model.train(train)

    total_loss = 0.0
    n_samples = 0
    all_preds: list[int] = []
    all_labels: list[int] = []
    n_top5_correct = 0

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

            bs = labels.size(0)
            total_loss += loss.item() * bs
            n_samples += bs

            preds_top1 = logits.argmax(dim=1)
            all_preds.extend(preds_top1.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

            # top-5 accuracy
            preds_top5 = logits.topk(min(5, logits.size(1)), dim=1).indices  # (bs, 5)
            n_top5_correct += (preds_top5 == labels.unsqueeze(1)).any(dim=1).sum().item()

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    return {
        "loss": total_loss / max(n_samples, 1),
        "top1": float(accuracy_score(y_true, y_pred)),
        "top5": n_top5_correct / max(n_samples, 1),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_dir = args.data_root / "train"
    val_dir = args.data_root / "valid"
    for d in [train_dir, val_dir]:
        if not d.exists():
            raise FileNotFoundError(f"No existe {d}")

    print(f"Train dir: {train_dir}")
    print(f"Val dir:   {val_dir}")

    # ImageFolder construye automaticamente el mapeo carpeta -> indice.
    train_ds = ImageFolder(str(train_dir), transform=build_transforms(args.image_size, True, args.freq_normalize))
    val_ds   = ImageFolder(str(val_dir),   transform=build_transforms(args.image_size, False, args.freq_normalize))

    num_classes = len(train_ds.classes)
    print(f"\nClases detectadas en train: {num_classes}")
    if set(train_ds.classes) != set(val_ds.classes):
        raise ValueError(
            "Los conjuntos de clases de train y val no coinciden.\n"
            f"  Solo en train: {set(train_ds.classes) - set(val_ds.classes)}\n"
            f"  Solo en val:   {set(val_ds.classes) - set(train_ds.classes)}"
        )

    # Guardar mapeo clase -> indice para inferencia posterior
    class_mapping = {idx: name for name, idx in train_ds.class_to_idx.items()}
    with open(args.artifacts_dir / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump(class_mapping, f, indent=2, ensure_ascii=False)
    print(f"Mapeo de clases guardado en {args.artifacts_dir / 'class_mapping.json'}")

    print(f"\nTrain: {len(train_ds)} muestras | Val: {len(val_ds)} muestras")
    counts = np.bincount([y for _, y in train_ds.samples], minlength=num_classes)
    print("Distribucion en train (las 5 clases con menos / mas muestras):")
    sorted_idx = np.argsort(counts)
    for i in list(sorted_idx[:3]) + list(sorted_idx[-3:]):
        print(f"  {counts[i]:5d}  {class_mapping[i]}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    model = build_model(num_classes, args.freeze_backbone).to(device)

    # Pesos de clase inversamente proporcionales a frecuencia (compensa
    # desbalanceo entre las 37 clases).
    class_weights = torch.tensor(
        len(train_ds) / (num_classes * np.maximum(counts, 1)),
        dtype=torch.float32,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config["num_classes"] = num_classes
    with open(args.artifacts_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    metrics_csv = args.artifacts_dir / "metrics.csv"
    metrics_fields = ["epoch", "phase", "loss", "top1", "top5", "f1_macro", "lr"]
    with open(metrics_csv, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=metrics_fields).writeheader()

    best_val_top1 = -1.0
    best_epoch = -1

    print(
        f"\nIniciando entrenamiento: {args.epochs} epochs | "
        f"freeze_backbone={args.freeze_backbone} | lr={args.lr} | batch_size={args.batch_size}\n"
    )

    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        train_m = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_m = run_epoch(model, val_loader, criterion, None, device, train=False)

        scheduler.step()

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train  loss={train_m['loss']:.4f} top1={train_m['top1']:.4f} f1={train_m['f1_macro']:.4f} | "
            f"val    loss={val_m['loss']:.4f} top1={val_m['top1']:.4f} top5={val_m['top5']:.4f} f1={val_m['f1_macro']:.4f}"
        )

        with open(metrics_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics_fields)
            for phase, m in [("train", train_m), ("val", val_m)]:
                writer.writerow({
                    "epoch": epoch,
                    "phase": phase,
                    "lr": round(current_lr, 8),
                    **{k: round(v, 6) for k, v in m.items()},
                })

        if val_m["top1"] > best_val_top1:
            best_val_top1 = val_m["top1"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_top1": best_val_top1,
                    "val_top5": val_m["top5"],
                    "val_f1_macro": val_m["f1_macro"],
                    "class_mapping": class_mapping,
                    "num_classes": num_classes,
                    "image_size": args.image_size,
                    "freq_normalize": args.freq_normalize,
                },
                args.artifacts_dir / "best_model.pth",
            )
            print(f"  -> Nuevo mejor modelo (val_top1={best_val_top1:.4f})")

    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "class_mapping": class_mapping,
            "num_classes": num_classes,
            "image_size": args.image_size,
            "freq_normalize": args.freq_normalize,
        },
        args.artifacts_dir / "last_model.pth",
    )

    print(f"\nEntrenamiento completado. Mejor epoch: {best_epoch} (val_top1={best_val_top1:.4f})")

    results = {
        "best_epoch": best_epoch,
        "best_val_top1": round(best_val_top1, 6),
        "num_classes": num_classes,
        "config": config,
    }
    with open(args.artifacts_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nArtefactos guardados en: {args.artifacts_dir}")
    print("Siguiente paso: scripts/evaluate_rfuav.py (matriz de confusion 37x37)")
    print("y scripts/cross_validate_mini3.py (aplicar el modelo a nuestras capturas).")


if __name__ == "__main__":
    main()
