"""
train_stage2_classifier_v2.py
=============================

Entrena un clasificador multiclase Stage 2 sobre el CSV:

    spectrograms_stage2_clean.csv

Clases concretas del experimento:

    0 = f450
    1 = hunter
    2 = mavic
    3 = mini3
    4 = interference

La idea es equivalente a train_detector.py, pero adaptada a 5 clases:
  - Lee un CSV maestro con columna filename y label.
  - Hace split por capture_id para evitar fuga de informacion.
  - Entrena ResNet18 preentrenada en ImageNet.
  - Usa pesos de clase para compensar el desbalanceo.
  - Guarda best_model.pth, last_model.pth, metrics.csv, run_config.json,
    results.json, class_mapping.json y confusion_matrix_test.png.

Uso tipico desde la raiz del proyecto:

    uv run python scripts/train_stage2_classifier_v2.py ^
        --csv data/stage2_classifier_v1/metadata/spectrograms_stage2_clean.csv ^
        --data-root . ^
        --epochs 30 ^
        --batch-size 32 ^
        --freeze-backbone 3

En PowerShell, usando backticks:

    uv run python scripts/train_stage2_classifier_v2.py `
        --csv data/stage2_classifier_v1/metadata/spectrograms_stage2_clean.csv `
        --data-root . `
        --epochs 30 `
        --batch-size 32 `
        --freeze-backbone 3
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuracion fija del experimento Stage 2
# ---------------------------------------------------------------------------

CLASS_NAMES = ["f450", "hunter", "mavic", "mini3", "interference"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEFAULT_CSV = Path("data/stage2_classifier_v1/metadata/spectrograms_stage2_clean.csv")
DEFAULT_ARTIFACTS = Path("data/stage2_classifier_v1/artifacts/resnet18_stage2_v2")


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Fija semillas para mejorar la reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class FrequencyNormalize:
    """
    Recorte vertical centrado en la banda con mayor energia media.

    En los espectrogramas RF, el eje horizontal representa el tiempo y el eje
    vertical representa frecuencia. Esta transformacion reduce la dependencia
    de la frecuencia central absoluta y obliga al modelo a mirar mas la
    morfologia de la senal.
    """

    def __init__(self, window_frac: float = 0.5) -> None:
        if not (0 < window_frac <= 1):
            raise ValueError("window_frac debe estar en (0, 1].")
        self.window_frac = window_frac

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32)
        height, width, _ = arr.shape
        crop_h = max(1, int(round(height * self.window_frac)))
        if crop_h >= height:
            return img

        # Energia por fila de frecuencia. Promediamos tiempo y canales RGB.
        row_energy = arr.mean(axis=(1, 2))
        center = int(np.argmax(row_energy))

        top = center - crop_h // 2
        top = max(0, min(top, height - crop_h))
        bottom = top + crop_h

        cropped = arr[top:bottom, :, :].astype(np.uint8)
        return Image.fromarray(cropped)


class PowerAblationNormalize:
    """
    Normalizacion opcional por imagen para reducir informacion de brillo global.

    Sirve como ablacion: si el rendimiento se mantiene, el modelo esta usando
    morfologia; si cae mucho, dependia de intensidad/potencia media.
    """

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32)
        mean = arr.mean(axis=(0, 1), keepdims=True)
        std = arr.std(axis=(0, 1), keepdims=True) + 1e-6
        arr = (arr - mean) / std
        arr = 127.5 + 40.0 * arr
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Dataset CSV
# ---------------------------------------------------------------------------

class Stage2CSVDataset(Dataset):
    """Dataset que lee imagenes desde un CSV con columnas filename y label."""

    def __init__(self, csv_path: Path, data_root: Path, transform=None) -> None:
        self.csv_path = Path(csv_path)
        self.data_root = Path(data_root)
        self.transform = transform

        if not self.csv_path.exists():
            raise FileNotFoundError(f"No existe el CSV: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path).reset_index(drop=True)
        required = {"filename", "label"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"El CSV debe contener columnas {required}. Faltan: {missing}")

        unknown = sorted(set(self.df["label"]) - set(CLASS_NAMES))
        if unknown:
            raise ValueError(
                "El CSV contiene clases no esperadas: "
                f"{unknown}. Clases validas: {CLASS_NAMES}"
            )

        self.paths = [self.data_root / str(p) for p in self.df["filename"].tolist()]
        self.labels = [CLASS_TO_IDX[str(y)] for y in self.df["label"].tolist()]

        missing_files = [p for p in self.paths if not p.exists()]
        if missing_files:
            preview = "\n".join(f"  - {p}" for p in missing_files[:10])
            raise FileNotFoundError(
                f"Hay {len(missing_files)} imagenes del CSV que no existen. "
                f"Primeros ejemplos:\n{preview}\n\n"
                "Comprueba --data-root y que las rutas de filename sean relativas a la raiz del proyecto."
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# Split por capture_id
# ---------------------------------------------------------------------------

@dataclass
class SplitPaths:
    train_csv: Path
    val_csv: Path
    test_csv: Path


def make_capture_level_splits(
    csv_path: Path,
    output_dir: Path,
    seed: int,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> SplitPaths:
    """
    Genera train/val/test por capture_id estratificando por label.

    El split no es por imagen, sino por captura completa. Asi evitamos que
    segmentos consecutivos de la misma grabacion aparezcan simultaneamente en
    entrenamiento y validacion/test.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path).reset_index(drop=True)
    required = {"filename", "label", "capture_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Para split por captura hacen falta columnas {required}. Faltan: {missing}")

    rng = random.Random(seed)
    split_by_capture: dict[str, str] = {}

    for label, sub in df.groupby("label"):
        captures = sorted(sub["capture_id"].astype(str).unique().tolist())
        rng.shuffle(captures)
        n = len(captures)
        if n < 3:
            raise ValueError(
                f"La clase {label!r} solo tiene {n} capture_id. "
                "Se necesitan al menos 3 para train/val/test."
            )

        n_train = max(1, int(round(n * train_frac)))
        n_val = max(1, int(round(n * val_frac)))
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1

        train_caps = captures[:n_train]
        val_caps = captures[n_train:n_train + n_val]
        test_caps = captures[n_train + n_val:]

        for cap in train_caps:
            split_by_capture[cap] = "train"
        for cap in val_caps:
            split_by_capture[cap] = "val"
        for cap in test_caps:
            split_by_capture[cap] = "test"

    df["split"] = df["capture_id"].astype(str).map(split_by_capture)
    if df["split"].isna().any():
        raise RuntimeError("Algunas filas quedaron sin split asignado.")

    train_df = df[df["split"] == "train"].drop(columns=["split"])
    val_df = df[df["split"] == "val"].drop(columns=["split"])
    test_df = df[df["split"] == "test"].drop(columns=["split"])

    train_csv = output_dir / "train.csv"
    val_csv = output_dir / "val.csv"
    test_csv = output_dir / "test.csv"

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print("\nSplit por capture_id generado:")
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"\n{name}: {len(part)} imagenes")
        print(part["label"].value_counts().reindex(CLASS_NAMES, fill_value=0).to_string())
        print("capture_id por clase:")
        print(part.groupby("label")["capture_id"].nunique().reindex(CLASS_NAMES, fill_value=0).to_string())

    return SplitPaths(train_csv=train_csv, val_csv=val_csv, test_csv=test_csv)


# ---------------------------------------------------------------------------
# Transformaciones y DataLoaders
# ---------------------------------------------------------------------------

def build_transforms(
    image_size: int,
    train: bool,
    freq_normalize: bool,
    power_ablation: bool,
) -> transforms.Compose:
    """
    Transformaciones conservadoras.

    No se usan flips ni rotaciones porque los ejes tiempo/frecuencia tienen
    significado fisico. Solo se permite una traslacion y escala muy pequena.
    """
    steps = []

    if freq_normalize:
        steps.append(FrequencyNormalize(window_frac=0.5))

    if power_ablation:
        steps.append(PowerAblationNormalize())

    steps.append(transforms.Resize((image_size, image_size)))

    if train:
        steps += [
            transforms.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.95, 1.05)),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.15),
        ]

    steps += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(steps)


def make_loader(
    csv_path: Path,
    data_root: Path,
    train: bool,
    batch_size: int,
    num_workers: int,
    image_size: int,
    freq_normalize: bool,
    power_ablation: bool,
    seed: int,
) -> DataLoader:
    ds = Stage2CSVDataset(
        csv_path=csv_path,
        data_root=data_root,
        transform=build_transforms(image_size, train, freq_normalize, power_ablation),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator if train else None,
    )


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

def build_model(num_classes: int, freeze_backbone: int) -> nn.Module:
    """ResNet18 preentrenada con cabeza multiclase."""
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


def compute_class_weights(train_csv: Path) -> torch.Tensor:
    """Pesos inversamente proporcionales a la frecuencia de cada clase."""
    df = pd.read_csv(train_csv)
    counts = df["label"].value_counts().reindex(CLASS_NAMES, fill_value=0).values
    weights = len(df) / (len(CLASS_NAMES) * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Entrenamiento/evaluacion
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[AdamW],
    device: torch.device,
    train: bool,
) -> dict[str, float]:
    """Ejecuta un epoch y devuelve metricas multiclase."""
    model.train(train)

    total_loss = 0.0
    n_samples = 0
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
                if optimizer is None:
                    raise RuntimeError("optimizer no puede ser None en modo train")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            bs = labels.size(0)
            total_loss += loss.item() * bs
            n_samples += bs
            all_preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    return {
        "loss": total_loss / max(n_samples, 1),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def predict_all(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for images, labels in tqdm(loader, leave=False, desc=" test"):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
    return np.array(all_labels), np.array(all_preds)


def save_confusion_matrix_png(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    """Guarda matriz de confusion en PNG usando matplotlib."""
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, interpolation="nearest", vmin=0, vmax=1)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicho",
        ylabel="Real",
        title="Matriz de confusion normalizada - Stage 2",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Argumentos
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrenamiento clasificador Stage 2 multiclase")

    p.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="CSV maestro con filename, label y capture_id.")
    p.add_argument("--data-root", type=Path, default=Path("."), help="Raiz respecto a la que se resuelven los filenames del CSV.")
    p.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)

    p.add_argument("--train-csv", type=Path, default=None, help="CSV train ya generado. Si se omite, se crea desde --csv.")
    p.add_argument("--val-csv", type=Path, default=None, help="CSV val ya generado. Si se omite, se crea desde --csv.")
    p.add_argument("--test-csv", type=Path, default=None, help="CSV test ya generado. Si se omite, se crea desde --csv.")
    p.add_argument("--splits-dir", type=Path, default=None, help="Directorio donde guardar train/val/test.csv.")

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--freeze-backbone", type=int, default=3, help="Grupos congelados del backbone: 0-5.")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0, help="0 recomendado en Windows.")

    p.add_argument("--no-freq-normalize", action="store_true", default=False, help="Desactiva FrequencyNormalize.")
    p.add_argument("--power-ablation", action="store_true", default=False, help="Reduce informacion de potencia/brillo global.")
    p.add_argument("--no-eval-test", action="store_true", default=False, help="No evaluar en test al final.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = args.splits_dir or (args.artifacts_dir / "splits")

    # Guardar mapeo de clases desde el principio.
    with open(args.artifacts_dir / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump(IDX_TO_CLASS, f, indent=2, ensure_ascii=False)

    # Crear o usar splits.
    if args.train_csv and args.val_csv and args.test_csv:
        split_paths = SplitPaths(args.train_csv, args.val_csv, args.test_csv)
    else:
        split_paths = make_capture_level_splits(
            csv_path=args.csv,
            output_dir=splits_dir,
            seed=args.seed,
        )

    freq_normalize = not args.no_freq_normalize

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDispositivo: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\nClases: {CLASS_NAMES}")
    print(f"CSV maestro: {args.csv}")
    print(f"Data root:   {args.data_root}")
    print(f"Artifacts:   {args.artifacts_dir}")
    print(f"FrequencyNormalize: {'activado' if freq_normalize else 'desactivado'}")
    print(f"Power ablation:      {'activado' if args.power_ablation else 'desactivado'}")

    train_loader = make_loader(
        split_paths.train_csv, args.data_root, train=True,
        batch_size=args.batch_size, num_workers=args.num_workers,
        image_size=args.image_size, freq_normalize=freq_normalize,
        power_ablation=args.power_ablation, seed=args.seed,
    )
    val_loader = make_loader(
        split_paths.val_csv, args.data_root, train=False,
        batch_size=args.batch_size, num_workers=args.num_workers,
        image_size=args.image_size, freq_normalize=freq_normalize,
        power_ablation=args.power_ablation, seed=args.seed,
    )

    print(f"\nTrain: {len(train_loader.dataset)} muestras")
    print(f"Val:   {len(val_loader.dataset)} muestras")

    model = build_model(num_classes=len(CLASS_NAMES), freeze_backbone=args.freeze_backbone).to(device)

    class_weights = compute_class_weights(split_paths.train_csv).to(device)
    print("\nPesos de clase:")
    for idx, w in enumerate(class_weights.detach().cpu().tolist()):
        print(f"  {IDX_TO_CLASS[idx]:<12s}: {w:.3f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({
        "classes": CLASS_NAMES,
        "class_to_idx": CLASS_TO_IDX,
        "train_csv_used": str(split_paths.train_csv),
        "val_csv_used": str(split_paths.val_csv),
        "test_csv_used": str(split_paths.test_csv),
        "freq_normalize": freq_normalize,
    })
    with open(args.artifacts_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    metrics_csv = args.artifacts_dir / "metrics.csv"
    metrics_fields = [
        "epoch", "phase", "loss", "accuracy", "f1_macro", "f1_weighted",
        "precision_macro", "recall_macro", "lr",
    ]
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=metrics_fields).writeheader()

    best_val_f1 = -1.0
    best_epoch = -1

    print(
        f"\nIniciando entrenamiento: {args.epochs} epochs | "
        f"batch_size={args.batch_size} | freeze_backbone={args.freeze_backbone} | lr={args.lr}\n"
    )

    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        train_m = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_m = run_epoch(model, val_loader, criterion, None, device, train=False)
        scheduler.step()

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train loss={train_m['loss']:.4f} acc={train_m['accuracy']:.4f} f1_macro={train_m['f1_macro']:.4f} | "
            f"val loss={val_m['loss']:.4f} acc={val_m['accuracy']:.4f} f1_macro={val_m['f1_macro']:.4f}"
        )

        with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=metrics_fields)
            for phase, m in [("train", train_m), ("val", val_m)]:
                writer.writerow({
                    "epoch": epoch,
                    "phase": phase,
                    "lr": round(current_lr, 8),
                    **{k: round(v, 6) for k, v in m.items()},
                })

        # En multiclase desbalanceado conviene elegir por F1 macro, no solo accuracy.
        if val_m["f1_macro"] > best_val_f1:
            best_val_f1 = val_m["f1_macro"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1_macro": best_val_f1,
                    "val_accuracy": val_m["accuracy"],
                    "class_mapping": IDX_TO_CLASS,
                    "class_to_idx": CLASS_TO_IDX,
                    "num_classes": len(CLASS_NAMES),
                    "image_size": args.image_size,
                    "freq_normalize": freq_normalize,
                    "power_ablation": args.power_ablation,
                },
                args.artifacts_dir / "best_model.pth",
            )
            print(f"  -> Nuevo mejor modelo (val_f1_macro={best_val_f1:.4f})")

    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "class_mapping": IDX_TO_CLASS,
            "class_to_idx": CLASS_TO_IDX,
            "num_classes": len(CLASS_NAMES),
            "image_size": args.image_size,
            "freq_normalize": freq_normalize,
            "power_ablation": args.power_ablation,
        },
        args.artifacts_dir / "last_model.pth",
    )

    results: dict = {
        "best_epoch": best_epoch,
        "best_val_f1_macro": round(best_val_f1, 6),
        "config": config,
    }

    print(f"\nEntrenamiento completado. Mejor epoch: {best_epoch} (val_f1_macro={best_val_f1:.4f})")

    if not args.no_eval_test:
        checkpoint = torch.load(args.artifacts_dir / "best_model.pth", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        test_loader = make_loader(
            split_paths.test_csv, args.data_root, train=False,
            batch_size=args.batch_size, num_workers=args.num_workers,
            image_size=args.image_size, freq_normalize=freq_normalize,
            power_ablation=args.power_ablation, seed=args.seed,
        )
        test_m = run_epoch(model, test_loader, criterion, None, device, train=False)
        y_true, y_pred = predict_all(model, test_loader, device)

        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
        cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
        cm_csv = args.artifacts_dir / "confusion_matrix_test.csv"
        cm_png = args.artifacts_dir / "confusion_matrix_test.png"
        cm_df.to_csv(cm_csv)
        save_confusion_matrix_png(y_true, y_pred, cm_png)

        print(f"\nTest (mejor modelo, epoch {best_epoch}):")
        for k, v in test_m.items():
            print(f"  {k}: {v:.4f}")
        print(f"\nMatriz de confusion guardada en:")
        print(f"  {cm_csv}")
        print(f"  {cm_png}")

        results["test"] = {k: round(v, 6) for k, v in test_m.items()}
        results["confusion_matrix_csv"] = str(cm_csv)
        results["confusion_matrix_png"] = str(cm_png)

    with open(args.artifacts_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nArtefactos guardados en: {args.artifacts_dir}")
    print("Archivos principales:")
    print(f"  - {args.artifacts_dir / 'best_model.pth'}")
    print(f"  - {args.artifacts_dir / 'last_model.pth'}")
    print(f"  - {args.artifacts_dir / 'metrics.csv'}")
    print(f"  - {args.artifacts_dir / 'results.json'}")
    print(f"  - {args.artifacts_dir / 'class_mapping.json'}")


if __name__ == "__main__":
    main()
