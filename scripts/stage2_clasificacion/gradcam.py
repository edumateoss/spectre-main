"""
Visualizacion GradCAM para el detector binario drone/interference.

Muestra que zona del espectrograma activa la red para cada clase,
permitiendo verificar si aprende morfologia de señal o artefactos.

Uso:
    uv run python scripts/gradcam.py
    uv run python scripts/gradcam.py --n-samples 6 --split val

Salidas (--output-dir):
    gradcam_grid.png          cuadricula comparativa drone vs interference
    gradcam_drone.png         muestras de dron con heatmap
    gradcam_interference.png  muestras de interferencia con heatmap
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import CLASS_TO_IDX, IDX_TO_CLASS, FrequencyNormalize

_DATA_DIR  = Path("data/mini3_detector_python_v1")
_ARTIFACTS = _DATA_DIR / "artifacts"

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GradCAM sobre el detector de drones")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--n-samples", type=int, default=4,
                   help="Muestras por clase a visualizar")
    p.add_argument("--checkpoint", type=Path, default=_ARTIFACTS / "best_model.pth")
    p.add_argument("--metadata-dir", type=Path,
                   default=_DATA_DIR / "metadata")
    p.add_argument("--data-root", type=Path, default=_DATA_DIR)
    p.add_argument("--output-dir", type=Path, default=_ARTIFACTS / "gradcam")
    p.add_argument("--no-freq-normalize", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# GradCAM
# ---------------------------------------------------------------------------

class GradCAM:
    """
    GradCAM sobre la ultima capa convolucional (layer4) de ResNet18.

    Para cada imagen calcula un mapa de activacion que indica que zonas
    del espectrograma influyen mas en la decision de la red.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        # Hook en la ultima capa residual (salida 7x7 para entrada 224x224)
        target = model.layer4[-1]
        target.register_forward_hook(self._hook_activation)
        target.register_full_backward_hook(self._hook_gradient)

    def _hook_activation(self, module, inp, out) -> None:
        self._activations = out.detach()

    def _hook_gradient(self, module, grad_in, grad_out) -> None:
        self._gradients = grad_out[0].detach()

    def compute(self, tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Devuelve el mapa GradCAM normalizado [0,1] para la clase class_idx.
        tensor: (1, 3, 224, 224) ya en el dispositivo del modelo.
        """
        self.model.eval()
        tensor = tensor.requires_grad_(True)

        logits = self.model(tensor)
        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Pesos = media global de los gradientes por canal
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1).squeeze()   # (H, W)
        cam = torch.relu(cam).cpu().numpy()

        # Normalizar a [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam


# ---------------------------------------------------------------------------
# Utilidades de imagen
# ---------------------------------------------------------------------------

def load_display_image(img_path: Path, freq_normalize: bool) -> Image.Image:
    """
    Carga la imagen tal y como la ve la red antes de ToTensor/Normalize.
    Util para superponer el heatmap sobre algo interpretable.
    """
    img = Image.open(img_path).convert("RGB")
    if freq_normalize:
        img = FrequencyNormalize(window_frac=0.5)(img)
    return img.resize((224, 224), Image.BILINEAR)


def image_to_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    """Convierte la imagen display a tensor normalizado para el modelo."""
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])
    return t(img).unsqueeze(0).to(device)


def overlay_heatmap(
    img: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """Superpone el heatmap GradCAM (jet) sobre la imagen original."""
    cmap = plt.get_cmap("jet")
    cam_resized = np.array(
        Image.fromarray(np.uint8(cam * 255)).resize(img.size, Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    heatmap_rgb = np.uint8(cmap(cam_resized)[:, :, :3] * 255)
    img_arr = np.array(img, dtype=np.float32)
    blended = (1 - alpha) * img_arr + alpha * heatmap_rgb
    return Image.fromarray(np.uint8(blended))


# ---------------------------------------------------------------------------
# Construccion de figura
# ---------------------------------------------------------------------------

def _clean_axis(ax) -> None:
    """Quita ticks y marco pero conserva la posibilidad de poner ylabel."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def make_figure(
    rows: list[dict],
    title: str,
    output_path: Path,
) -> None:
    """
    Genera una figura de dos columnas: [Entrada (tras FreqNorm) | GradCAM de la clase verdadera].

    Se superpone unicamente el mapa de la clase real de cada fila, de modo que
    el contraste relevante se ve entre filas (dron -> calor sobre las rafagas,
    interferencia -> activacion difusa) y no entre columnas redundantes sobre
    la misma imagen.
    """
    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(7, 3.4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]  # asegurar 2D

    # El titulo de la columna del heatmap depende de las clases presentes:
    # si todas las filas son de la misma clase lo indicamos explicitamente.
    labels_present = {row["true_label"] for row in rows}
    if len(labels_present) == 1:
        cam_col_title = f"GradCAM clase {next(iter(labels_present))}"
    else:
        cam_col_title = "GradCAM (clase verdadera)"
    axes[0, 0].set_title("Entrada (tras FreqNorm)", fontsize=9, pad=4)
    axes[0, 1].set_title(cam_col_title, fontsize=9, pad=4)

    for row_idx, row in enumerate(rows):
        display_img = row["display_img"]
        true_label  = row["true_label"]
        pred_label  = row["pred_label"]
        confidence  = row["confidence"]

        # Mapa de la clase verdadera de esta fila (no se mezclan clases).
        cam_true = row["cam_drone"] if true_label == "drone" else row["cam_interference"]
        overlay  = overlay_heatmap(display_img, cam_true)

        axes[row_idx, 0].imshow(display_img)
        _clean_axis(axes[row_idx, 0])
        axes[row_idx, 1].imshow(overlay)
        axes[row_idx, 1].axis("off")

        # Etiqueta lateral con resultado (verde si acierta, rojo si falla)
        color = "green" if pred_label == true_label else "red"
        axes[row_idx, 0].set_ylabel(
            f"true: {true_label}\npred: {pred_label} ({confidence:.2f})",
            fontsize=8, color=color, rotation=0, labelpad=55, va="center",
        )

    fig.suptitle(title, fontsize=11, y=1.005)
    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"No existe checkpoint: {args.checkpoint}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    freq_normalize = not args.no_freq_normalize
    print(f"FrequencyNormalize: {'activado' if freq_normalize else 'desactivado'}")

    # Cargar modelo
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"Checkpoint: epoch={ckpt.get('epoch','?')}, val_f1={ckpt.get('val_f1','?')}")

    grad_cam = GradCAM(model)

    # Cargar CSV y seleccionar muestras por clase
    csv_path = args.metadata_dir / f"{args.split}.csv"
    df = pd.read_csv(csv_path)

    samples_per_class: dict[str, list[dict]] = {}
    for label in ["drone", "interference"]:
        subset = df[df["label"] == label].sample(
            n=min(args.n_samples, len(df[df["label"] == label])),
            random_state=args.seed,
        )
        rows = []
        for _, row in subset.iterrows():
            img_path = args.data_root / row["filename"]
            display_img = load_display_image(img_path, freq_normalize)
            tensor = image_to_tensor(display_img, device)

            cam_drone  = grad_cam.compute(tensor.clone(), CLASS_TO_IDX["drone"])
            cam_interf = grad_cam.compute(tensor.clone(), CLASS_TO_IDX["interference"])

            with torch.no_grad():
                logits = model(tensor)
                probs  = torch.softmax(logits, dim=1).squeeze()
                pred_idx = logits.argmax(dim=1).item()

            rows.append({
                "display_img":     display_img,
                "cam_drone":       cam_drone,
                "cam_interference": cam_interf,
                "true_label":      label,
                "pred_label":      IDX_TO_CLASS[pred_idx],
                "confidence":      float(probs[pred_idx]),
            })

        samples_per_class[label] = rows
        print(f"  {label}: {len(rows)} muestras procesadas")

    # Figura por clase
    for label, rows in samples_per_class.items():
        make_figure(
            rows,
            title=f"GradCAM — clase '{label}' ({args.split})",
            output_path=args.output_dir / f"gradcam_{label}.png",
        )

    # Figura combinada: primero drone, luego interference
    all_rows = samples_per_class["drone"] + samples_per_class["interference"]
    make_figure(
        all_rows,
        title=f"GradCAM — comparativa drone vs interference ({args.split})",
        output_path=args.output_dir / "gradcam_grid.png",
    )

    print(f"\nImagenes GradCAM guardadas en: {args.output_dir}")


if __name__ == "__main__":
    main()
