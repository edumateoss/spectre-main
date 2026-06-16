"""
Interpretabilidad del clasificador ResNet18 de 6 clases (segunda etapa).

Responde a la pregunta "que esta aprendiendo la red" desde dentro del modelo,
de forma complementaria al barrido de SNR (energia vs morfologia) y al EigenCAM
del detector YOLO (ciego a la clase):

  1. Embeddings de la penultima capa (512-dim) de las 941 muestras de test,
     proyectados a 2D con t-SNE y PCA y coloreados por clase real. Muestra la
     GEOMETRIA del espacio de representacion aprendido: que clases forman
     clusteres compactos y separados y cuales se solapan.
  2. GradCAM por clase sobre la ultima capa convolucional (layer4). Al ser la
     capa inmediatamente anterior al global average pooling, GradCAM se reduce
     de forma analitica a CAM: mapa_c = ReLU(sum_k W_fc[c,k] * A_k), donde A es
     la salida de layer4 y W_fc la matriz de la capa lineal. No requiere
     backward. Es discriminativo por clase, a diferencia de EigenCAM.

Reproduce ademas la accuracy y la matriz de confusion del test como control.

Uso:
    uv run python scripts/interpret_stage2.py
    uv run python scripts/interpret_stage2.py --n-cam 8

Salidas (--output-dir, por defecto la carpeta de artefactos del modelo):
    interpret/embeddings.npz                features 512-dim, etiquetas, predicciones
    interpret/validation.json               accuracy y matriz de confusion
    interpret/resnet18_tsne_embeddings.png  t-SNE coloreado por clase
    interpret/resnet18_embeddings_tsne_pca.png  t-SNE + PCA lado a lado
    interpret/resnet18_gradcam_por_clase.png    GradCAM/CAM por clase
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import FrequencyNormalize  # misma normalizacion en frecuencia que en entrenamiento

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_DEFAULT_ART = Path("data/stage2_classifier_v2_mavic_split/artifacts/resnet18_stage2_mavic_split")

_COLORS = {
    "f450": "#1f77b4",
    "hunter": "#ff7f0e",
    "mavic_video": "#2ca02c",
    "mavic_novideo": "#d62728",
    "mini3": "#9467bd",
    "interference": "#7f7f7f",
}


def build_transform() -> transforms.Compose:
    """Transformacion identica a la de evaluacion: FrequencyNormalize -> 224 -> ImageNet."""
    return transforms.Compose([
        FrequencyNormalize(window_frac=0.5),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


def load_model(ckpt: Path, num_classes: int) -> torch.nn.Module:
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts-dir", type=Path, default=_DEFAULT_ART)
    ap.add_argument("--data-root", type=Path, default=Path("."))
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-cam", type=int, default=8, help="numero de muestras en la figura GradCAM")
    ap.add_argument("--perplexity", type=float, default=30.0)
    args = ap.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    art = args.artifacts_dir
    out = args.output_dir or (art / "interpret")
    out.mkdir(parents=True, exist_ok=True)

    class_map = json.load(open(art / "class_mapping.json"))
    idx2cls = {int(k): v for k, v in class_map.items()}
    cls2idx = {v: k for k, v in idx2cls.items()}
    n_cls = len(idx2cls)

    tf = build_transform()
    model = load_model(art / "best_model.pth", n_cls)
    fc_w = model.fc.weight.detach().numpy()  # (n_cls, 512)

    acts: dict[str, torch.Tensor] = {}
    model.layer4.register_forward_hook(lambda m, i, o: acts.__setitem__("v", o.detach()))

    rows = list(csv.DictReader(open(art / "splits" / "test.csv")))

    # --- Forward de todo el test: embeddings + predicciones ---
    embs, y_true, y_pred, paths = [], [], [], []
    feats: list[np.ndarray] = []
    h = model.avgpool.register_forward_hook(
        lambda m, i, o: feats.append(o.detach().flatten(1).numpy())
    )
    with torch.no_grad():
        for i in range(0, len(rows), args.batch_size):
            chunk = rows[i:i + args.batch_size]
            xs = torch.stack([
                tf(Image.open(args.data_root / r["filename"]).convert("RGB")) for r in chunk
            ])
            feats.clear()
            logits = model(xs)
            embs.extend(list(feats[0]))
            y_pred.extend(logits.argmax(1).tolist())
            y_true.extend(cls2idx[r["label"]] for r in chunk)
            paths.extend(r["filename"] for r in chunk)
            print(f"  {min(i + args.batch_size, len(rows))}/{len(rows)}", flush=True)
    h.remove()

    embs = np.asarray(embs)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    np.savez(out / "embeddings.npz", embs=embs, y_true=y_true, y_pred=y_pred,
             paths=np.asarray(paths, dtype=object))

    # --- Control: accuracy y matriz de confusion ---
    acc = float((y_true == y_pred).mean())
    cm = np.zeros((n_cls, n_cls), int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    print(f"\nAccuracy test = {acc:.3f}")
    json.dump({"accuracy": acc, "confusion_matrix": cm.tolist(),
               "classes": [idx2cls[i] for i in range(n_cls)]},
              open(out / "validation.json", "w"), indent=2)

    # --- Proyecciones t-SNE y PCA ---
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    print("Proyectando embeddings (t-SNE / PCA)...", flush=True)
    z_tsne = TSNE(n_components=2, perplexity=args.perplexity, init="pca",
                  learning_rate="auto", random_state=42).fit_transform(embs)
    z_pca = PCA(n_components=2, random_state=42).fit_transform(embs)
    np.savez(out / "projections.npz", tsne=z_tsne, pca=z_pca)

    def scatter(ax, z, title):
        for ci in range(n_cls):
            mask = y_true == ci
            ax.scatter(z[mask, 0], z[mask, 1], s=16, alpha=0.72,
                       c=_COLORS.get(idx2cls[ci], None), label=idx2cls[ci], edgecolors="none")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    scatter(axes[0], z_tsne, "Representacion ResNet18 (t-SNE)")
    scatter(axes[1], z_pca, "Representacion ResNet18 (PCA)")
    axes[0].legend(markerscale=2, fontsize=10, framealpha=0.9)
    fig.suptitle("Embeddings de la penultima capa (512-dim) sobre el test")
    fig.tight_layout()
    fig.savefig(out / "resnet18_embeddings_tsne_pca.png", dpi=140, bbox_inches="tight")

    fig2, ax = plt.subplots(figsize=(9, 7.5))
    scatter(ax, z_tsne, "Espacio de representacion aprendido (t-SNE de embeddings ResNet18)")
    ax.legend(markerscale=2, fontsize=11, framealpha=0.9)
    fig2.tight_layout()
    fig2.savefig(out / "resnet18_tsne_embeddings.png", dpi=150, bbox_inches="tight")

    # --- GradCAM/CAM por clase ---
    def cam_for(path, cls):
        img = transforms.Resize((224, 224))(FrequencyNormalize(0.5)(Image.open(args.data_root / path).convert("RGB")))
        base = np.asarray(img, dtype=np.float32) / 255.0
        x = transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)(transforms.ToTensor()(img)).unsqueeze(0)
        with torch.no_grad():
            out_logits = model(x)
        probs = out_logits.softmax(1)[0].numpy()
        pred = int(out_logits.argmax(1))
        c = pred if cls is None else cls
        A = acts["v"][0].numpy()  # (512, 7, 7)
        cam = np.maximum((fc_w[c][:, None, None] * A).sum(0), 0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)) / 255.0
        return base, cam, pred, probs

    # Plan: ejemplos bien clasificados + un mavic_novideo mal clasificado (->hunter)
    plan = [("mini3", True), ("mini3", True), ("interference", True), ("interference", True),
            ("hunter", True), ("mavic_video", True), ("f450", True), ("mavic_novideo", False)]
    plan = plan[:args.n_cam]
    picked, used = [], set()
    for lab, ok in plan:
        for r in rows:
            if r["label"] != lab or r["filename"] in used:
                continue
            _, _, pred, _ = cam_for(r["filename"], None)
            if (pred == cls2idx[lab]) == ok:
                picked.append((r, pred))
                used.add(r["filename"])
                break

    n = len(picked)
    fig3, axes3 = plt.subplots(2, n, figsize=(2.9 * n, 6.0))
    for j, (r, pred) in enumerate(picked):
        base, cam, _, probs = cam_for(r["filename"], pred)
        gray = base.mean(2)
        axes3[0, j].imshow(gray, cmap="gray"); axes3[0, j].axis("off")
        axes3[0, j].set_title(f"real: {r['label']}", fontsize=9)
        axes3[1, j].imshow(gray, cmap="gray")
        axes3[1, j].imshow(cam, cmap="jet", alpha=0.5); axes3[1, j].axis("off")
        axes3[1, j].set_title(f"pred: {idx2cls[pred]} ({probs[pred]:.2f})", fontsize=9)
    fig3.suptitle("GradCAM/CAM por clase (layer4 del ResNet18): region que sostiene la decision")
    fig3.tight_layout()
    fig3.savefig(out / "resnet18_gradcam_por_clase.png", dpi=140, bbox_inches="tight")

    print(f"\nSalidas en {out}")


if __name__ == "__main__":
    main()
