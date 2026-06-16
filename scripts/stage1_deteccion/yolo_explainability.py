"""Interpretabilidad del detector YOLOv8: en que se fija para decidir drone vs interference.

Genera dos tipos de evidencia sobre el modelo entrenado (run02_60ep/best.pt):

1. Ejemplos con cajas predichas: superpone las detecciones del modelo (coloreadas
   por clase y con su confianza) sobre imagenes representativas del conjunto de test.
   Permite ver que regiones del espectrograma etiqueta como dron y cuales como
   interferencia.

2. Mapas de saliencia EigenCAM: mapa de calor que resalta las zonas del espectrograma
   que mas activan el detector. Responde de forma visual a "donde mira" la red. Se
   construye proyectando las activaciones de la capa de mayor resolucion del cuello
   (la que alimenta la cabeza de deteccion de objetos pequenos) sobre su primera
   componente principal. No requiere gradientes ni clase objetivo.

Uso tipico:
    uv run python scripts/yolo_explainability.py
    uv run python scripts/yolo_explainability.py --n-per-class 6 --conf 0.10

Salidas en data/mini3_detector_python_v3/artifacts/yolo/explainability/:
    pred_drone_*.png, pred_interference_*.png   (cajas predichas)
    cam_drone_*.png, cam_interference_*.png       (EigenCAM individual)
    grid_pred.png, grid_cam.png                   (montajes para la memoria)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Indice de clase -> nombre (segun data.yaml del dataset v3).
CLASS_NAMES = {0: "interference", 1: "drone"}
# Colores BGR por clase para las cajas.
CLASS_COLORS = {0: (60, 180, 75), 1: (40, 40, 230)}  # interference=verde, drone=rojo


def repo_root() -> Path:
    """Raiz del repositorio (carpeta padre de scripts/)."""
    return Path(__file__).resolve().parent.parent


def resolve_test_images(test_list: Path, dataset_root: Path) -> list[Path]:
    """Lee el .txt de test y reconstruye rutas validas en el sistema actual.

    Las listas se generaron en Windows (rutas con backslash y unidad C:). Se
    reconstruye cada ruta como dataset_root/all/<clase>/<fichero> para que
    funcione tanto en Windows como en Linux.
    """
    imgs: list[Path] = []
    for raw in test_list.read_text(encoding="utf-8").splitlines():
        raw = raw.strip().replace("\\", "/")
        if not raw:
            continue
        name = Path(raw).name
        # La carpeta de clase es el directorio inmediatamente anterior al fichero.
        parts = raw.split("/all/")
        class_folder = parts[1].split("/")[0] if len(parts) == 2 else Path(raw).parent.name
        candidate = dataset_root / "all" / class_folder / name
        if candidate.exists():
            imgs.append(candidate)
    return imgs


def origin_class(img_path: Path) -> str:
    """Clase de origen segun la carpeta (drone/interference/...)."""
    return img_path.parent.name


def letterbox_params(orig_h: int, orig_w: int, imgsz: int = 640):
    """Parametros del letterbox cuadrado de ultralytics (igual que en inferencia)."""
    r = min(imgsz / orig_h, imgsz / orig_w)
    new_w, new_h = round(orig_w * r), round(orig_h * r)
    dw, dh = (imgsz - new_w) / 2, (imgsz - new_h) / 2
    return r, new_w, new_h, dw, dh


def eigencam_from_activation(act: np.ndarray, percentile: float = 99.0,
                             blur_sigma: float = 0.8) -> np.ndarray:
    """Calcula el mapa EigenCAM (HxW, normalizado 0-1) de una activacion [C,H,W].

    Mejoras de contraste para que el mapa sea legible como figura:
    - suavizado gaussiano del mapa de baja resolucion (elimina el moteado);
    - normalizacion por percentil (recorta la cola superior, p.ej. p99), de modo
      que los focos de activacion ocupen todo el rango de color en lugar de quedar
      aplastados por unos pocos pixeles extremos.
    """
    c, h, w = act.shape
    reshaped = act.reshape(c, h * w).T  # [HW, C]
    reshaped = reshaped - reshaped.mean(axis=0, keepdims=True)
    # Primera componente principal de las activaciones.
    _, _, vt = np.linalg.svd(reshaped, full_matrices=False)
    proj = reshaped @ vt[0]  # [HW]
    cam = proj.reshape(h, w)
    # El signo de la componente principal es arbitrario: se orienta para que las
    # zonas de mayor energia de activacion queden con valor alto.
    mean_act = act.mean(axis=0)
    if np.corrcoef(cam.ravel(), mean_act.ravel())[0, 1] < 0:
        cam = -cam
    cam = np.maximum(cam, 0)
    # Suavizado para quitar el moteado pixel a pixel del mapa 80x80.
    if blur_sigma > 0:
        cam = cv2.GaussianBlur(cam, (0, 0), blur_sigma)
    # Normalizacion robusta por percentil: el techo es el percentil indicado,
    # no el maximo absoluto, lo que aumenta el rango dinamico de los focos.
    hi = np.percentile(cam, percentile)
    cam = np.clip(cam / (hi + 1e-8), 0, 1)
    return cam


def overlay_cam(orig_bgr: np.ndarray, cam: np.ndarray, imgsz: int = 640,
                gamma: float = 0.6) -> np.ndarray:
    """Superpone el CAM sobre el espectrograma en gris con transparencia por pixel.

    El fondo se muestra en escala de grises (las rafagas siguen siendo visibles) y
    el color solo aparece donde hay activacion: la transparencia de cada pixel es
    proporcional al valor del CAM (elevado a gamma para atenuar lo poco activado).
    Asi se ve con claridad si los focos coinciden con las rafagas.
    """
    oh, ow = orig_bgr.shape[:2]
    _, new_w, new_h, dw, dh = letterbox_params(oh, ow, imgsz)
    cam_640 = cv2.resize(cam, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    top, left = int(round(dh)), int(round(dw))
    cam_crop = cam_640[top:top + new_h, left:left + new_w]
    cam_orig = cv2.resize(cam_crop, (ow, oh), interpolation=cv2.INTER_LINEAR)

    gray = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    heat = cv2.applyColorMap(np.uint8(255 * cam_orig), cv2.COLORMAP_JET)
    alpha = np.power(cam_orig, gamma)[..., None]  # peso por pixel [H,W,1]
    out = base.astype(np.float32) * (1 - alpha) + heat.astype(np.float32) * alpha
    return out.astype(np.uint8)


def draw_predictions(result, orig_bgr: np.ndarray) -> np.ndarray:
    """Dibuja las cajas predichas coloreadas por clase con su confianza."""
    img = orig_bgr.copy()
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        cv2.putText(img, "sin detecciones", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)
        return img
    for xyxy, cls, conf in zip(boxes.xyxy.cpu().numpy(),
                               boxes.cls.cpu().numpy().astype(int),
                               boxes.conf.cpu().numpy()):
        x1, y1, x2, y2 = xyxy.astype(int)
        color = CLASS_COLORS.get(int(cls), (200, 200, 200))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    # Resumen de conteo por clase en la esquina.
    cls_arr = boxes.cls.cpu().numpy().astype(int)
    n_drone = int((cls_arr == 1).sum())
    n_intf = int((cls_arr == 0).sum())
    txt = f"drone={n_drone}  interf={n_intf}"
    cv2.putText(img, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


def make_grid(images: list[np.ndarray], titles: list[str], cols: int) -> np.ndarray:
    """Montaje sencillo en rejilla con titulos."""
    if not images:
        return np.zeros((100, 100, 3), np.uint8)
    h = max(im.shape[0] for im in images)
    w = max(im.shape[1] for im in images)
    pad_top = 28
    cells = []
    for im, title in zip(images, titles):
        canvas = np.full((h + pad_top, w, 3), 30, np.uint8)
        canvas[pad_top:pad_top + im.shape[0], :im.shape[1]] = im
        cv2.putText(canvas, title, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1)
        cells.append(canvas)
    rows = []
    for i in range(0, len(cells), cols):
        row = cells[i:i + cols]
        while len(row) < cols:
            row.append(np.full_like(cells[0], 30))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def main() -> None:
    root = repo_root()
    ds_default = root / "data" / "mini3_detector_python_v3"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path,
                        default=ds_default / "artifacts" / "yolo" / "run02_60ep" / "weights" / "best.pt")
    parser.add_argument("--dataset-root", type=Path, default=ds_default)
    parser.add_argument("--test-list", type=Path, default=ds_default / "yolo" / "test.txt")
    parser.add_argument("--out", type=Path,
                        default=ds_default / "artifacts" / "yolo" / "explainability")
    parser.add_argument("--n-per-class", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--cam-layer", type=int, default=-1,
                        help="Indice de capa para EigenCAM. -1 = auto (P3 que alimenta la cabeza).")
    parser.add_argument("--cam-percentile", type=float, default=99.0,
                        help="Percentil usado como techo en la normalizacion del CAM.")
    parser.add_argument("--cam-blur", type=float, default=0.8,
                        help="Sigma del suavizado gaussiano del mapa CAM (0 = sin suavizar).")
    parser.add_argument("--cam-gamma", type=float, default=0.6,
                        help="Exponente de la transparencia del overlay (menor = focos mas marcados).")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)

    print(f"Cargando modelo: {args.weights}")
    model = YOLO(str(args.weights))
    seq = model.model.model  # nn.Sequential de capas

    # Capa objetivo para EigenCAM: por defecto, la fuente de mayor resolucion de la
    # cabeza Detect (deteccion de objetos pequenos = nuestras rafagas FHSS).
    if args.cam_layer >= 0:
        target_idx = args.cam_layer
    else:
        detect = seq[-1]
        srcs = detect.f if isinstance(detect.f, (list, tuple)) else [detect.f]
        target_idx = int(srcs[0])
    target_layer = seq[target_idx]
    print(f"Capa EigenCAM: indice {target_idx} ({type(target_layer).__name__})")

    activation: dict[str, np.ndarray] = {}

    def hook(_module, _inp, out):
        t = out[0] if isinstance(out, (list, tuple)) else out
        activation["a"] = t.detach().cpu().float().numpy()[0]

    target_layer.register_forward_hook(hook)

    images = resolve_test_images(args.test_list, args.dataset_root)
    by_class = {"drone": [], "interference": []}
    for p in images:
        oc = origin_class(p)
        if oc in by_class:
            by_class[oc].append(p)
    print(f"Imagenes test: drone={len(by_class['drone'])} interference={len(by_class['interference'])}")

    def class_index(folder: str) -> int:
        return 1 if folder == "drone" else 0

    # Para cada clase, se ejecuta la inferencia y se eligen las imagenes con
    # detecciones mas confiadas de la clase correcta (casos claros y representativos).
    selected: dict[str, list[Path]] = {}
    for folder in ("drone", "interference"):
        target_cls = class_index(folder)
        scored = []
        for p in by_class[folder]:
            res = model.predict(str(p), imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            if res.boxes is None or len(res.boxes) == 0:
                continue
            cls_arr = res.boxes.cls.cpu().numpy().astype(int)
            conf_arr = res.boxes.conf.cpu().numpy()
            mask = cls_arr == target_cls
            if mask.sum() == 0:
                continue
            score = float(conf_arr[mask].mean()) * float(mask.sum())
            scored.append((score, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        selected[folder] = [p for _, p in scored[:args.n_per_class]]
        print(f"  {folder}: seleccionadas {len(selected[folder])} imagenes")

    pred_imgs, pred_titles = [], []
    cam_imgs, cam_titles = [], []
    for folder in ("drone", "interference"):
        for i, p in enumerate(selected[folder]):
            res = model.predict(str(p), imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            orig = res.orig_img.copy()

            pred = draw_predictions(res, orig)
            pred_path = args.out / f"pred_{folder}_{i:02d}.png"
            cv2.imwrite(str(pred_path), pred)
            pred_imgs.append(pred)
            pred_titles.append(f"{folder} #{i}")

            cam = eigencam_from_activation(activation["a"],
                                           percentile=args.cam_percentile,
                                           blur_sigma=args.cam_blur)
            cam_over = overlay_cam(orig, cam, args.imgsz, gamma=args.cam_gamma)
            cam_path = args.out / f"cam_{folder}_{i:02d}.png"
            cv2.imwrite(str(cam_path), cam_over)
            cam_imgs.append(cam_over)
            cam_titles.append(f"{folder} #{i}")

    cv2.imwrite(str(args.out / "grid_pred.png"),
                make_grid(pred_imgs, pred_titles, cols=args.n_per_class))
    cv2.imwrite(str(args.out / "grid_cam.png"),
                make_grid(cam_imgs, cam_titles, cols=args.n_per_class))
    print(f"Listo. Figuras en: {args.out}")


if __name__ == "__main__":
    main()
