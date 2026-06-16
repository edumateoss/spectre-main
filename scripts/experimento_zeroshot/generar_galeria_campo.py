"""
generar_galeria_campo.py
========================

Genera una galeria visual de predicciones zero-shot del detector YOLOv8
(entrenado en Mini3) sobre los espectrogramas de campo (F450, Mavic, Hunter).

Para cada tipo de dron selecciona imagenes representativas:
  - Alta deteccion: imagenes donde el modelo encontro muchas rafagas
  - Deteccion media: imagenes con pocas rafagas detectadas
  - Sin deteccion: imagenes donde el modelo no encontro nada

Salida:
  data/stage2_classifier_v1/artifacts/zeroshot/galeria/
    {drone_type}_alta_{n}.png     alta deteccion
    {drone_type}_media_{n}.png    deteccion media
    {drone_type}_nula_{n}.png     sin deteccion

Uso:
  uv run python scripts/generar_galeria_campo.py
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

DEFAULT_WEIGHTS = Path(
    "data/mini3_detector_python_v3/artifacts/yolo/run02_60ep/weights/best.pt"
)
DEFAULT_DATASET  = Path("data/stage2_classifier_v1")
DEFAULT_REPORT   = DEFAULT_DATASET / "artifacts" / "zeroshot" / "zeroshot_report.csv"
DEFAULT_GALERIA  = DEFAULT_DATASET / "artifacts" / "zeroshot" / "galeria"

CLASS_NAMES   = {0: "interference", 1: "drone"}
COLORES       = {0: (0, 230, 0), 1: (255, 50, 50)}   # verde / rojo
SEED = 42


def dibujar(
    img_path: Path,
    xyxyn: np.ndarray,
    clases: list[int],
    confs: list[float],
) -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    for i, (cls, conf) in enumerate(zip(clases, confs)):
        x1, y1, x2, y2 = xyxyn[i]
        color = COLORES.get(cls, (255, 255, 255))
        draw.rectangle(
            [x1 * w, y1 * h, x2 * w, y2 * h],
            outline=color,
            width=3,
        )
        etiqueta = f"{CLASS_NAMES.get(cls, str(cls))} {conf:.2f}"
        draw.text((x1 * w + 3, y1 * h + 3), etiqueta, fill=color)

    return img


def seleccionar_muestras(
    df: pd.DataFrame,
    n_alta: int,
    n_media: int,
    n_nula: int,
) -> dict[str, pd.DataFrame]:
    """
    Divide el dataframe de un tipo de dron en tres grupos por numero de
    detecciones y devuelve una muestra aleatoria de cada grupo.
    """
    rng = random.Random(SEED)

    max_boxes = df["n_boxes_drone"].max() if len(df) > 0 else 0

    if max_boxes > 0:
        umbral_alta  = max(1, max_boxes * 0.5)
        umbral_media = 1
    else:
        umbral_alta  = 1
        umbral_media = 1

    alta  = df[df["n_boxes_drone"] >= umbral_alta]
    media = df[(df["n_boxes_drone"] > 0) & (df["n_boxes_drone"] < umbral_alta)]
    nula  = df[df["n_boxes_drone"] == 0]

    def muestra(subdf: pd.DataFrame, n: int) -> pd.DataFrame:
        if len(subdf) == 0:
            return subdf
        idx = rng.sample(range(len(subdf)), min(n, len(subdf)))
        return subdf.iloc[idx].reset_index(drop=True)

    return {
        "alta":  muestra(alta,  n_alta),
        "media": muestra(media, n_media),
        "nula":  muestra(nula,  n_nula),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights",    type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--dataset-dir",type=Path, default=DEFAULT_DATASET)
    p.add_argument("--report-csv", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--galeria-dir",type=Path, default=DEFAULT_GALERIA)
    p.add_argument("--conf",       type=float, default=0.10)
    p.add_argument("--n-alta",     type=int,   default=4)
    p.add_argument("--n-media",    type=int,   default=3)
    p.add_argument("--n-nula",     type=int,   default=3)
    p.add_argument("--device",     default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit("Falta 'ultralytics'. Ejecuta 'uv sync'.") from e

    report_csv  = args.report_csv.resolve()
    dataset_dir = args.dataset_dir.resolve()
    galeria_dir = args.galeria_dir.resolve()
    galeria_dir.mkdir(parents=True, exist_ok=True)

    if not report_csv.exists():
        raise FileNotFoundError(
            f"No existe {report_csv}. Ejecuta primero infer_zeroshot_campo.py."
        )

    model = YOLO(str(args.weights.resolve()))
    df    = pd.read_csv(report_csv)

    print(f"Galeria destino: {galeria_dir}")
    print()

    for drone_type, grupo in df.groupby("drone_type"):
        print(f"--- {drone_type.upper()} ({len(grupo)} imagenes) ---")

        muestras = seleccionar_muestras(
            grupo, args.n_alta, args.n_media, args.n_nula
        )

        for nivel, subdf in muestras.items():
            if subdf.empty:
                print(f"  [{nivel}] sin imagenes en este rango")
                continue

            for i, row in subdf.iterrows():
                img_path = dataset_dir / row["filename"]
                if not img_path.exists():
                    continue

                resultados = model.predict(
                    str(img_path),
                    conf=args.conf,
                    device=args.device,
                    verbose=False,
                    imgsz=640,
                )
                r = resultados[0]

                if r.boxes is not None and len(r.boxes) > 0:
                    clases  = r.boxes.cls.cpu().numpy().astype(int).tolist()
                    confs   = r.boxes.conf.cpu().numpy().tolist()
                    xyxyn   = r.boxes.xyxyn.cpu().numpy()
                else:
                    clases, confs = [], []
                    xyxyn = np.empty((0, 4))

                n_drone = clases.count(1)
                img_out = dibujar(img_path, xyxyn, clases, confs)

                nombre = f"{drone_type}_{nivel}_{i:02d}_n{n_drone}.png"
                img_out.save(galeria_dir / nombre)
                print(f"  [{nivel}] {nombre}  (drone={n_drone}, total={len(clases)})")

    print()
    print(f"Galeria guardada en: {galeria_dir}")
    print(f"Total imagenes:      {len(list(galeria_dir.glob('*.png')))}")


if __name__ == "__main__":
    main()
