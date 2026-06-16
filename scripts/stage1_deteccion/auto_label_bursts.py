"""
auto_label_bursts.py
====================

Generador automatico de anotaciones YOLO sobre los espectrogramas del
dataset v3 para entrenar el detector de la primera etapa del pipeline
two-stage (Stage 1: localizacion de bursts FHSS / WiFi).

Estrategia de etiquetado:
  - Cada captura del dataset v3 contiene un unico tipo de senal:
    o bien dron solo (capturas mini3_drone_*) o bien interferencia
    sola (capturas mini3_nodrone_*). Por tanto, todo blob de alta
    energia detectado en un PNG hereda la clase de su captura.
  - Es etiquetado debil pero consistente con la procedencia, y evita
    tener que anotar manualmente miles de bursts.

Pipeline para cada PNG:
  1. Lectura como imagen en escala de grises (proxy de energia en dB,
     monotono respecto al colormap parula).
  2. Suavizado gaussiano ligero para reducir ruido pixel a pixel.
  3. Umbralizacion por percentil global de la imagen (adaptativa al
     contenido: en espectrogramas ruidosos el umbral sube, en limpios
     baja).
  4. Operaciones morfologicas (closing + opening) para consolidar
     regiones contiguas y eliminar speckle.
  5. Componentes conectados (scipy.ndimage.label).
  6. Filtrado de blobs por area minima/maxima y aspect ratio.
  7. Conversion de bounding box a formato YOLO normalizado (cx, cy, w, h).
  8. Escritura del .txt junto al PNG.

Salida:
  - {root}/all/{clase}/{stem}.txt          (un .txt por PNG)
  - {root}/yolo/data.yaml                  (manifest YOLO)
  - {root}/yolo/train.txt, val.txt, test.txt
  - {root}/yolo/preview/{stem}.png         (si --preview-n > 0)
  - {root}/yolo/labeling_report.csv        (n_bursts por PNG, util para QA)

Uso tipico:
  uv run python scripts/auto_label_bursts.py \\
      --metadata-csv data/mini3_detector_python_v3/metadata/spectrograms.csv \\
      --root-dir     data/mini3_detector_python_v3 \\
      --threshold-percentile 92 \\
      --min-area 30 \\
      --max-area-frac 0.40 \\
      --preview-n 12
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage

# ---------------------------------------------------------------------------
# Mapeo de clases YOLO. Mantenemos el mismo orden que CLASS_TO_IDX en
# common.py (interference=0, drone=1) para que el clasificador stage 2 sea
# coherente con el detector stage 1.
# ---------------------------------------------------------------------------
CLASS_TO_YOLO_ID: dict[str, int] = {"interference": 0, "drone": 1}
YOLO_ID_TO_NAME: dict[int, str] = {v: k for k, v in CLASS_TO_YOLO_ID.items()}


@dataclass
class BboxParams:
    """
    Parametros del algoritmo de extraccion de bounding boxes.

    Defaults calibrados experimentalmente sobre v3 para detectar bursts FHSS
    pequenos (OcuSync del dron, BT) y descartar barras WiFi anchas:
      - threshold_percentile=99.8 -> solo el 0.2% mas brillante (picos de burst)
      - min_area_px=8             -> permite bursts FHSS muy pequenos
      - max_area_frac=0.005       -> descarta cajas grandes (barras WiFi)
      - morph_kernel=1            -> sin consolidacion, preserva blobs sueltos
      - smooth_sigma=0.4          -> suavizado minimo
    """
    threshold_percentile: float = 99.8
    min_area_px: int = 8
    max_area_frac: float = 0.005
    morph_kernel: int = 1
    smooth_sigma: float = 0.4


# ---------------------------------------------------------------------------
# Argumentos de linea de comandos
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--metadata-csv",
        default="data/mini3_detector_python_v3/metadata/spectrograms.csv",
        help="CSV maestro con columnas 'filename', 'label', 'capture_id'.",
    )
    p.add_argument(
        "--root-dir",
        default="data/mini3_detector_python_v3",
        help="Raiz del dataset (contiene all/, metadata/, yolo/).",
    )
    p.add_argument(
        "--splits-dir",
        default=None,
        help=(
            "Directorio con train.csv, val.csv, test.csv. Si no se indica, "
            "se asume {root-dir}/metadata."
        ),
    )
    p.add_argument(
        "--threshold-percentile",
        type=float,
        default=99.8,
        help=(
            "Percentil sobre el que se umbraliza. 99.8 solo deja pasar los "
            "picos mas brillantes (los bursts FHSS), descartando el resto."
        ),
    )
    p.add_argument(
        "--min-area",
        type=int,
        default=8,
        help=(
            "Area minima de un blob (pixeles). Bajado a 8 para detectar los "
            "bursts FHSS de OcuSync, que son muy pequenos en el espectrograma."
        ),
    )
    p.add_argument(
        "--max-area-frac",
        type=float,
        default=0.005,
        help=(
            "Area maxima de un blob como fraccion del area de la imagen. "
            "0.005 (~2900 px en 1024x576) descarta las barras WiFi anchas."
        ),
    )
    p.add_argument(
        "--morph-kernel",
        type=int,
        default=1,
        help=(
            "Tamano del kernel cuadrado para closing/opening morfologico. "
            "1 desactiva la morfologia: preserva blobs pequenos sin fundirlos."
        ),
    )
    p.add_argument(
        "--smooth-sigma",
        type=float,
        default=0.4,
        help=(
            "Sigma del filtro gaussiano de suavizado previo. 0.4 da un "
            "suavizado minimo que reduce ruido sin difuminar los picos."
        ),
    )
    p.add_argument(
        "--preview-n",
        type=int,
        default=0,
        help=(
            "Numero de previews (cajas dibujadas sobre el PNG) a generar por "
            "clase. 0 = no generar previews."
        ),
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Procesar solo los primeros N PNGs (debug rapido).",
    )

    p.add_argument(
        "--preview-random",
        action="store_true",
        help="Generar previews a partir de capturas aleatorias en vez de las primeras.",
    )

    p.add_argument(
        "--preview-seed",
        type=int,
        default=42,
        help="Semilla aleatoria para seleccionar previews reproducibles.",
    )
    p.add_argument(
        "--regenerate-splits-only",
        action="store_true",
        default=False,
        help=(
            "No re-etiquetar PNGs. Solo regenera data.yaml + train/val/test.txt "
            "con las rutas absolutas correctas. Util cuando ya etiquetaste antes "
            "y solo necesitas actualizar el manifest YOLO."
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Funciones de procesado de imagen
# ---------------------------------------------------------------------------
def png_to_energy(img_path: Path) -> tuple[np.ndarray, int, int]:
    """
    Convierte un PNG con colormap parula a un mapa de energia en escala de grises.

    Devuelve (energy, width, height). 'energy' es float32 en rango aproximado
    [0, 255], monotono respecto al nivel de senal en dB para el colormap parula.

    Justificacion: la luminancia ITU-R 601 (L = 0.299 R + 0.587 G + 0.114 B)
    da valores bajos para azul oscuro (baja energia) y altos para amarillo
    (alta energia) en parula, lo que es suficiente para umbralizar y detectar
    blobs por energia sin necesidad de invertir el colormap.
    """
    img = Image.open(img_path).convert("L")
    energy = np.asarray(img, dtype=np.float32)
    return energy, img.width, img.height


def extract_bboxes(
    energy: np.ndarray,
    params: BboxParams,
) -> list[tuple[int, int, int, int]]:
    """
    Extrae bounding boxes de blobs de alta energia.

    Devuelve lista de tuplas (x_min, y_min, x_max, y_max) en pixeles.
    """
    h, w = energy.shape

    # 1. Suavizado para reducir ruido pixel a pixel sin difuminar bursts grandes.
    if params.smooth_sigma > 0:
        smoothed = ndimage.gaussian_filter(energy, sigma=params.smooth_sigma)
    else:
        smoothed = energy

    # 2. Umbralizacion por percentil global. Adaptativa al contenido de la
    # imagen: en espectrogramas con mucho fondo sube el umbral.
    threshold = float(np.percentile(smoothed, params.threshold_percentile))
    binary = smoothed > threshold

    # 3. Closing para unir bursts contiguos, opening para eliminar speckle.
    k = params.morph_kernel
    if k > 1:
        struct = np.ones((k, k), dtype=bool)
        binary = ndimage.binary_closing(binary, structure=struct)
        binary = ndimage.binary_opening(binary, structure=struct)

    # 4. Componentes conectados.
    labeled, num = ndimage.label(binary)
    if num == 0:
        return []

    # 5. Bounding box por componente, con filtros de area.
    slices = ndimage.find_objects(labeled)
    image_area = h * w
    max_area_px = int(image_area * params.max_area_frac)

    bboxes: list[tuple[int, int, int, int]] = []
    for sl in slices:
        if sl is None:
            continue
        ys, xs = sl
        y_min, y_max = int(ys.start), int(ys.stop)
        x_min, x_max = int(xs.start), int(xs.stop)
        bb_w = x_max - x_min
        bb_h = y_max - y_min
        area = bb_w * bb_h
        if area < params.min_area_px or area > max_area_px:
            continue
        # Filtro adicional: descartar cajas degeneradas de 1 pixel de ancho/alto.
        if bb_w < 2 or bb_h < 2:
            continue
        bboxes.append((x_min, y_min, x_max, y_max))

    return bboxes


def bbox_to_yolo(
    bbox: tuple[int, int, int, int],
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float]:
    """
    Convierte bbox (x_min, y_min, x_max, y_max) en pixeles al formato YOLO
    normalizado (cx, cy, w, h), donde todos los valores estan en [0, 1]
    respecto al ancho y alto de la imagen.
    """
    x_min, y_min, x_max, y_max = bbox
    cx = (x_min + x_max) / 2.0 / img_w
    cy = (y_min + y_max) / 2.0 / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h
    return cx, cy, w, h


def write_yolo_label(
    label_path: Path,
    class_id: int,
    bboxes_yolo: list[tuple[float, float, float, float]],
) -> None:
    """
    Escribe un .txt en formato YOLO.

    Una linea por bbox: 'class_id cx cy w h' con 6 decimales.
    Si no hay bboxes, escribe un fichero vacio (negativo de entrenamiento).
    """
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
        for (cx, cy, bw, bh) in bboxes_yolo
    ]
    label_path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")


def render_preview(
    img_path: Path,
    bboxes: list[tuple[int, int, int, int]],
    class_name: str,
    save_path: Path,
) -> None:
    """Dibuja las bboxes detectadas sobre el PNG y lo guarda para QA visual."""
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    color = "red" if class_name == "drone" else "lime"
    for (x_min, y_min, x_max, y_max) in bboxes:
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=2)
    draw.text((5, 5), f"{class_name} | n={len(bboxes)}", fill=color)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(save_path)


# ---------------------------------------------------------------------------
# Generacion del dataset YOLO (data.yaml + train/val/test.txt)
# ---------------------------------------------------------------------------
def write_data_yaml(yolo_dir: Path, root_dir: Path) -> None:
    """
    Genera data.yaml en el formato esperado por ultralytics YOLO.

    El campo 'path' es absoluto al root del dataset; los splits son rutas
    relativas a 'path' apuntando a ficheros .txt con listas de imagenes.
    """
    lines = [
        "# Dataset YOLO generado por auto_label_bursts.py",
        f"path: {root_dir.resolve().as_posix()}",
        "train: yolo/train.txt",
        "val: yolo/val.txt",
        "test: yolo/test.txt",
        "",
        "names:",
    ]
    for idx in sorted(YOLO_ID_TO_NAME):
        lines.append(f"  {idx}: {YOLO_ID_TO_NAME[idx]}")
    (yolo_dir / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_split_txt(
    split_csv: Path,
    yolo_split_txt: Path,
    root_dir: Path,
) -> int:
    """
    Lee un split CSV (con columna 'filename') y escribe la lista de paths
    ABSOLUTOS de imagenes en el .txt del split YOLO. Devuelve el numero de
    imagenes.

    IMPORTANTE: Ultralytics resuelve las rutas relativas en .txt respecto a
    la posicion del propio .txt, no respecto al campo 'path' del data.yaml.
    Para evitar ambiguedades en Windows (y porque el dataset no necesita
    portabilidad), escribimos paths absolutos.

    Ultralytics buscara las labels reemplazando .png por .txt en el mismo
    directorio del PNG.
    """
    if not split_csv.exists():
        print(f"  [WARN] No existe {split_csv}, no se genera {yolo_split_txt.name}")
        return 0
    df = pd.read_csv(split_csv)
    if "filename" not in df.columns:
        raise ValueError(f"{split_csv} no tiene columna 'filename'")
    yolo_split_txt.parent.mkdir(parents=True, exist_ok=True)
    # Convertir cada filename relativo a path absoluto usando root_dir
    abs_paths = [
        str((root_dir / rel).resolve()) for rel in df["filename"].astype(str)
    ]
    yolo_split_txt.write_text(
        "\n".join(abs_paths) + "\n",
        encoding="utf-8",
    )
    return len(df)


# ---------------------------------------------------------------------------
# Bucle principal
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    metadata_csv = Path(args.metadata_csv).resolve()
    root_dir = Path(args.root_dir).resolve()
    splits_dir = Path(args.splits_dir).resolve() if args.splits_dir else root_dir / "metadata"
    yolo_dir = root_dir / "yolo"
    preview_dir = yolo_dir / "preview"

    if not metadata_csv.exists():
        raise FileNotFoundError(f"No existe {metadata_csv}")
    if not root_dir.exists():
        raise FileNotFoundError(f"No existe {root_dir}")
    yolo_dir.mkdir(parents=True, exist_ok=True)

    params = BboxParams(
        threshold_percentile=args.threshold_percentile,
        min_area_px=args.min_area,
        max_area_frac=args.max_area_frac,
        morph_kernel=args.morph_kernel,
        smooth_sigma=args.smooth_sigma,
    )

    print(f"CSV maestro:   {metadata_csv}")
    print(f"Root dataset:  {root_dir}")
    print(f"Salida YOLO:   {yolo_dir}")
    print(f"Parametros:    {params}")
    print()

    df = pd.read_csv(metadata_csv)

    if args.preview_random and args.preview_n > 0:
        df = df.sample(frac=1.0, random_state=args.preview_seed).reset_index(drop=True)
        
    required = {"filename", "label", "capture_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en {metadata_csv}: {sorted(missing)}")

    unknown = set(df["label"].unique()) - set(CLASS_TO_YOLO_ID)
    if unknown:
        raise ValueError(
            f"Etiquetas no soportadas: {unknown}. "
            f"Esperadas: {set(CLASS_TO_YOLO_ID)}"
        )

    if args.limit is not None:
        df = df.head(args.limit).copy()
        print(f"[DEBUG] Limitando a las primeras {args.limit} filas")

    # -----------------------------------------------------------------------
    # 1. Generacion de labels para todos los PNGs (o saltarse si solo
    #    queremos regenerar el manifest YOLO con paths absolutos)
    # -----------------------------------------------------------------------
    if args.regenerate_splits_only:
        print("[regenerate-splits-only] Salto del etiquetado. Solo regenero "
              "data.yaml + train/val/test.txt con paths absolutos.\n")
        print("Generando manifest YOLO...")
        write_data_yaml(yolo_dir, root_dir)
        total_split = 0
        for split in ("train", "val", "test"):
            split_csv = splits_dir / f"{split}.csv"
            out_txt = yolo_dir / f"{split}.txt"
            n = write_split_txt(split_csv, out_txt, root_dir)
            if n > 0:
                print(f"  {split}.txt: {n} imagenes (paths absolutos)")
            total_split += n
        print(f"  TOTAL en splits: {total_split}")
        print(f"\ndata.yaml:         {yolo_dir / 'data.yaml'}")
        print("Listo. Vuelve a ejecutar train_yolo.py.")
        return

    print(f"Procesando {len(df)} espectrogramas...")
    report_rows: list[dict] = []
    n_preview_per_class: dict[str, int] = {c: 0 for c in CLASS_TO_YOLO_ID}

    for i, row in df.iterrows():
        rel_path = row["filename"]
        label = row["label"]
        class_id = CLASS_TO_YOLO_ID[label]
        img_path = root_dir / rel_path
        label_path = img_path.with_suffix(".txt")

        if not img_path.exists():
            print(f"  [WARN] No existe {img_path}, se omite")
            continue

        energy, img_w, img_h = png_to_energy(img_path)
        bboxes_px = extract_bboxes(energy, params)
        bboxes_yolo_norm = [bbox_to_yolo(bb, img_w, img_h) for bb in bboxes_px]
        write_yolo_label(label_path, class_id, bboxes_yolo_norm)

        report_rows.append({
            "filename": rel_path,
            "label": label,
            "class_id": class_id,
            "n_bursts": len(bboxes_px),
            "img_w": img_w,
            "img_h": img_h,
            "capture_id": row["capture_id"],
        })

        # Previews: hasta args.preview-n por clase, balanceando capturas.
        if (
            args.preview_n > 0
            and n_preview_per_class[label] < args.preview_n
            and len(bboxes_px) > 0
        ):
            stem = Path(rel_path).stem
            render_preview(img_path, bboxes_px, label, preview_dir / f"{label}_{stem}.png")
            n_preview_per_class[label] += 1

        if (i + 1) % 200 == 0:
            print(f"  procesados {i + 1}/{len(df)}")

    print(f"  procesados {len(df)}/{len(df)}")

    # -----------------------------------------------------------------------
    # 2. Generar data.yaml y train.txt / val.txt / test.txt
    # -----------------------------------------------------------------------
    print()
    print("Generando manifest YOLO...")
    write_data_yaml(yolo_dir, root_dir)

    total_split = 0
    for split in ("train", "val", "test"):
        split_csv = splits_dir / f"{split}.csv"
        out_txt = yolo_dir / f"{split}.txt"
        n = write_split_txt(split_csv, out_txt, root_dir)
        if n > 0:
            print(f"  {split}.txt: {n} imagenes")
        total_split += n
    print(f"  TOTAL en splits: {total_split}")

    # -----------------------------------------------------------------------
    # 3. Reporte de labeling
    # -----------------------------------------------------------------------
    report_df = pd.DataFrame(report_rows)
    report_csv = yolo_dir / "labeling_report.csv"
    report_df.to_csv(report_csv, index=False)

    print()
    print("Resumen del etiquetado:")
    by_class = report_df.groupby("label")["n_bursts"].agg(["count", "sum", "mean", "min", "max"])
    print(by_class.to_string())

    n_empty = (report_df["n_bursts"] == 0).sum()
    print(f"\nPNGs sin ninguna caja detectada: {n_empty} ({n_empty / len(report_df):.1%})")
    print(f"Reporte detallado: {report_csv}")
    print(f"data.yaml:         {yolo_dir / 'data.yaml'}")
    if args.preview_n > 0:
        print(f"Previews:          {preview_dir}")
    print("\nListo. Siguiente paso: revisar las previews visualmente y, si la")
    print("calidad de las cajas es razonable, entrenar YOLOv8 con data.yaml.")


if __name__ == "__main__":
    main()
