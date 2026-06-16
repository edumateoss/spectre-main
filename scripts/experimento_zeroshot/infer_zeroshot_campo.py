"""
infer_zeroshot_campo.py
=======================

Evaluacion zero-shot del detector YOLOv8 (entrenado solo sobre DJI Mini 3,
dataset v3) sobre los espectrogramas del dia de campo outdoor (F450, Mavic
Pro, Hunter).

El modelo NO se reentrena. Se evalua directamente si los pesos del Stage 1
(run02_60ep, entrenado solo con OcuSync del Mini 3) generalizan y detectan
rafagas FHSS de otros protocolos/drones sin haber visto ningun ejemplo de
ellos durante el entrenamiento.

Hipotesis positiva: si el modelo reconoce bursts FHSS de F450, Mavic y
Hunter como "drone", demuestra que ha aprendido morfologia FHSS generica,
no la firma especifica del Mini 3.

Metricas por tipo de dron:
  - Recall de presencia: % de imagenes donde el modelo detecta >=K rafagas
  - Media de cajas drone por imagen
  - Distribucion de clases predichas

Pre-requisito:
  Ejecutar primero make_spectrograms_campo.py (al menos con --quick o
  --skip-interference --max-segs 50) para generar las imagenes PNG.

Salida:
  data/dataset_campo_v1/artifacts/zeroshot/
    zeroshot_report.csv       Una fila por imagen con conteos de deteccion
    zeroshot_summary.json     Resumen agregado por tipo de dron
    ejemplos/{drone_type}/    Imagenes con cajas dibujadas

Uso:
  uv run python scripts/infer_zeroshot_campo.py
  uv run python scripts/infer_zeroshot_campo.py --conf 0.25 --k 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


DEFAULT_WEIGHTS = Path(
    "data/mini3_detector_python_v3/artifacts/yolo/run02_60ep/weights/best.pt"
)
# Los espectrogramas de campo ya existen en stage2_classifier_v1
DEFAULT_DATASET = Path("data/stage2_classifier_v1")
DEFAULT_CSV = DEFAULT_DATASET / "metadata" / "spectrograms_stage2.csv"

# Etiquetas de dron presentes en el CSV del Stage 2
DRONE_LABELS = {"f450", "mavic", "hunter"}

# Clases del modelo v3 entrenado
CLASS_NAMES: dict[int, str] = {0: "interference", 1: "drone"}
COLOR_POR_CLASE: dict[int, str] = {0: "lime", 1: "red"}


# ---------------------------------------------------------------------------
# Dibujo de cajas sobre PNG
# ---------------------------------------------------------------------------
def dibujar_cajas(
    img_path: Path,
    xyxyn: np.ndarray,
    clases: list[int],
    confs: list[float],
    save_path: Path,
) -> None:
    """
    Guarda una copia del espectrograma con las cajas predichas dibujadas.
    xyxyn tiene shape (N, 4) con coordenadas normalizadas [x1, y1, x2, y2].
    """
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    for i, (cls, conf) in enumerate(zip(clases, confs)):
        x1, y1, x2, y2 = xyxyn[i]
        color = COLOR_POR_CLASE.get(cls, "white")
        draw.rectangle(
            [x1 * w, y1 * h, x2 * w, y2 * h],
            outline=color,
            width=2,
        )
        draw.text(
            (x1 * w + 2, y1 * h + 2),
            f"{CLASS_NAMES.get(cls, str(cls))} {conf:.2f}",
            fill=color,
        )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(save_path)


# ---------------------------------------------------------------------------
# Argumentos de linea de comandos
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Pesos del modelo YOLO a evaluar (best.pt del run02_60ep).",
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET,
        help="Directorio raiz del dataset (por defecto: stage2_classifier_v1).",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="CSV de metadatos con columnas filename, label, capture_id.",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.10,
        help=(
            "Umbral de confianza minimo para retener una deteccion. "
            "0.10 es el punto operativo recomendado del Stage 1 en v3."
        ),
    )
    p.add_argument(
        "--k",
        type=int,
        default=1,
        help=(
            "Minimo de cajas 'drone' para declarar presencia en una imagen. "
            "K=1 es el criterio mas sensible."
        ),
    )
    p.add_argument(
        "--n-examples",
        type=int,
        default=6,
        help="Numero de imagenes de ejemplo a guardar por tipo de dron.",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Dispositivo de inferencia: 'cuda:0', 'cpu', etc. Auto si no se indica.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit(
            "Falta la dependencia 'ultralytics'. Ejecuta 'uv sync'."
        ) from e

    dataset_dir = args.dataset_dir.resolve()
    weights_path = args.weights.resolve()
    csv_path = args.csv.resolve()
    artifacts_dir = dataset_dir / "artifacts" / "zeroshot"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"No existe {csv_path}.\n"
            "Verifica la ruta con --csv."
        )
    if not weights_path.exists():
        raise FileNotFoundError(
            f"No existen los pesos del modelo: {weights_path}.\n"
            "Verifica que el Stage 1 (run02_60ep) fue entrenado correctamente."
        )

    df = pd.read_csv(csv_path)

    # El CSV del Stage 2 usa la etiqueta del tipo de dron directamente
    # (f450, mavic, hunter). Filtramos solo esas filas.
    df_drones = df[df["label"].isin(DRONE_LABELS)].copy().reset_index(drop=True)
    # Columna drone_type = label en este CSV
    df_drones["drone_type"] = df_drones["label"]

    if df_drones.empty:
        raise RuntimeError(
            f"El CSV no contiene imagenes de dron ({DRONE_LABELS}). "
            "Verifica la ruta con --csv."
        )

    print(f"Modelo:       {weights_path}")
    print(f"Conf umbral:  {args.conf}")
    print(f"K minimo:     {args.k}")
    print(f"Imagenes:     {len(df_drones)}")
    print(df_drones.groupby("drone_type").size().rename("n_imagenes").to_string())
    print()

    model = YOLO(str(weights_path))

    # Verificar que las clases del modelo coinciden con lo esperado
    model_names = model.names
    print(f"Clases del modelo: {model_names}")
    print()

    report_rows: list[dict] = []
    ejemplos_por_tipo: dict[str, int] = {}

    for i, row in df_drones.iterrows():
        img_path = dataset_dir / row["filename"]
        if not img_path.exists():
            print(f"  [AVISO] No existe {img_path}, se omite.")
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
            clases = r.boxes.cls.cpu().numpy().astype(int).tolist()
            confs_det = r.boxes.conf.cpu().numpy().tolist()
            xyxyn = r.boxes.xyxyn.cpu().numpy()
            n_drone = clases.count(1)
            n_interf = clases.count(0)
        else:
            clases = []
            confs_det = []
            xyxyn = np.empty((0, 4))
            n_drone = 0
            n_interf = 0

        drone_type = row["drone_type"]
        detectado = n_drone >= args.k

        report_rows.append({
            "filename": row["filename"],
            "drone_type": drone_type,
            "capture_id": row["capture_id"],
            "n_boxes_total": len(clases),
            "n_boxes_drone": n_drone,
            "n_boxes_interf": n_interf,
            "detectado": detectado,
        })

        # Guardar ejemplos visuales (solo imagenes con al menos una deteccion)
        guardados = ejemplos_por_tipo.get(drone_type, 0)
        if guardados < args.n_examples and len(clases) > 0:
            save_path = (
                artifacts_dir
                / "ejemplos"
                / drone_type
                / f"{Path(row['filename']).stem}.png"
            )
            dibujar_cajas(img_path, xyxyn, clases, confs_det, save_path)
            ejemplos_por_tipo[drone_type] = guardados + 1

        if (i + 1) % 50 == 0:
            print(f"  Procesadas {i + 1}/{len(df_drones)} imagenes...")

    print(f"  Procesadas {len(df_drones)}/{len(df_drones)} imagenes.")

    report_df = pd.DataFrame(report_rows)
    report_csv = artifacts_dir / "zeroshot_report.csv"
    report_df.to_csv(report_csv, index=False)

    # -----------------------------------------------------------------------
    # Resumen por tipo de dron
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print(f"RESULTADO ZERO-SHOT  (conf={args.conf}, K={args.k})")
    print("=" * 60)
    print(
        f"{'Tipo dron':<12}  {'N imgs':>6}  {'Detectado':>9}  "
        f"{'Recall%':>8}  {'Avg cajas':>9}"
    )
    print("-" * 60)

    summary: dict[str, dict] = {}
    for drone_type, grp in report_df.groupby("drone_type"):
        n_imgs = len(grp)
        n_det = int(grp["detectado"].sum())
        recall_pct = 100.0 * n_det / n_imgs if n_imgs > 0 else 0.0
        avg_boxes = float(grp["n_boxes_drone"].mean())

        print(
            f"{drone_type:<12}  {n_imgs:>6}  {n_det:>9}  "
            f"{recall_pct:>7.1f}%  {avg_boxes:>9.1f}"
        )

        summary[drone_type] = {
            "n_imagenes": n_imgs,
            "n_detectado": n_det,
            "recall_presencia_pct": round(recall_pct, 1),
            "avg_cajas_drone_por_imagen": round(avg_boxes, 1),
        }

    print("=" * 60)

    # Comparacion con el resultado en v3 (Mini 3 entrenado)
    print()
    print("Referencia v3 (Mini 3 entrenado, conf=0.10, K=1): recall=99.0%")
    print()

    summary_path = artifacts_dir / "zeroshot_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "conf": args.conf,
                "k": args.k,
                "referencia_v3_recall_pct": 99.0,
                "por_tipo": summary,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Reporte por imagen:  {report_csv}")
    print(f"Resumen JSON:        {summary_path}")
    print(f"Ejemplos visuales:   {artifacts_dir / 'ejemplos'}")


if __name__ == "__main__":
    main()
