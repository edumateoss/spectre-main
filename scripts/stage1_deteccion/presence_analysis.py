"""
presence_analysis.py
====================

Analisis de DETECCION DE PRESENCIA DE DRON a nivel de imagen sobre el
test set del Stage 1 (YOLOv8 run02_60ep).

Pregunta operativa que responde este script:

    "Dada una ventana de 0.1 s de espectrograma, el sistema confirma la
    presencia del dron si encuentra AL MENOS K bursts etiquetados como
    drone (con K configurable). Exigir varios bursts en lugar de uno
    solo es mas robusto frente a falsos positivos puntuales, a costa
    de exigir mas evidencia en cada ventana."

A diferencia de las metricas mAP/Precision/Recall que reporta YOLO por
defecto (calculadas sobre bounding boxes individuales), este script
calcula metricas binarias a nivel de imagen sobre un grid bidimensional
de configuraciones (conf_threshold, min_drone_bursts):

    GT:   imagen es de captura `mini3_drone_*`   -> presencia=1
          imagen es de captura `mini3_nodrone_*` -> presencia=0
    PRED: la imagen tiene >=K bboxes de clase drone con conf >= thr,
          donde (thr, K) recorre el producto cartesiano de --conf
          y --min-drone-bursts.

Salidas en --output-dir:

    per_image_predictions.csv  Una fila por imagen del test set:
        path, capture_id, capture_type, distance_m, wifi_level,
        gt_presence, n_drone_bboxes, n_interf_bboxes, max_drone_conf,
        max_interf_conf

    presence_metrics.json      Lista de configuraciones evaluadas, cada
        una con TP/FP/TN/FN, accuracy, precision, recall, F1,
        specificity, FPR y desglose por captura.

    presence_report.txt        Lectura legible del JSON con todas las
        combinaciones (conf_thr, min_bursts) en orden de mas permisivo
        a mas estricto.

Uso:

    uv run python scripts/presence_analysis.py
    uv run python scripts/presence_analysis.py --conf 0.10 0.25 0.40
    uv run python scripts/presence_analysis.py --min-drone-bursts 1 3 5
    uv run python scripts/presence_analysis.py --weights data/.../run02_60ep/weights/best.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

DEFAULT_WEIGHTS = Path(
    "data/mini3_detector_python_v3/artifacts/yolo/run02_60ep/weights/best.pt"
)
DEFAULT_TEST_TXT = Path("data/mini3_detector_python_v3/yolo/test.txt")
DEFAULT_OUTPUT = Path(
    "data/mini3_detector_python_v3/artifacts/yolo/run02_60ep_test/presence"
)

# Clases YOLO: ver data.yaml
CLS_INTERFERENCE = 0
CLS_DRONE = 1

# Umbrales de confianza a evaluar (sweep). El default 0.25 coincide con el
# usado por YOLOv8 en model.val(), asi las cifras seran comparables con la
# matriz de confusion del YOLO_PROGRESO.md.
DEFAULT_CONF_THRS = [0.10, 0.25, 0.40, 0.50]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--test-txt", type=Path, default=DEFAULT_TEST_TXT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None, help="cuda:0, cpu, etc.")
    p.add_argument(
        "--conf",
        nargs="+",
        type=float,
        default=DEFAULT_CONF_THRS,
        help="Umbrales de confianza a evaluar (sweep).",
    )
    p.add_argument(
        "--inference-conf",
        type=float,
        default=0.001,
        help=(
            "Confianza minima en la inferencia. Se deja muy baja para "
            "que el sweep posterior pueda evaluar varios umbrales sin "
            "necesidad de reinferir."
        ),
    )

    p.add_argument(
        "--min-drone-bursts",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4],
        help=(
            "Numero minimo de bboxes drone necesarios para confirmar presencia. "
            "Ejemplo: 1 significa al menos un burst; 3 exige al menos tres bursts."
        ),
    )


    p.add_argument(
        "--iou-nms",
        type=float,
        default=0.7,
        help="IoU threshold para NMS en la inferencia.",
    )
    return p.parse_args()


# ----------------------------------------------------------------------
# Parsing del nombre de captura
# ----------------------------------------------------------------------
# Formatos esperados:
#   mini3_drone_d{D}_w{W}_s{S}_cap{NN}__seg{IIII}.png
#   mini3_nodrone_w{W}_s{S}_cap{NN}__seg{IIII}.png
_RE_DRONE = re.compile(
    r"^mini3_drone_d(?P<d>\d+)_w(?P<w>\d+)_s\d+_cap\d+"
)
_RE_NODRONE = re.compile(
    r"^mini3_nodrone_w(?P<w>\d+)_s\d+_cap\d+"
)


def parse_capture(filename: str) -> dict:
    """Extrae capture_id, capture_type, distance_m, wifi_level."""
    base = Path(filename).name
    capture_id = base.split("__")[0]
    if base.startswith("mini3_drone_"):
        m = _RE_DRONE.match(capture_id)
        return {
            "capture_id": capture_id,
            "capture_type": "drone",
            "distance_m": int(m.group("d")) if m else None,
            "wifi_level": int(m.group("w")) if m else None,
        }
    if base.startswith("mini3_nodrone_"):
        m = _RE_NODRONE.match(capture_id)
        return {
            "capture_id": capture_id,
            "capture_type": "nodrone",
            "distance_m": None,
            "wifi_level": int(m.group("w")) if m else None,
        }
    return {
        "capture_id": capture_id,
        "capture_type": "unknown",
        "distance_m": None,
        "wifi_level": None,
    }


# ----------------------------------------------------------------------
# Inferencia + tabla por imagen
# ----------------------------------------------------------------------
def run_inference(weights: Path, image_paths: list[str], imgsz: int,
                  device: str | None, inference_conf: float, iou_nms: float):
    """Devuelve una lista de dicts, uno por imagen del test set."""
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit(
            "Falta ultralytics. Instala con uv (ya esta en pyproject.toml)."
        ) from e

    print(f"Cargando {weights}...")
    model = YOLO(str(weights))

    # Procesamos imagen a imagen para no inflar memoria
    rows: list[dict] = []
    for i, raw_path in enumerate(image_paths):
        # Normaliza separador para Linux/Windows
        path = raw_path.replace("\\", "/")
        meta = parse_capture(path)

        result_list = model.predict(
            source=path,
            imgsz=imgsz,
            conf=inference_conf,
            iou=iou_nms,
            device=device,
            verbose=False,
            save=False,
        )
        res = result_list[0]

        # Cajas detectadas (cualquier confianza > inference_conf)
        boxes = res.boxes
        if boxes is None or boxes.cls is None or len(boxes) == 0:
            cls_arr = []
            conf_arr = []
        else:
            cls_arr = boxes.cls.cpu().numpy().astype(int).tolist()
            conf_arr = boxes.conf.cpu().numpy().tolist()

        # Listas por clase
        drone_confs = [c for cls, c in zip(cls_arr, conf_arr) if cls == CLS_DRONE]
        interf_confs = [c for cls, c in zip(cls_arr, conf_arr) if cls == CLS_INTERFERENCE]

        rows.append({
            "path": path,
            "capture_id": meta["capture_id"],
            "capture_type": meta["capture_type"],
            "distance_m": meta["distance_m"],
            "wifi_level": meta["wifi_level"],
            "gt_presence": 1 if meta["capture_type"] == "drone" else 0,
            "n_drone_bboxes": len(drone_confs),
            "n_interf_bboxes": len(interf_confs),
            "max_drone_conf": max(drone_confs) if drone_confs else 0.0,
            "max_interf_conf": max(interf_confs) if interf_confs else 0.0,
            "drone_confs": drone_confs,    # se usa para el sweep
            "interf_confs": interf_confs,
        })

        if (i + 1) % 50 == 0:
            print(f"  procesadas {i+1}/{len(image_paths)} imagenes")
    return rows


# ----------------------------------------------------------------------
# Metricas a un umbral
# ----------------------------------------------------------------------
def compute_presence_metrics(
    rows: list[dict], conf_thr: float, min_bursts: int = 1
) -> dict:
    """Calcula metricas binarias de presencia a una configuracion dada.

    Una imagen se considera "presencia drone" si y solo si tiene al menos
    `min_bursts` bboxes de clase drone con confianza >= `conf_thr`.
    """
    tp = fp = tn = fn = 0
    per_capture: dict[str, dict] = {}

    for r in rows:
        # Presencia predicha: >= min_bursts bboxes drone con conf >= thr
        n_drone_hits = sum(1 for c in r["drone_confs"] if c >= conf_thr)
        pred = n_drone_hits >= min_bursts
        gt = bool(r["gt_presence"])

        if gt and pred:      tp += 1
        elif gt and not pred: fn += 1
        elif not gt and pred: fp += 1
        else:                 tn += 1

        cap = r["capture_id"]
        if cap not in per_capture:
            per_capture[cap] = {
                "capture_type": r["capture_type"],
                "distance_m": r["distance_m"],
                "wifi_level": r["wifi_level"],
                "n_images": 0,
                "n_pred_present": 0,
            }
        per_capture[cap]["n_images"] += 1
        if pred:
            per_capture[cap]["n_pred_present"] += 1

    n_pos = tp + fn   # imagenes con dron real
    n_neg = tn + fp   # imagenes sin dron real

    def safe_div(a, b):
        return a / b if b > 0 else 0.0

    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, n_pos)            # sensibilidad / TPR
    specificity = safe_div(tn, n_neg)       # TNR
    fpr = safe_div(fp, n_neg)               # 1 - specificity
    f1 = safe_div(2 * precision * recall, precision + recall)

    # Resumen por captura: fraccion de imagenes en las que se detecta
    # presencia y veredicto "captura confirmada" si >=1 imagen positiva.
    for cap, d in per_capture.items():
        d["pct_pred_present"] = round(100.0 * d["n_pred_present"] / d["n_images"], 2)
        d["capture_confirmed"] = d["n_pred_present"] >= 1

    return {
        "conf_threshold": conf_thr,
        "min_drone_bursts": min_bursts,
        "counts": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "n_drone_imgs": n_pos,
        "n_nodrone_imgs": n_neg,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "false_positive_rate": round(fpr, 4),
        "f1": round(f1, 4),
        "per_capture": per_capture,
    }


# ----------------------------------------------------------------------
# CSV por imagen
# ----------------------------------------------------------------------
def write_per_image_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "path", "capture_id", "capture_type", "distance_m", "wifi_level",
        "gt_presence", "n_drone_bboxes", "n_interf_bboxes",
        "max_drone_conf", "max_interf_conf",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})


# ----------------------------------------------------------------------
# Reporte legible
# ----------------------------------------------------------------------
def format_report(all_metrics: list[dict]) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append("ANALISIS DE PRESENCIA DE DRON - test set (600 imgs)")
    lines.append("=" * 78)

    # 1) Tabla resumen del sweep completo (conf x min_bursts)
    lines.append("")
    lines.append("Tabla resumen (sweep conf_threshold x min_drone_bursts):")
    lines.append("")
    lines.append(
        f"  {'conf':>5s}  {'K':>3s}  {'TP':>4s}  {'FN':>4s}  {'TN':>4s}  {'FP':>4s}  "
        f"{'acc':>6s}  {'prec':>6s}  {'rec':>6s}  {'spec':>6s}  {'F1':>6s}"
    )
    lines.append("  " + "-" * 76)
    for m in all_metrics:
        c = m["counts"]
        lines.append(
            f"  {m['conf_threshold']:>5.2f}  {m['min_drone_bursts']:>3d}  "
            f"{c['TP']:>4d}  {c['FN']:>4d}  {c['TN']:>4d}  {c['FP']:>4d}  "
            f"{m['accuracy']:>6.4f}  {m['precision']:>6.4f}  "
            f"{m['recall']:>6.4f}  {m['specificity']:>6.4f}  {m['f1']:>6.4f}"
        )

    # 2) Detalle por configuracion (incluye desglose por captura)
    for m in all_metrics:
        c = m["counts"]
        lines.append("")
        lines.append(
            f"-- conf={m['conf_threshold']:.2f}  "
            f"min_bursts={m['min_drone_bursts']} --"
        )
        lines.append(
            f"  Imagenes drone:    {m['n_drone_imgs']:4d}    "
            f"Imagenes nodrone: {m['n_nodrone_imgs']:4d}"
        )
        lines.append(f"  TP={c['TP']:4d}  FN={c['FN']:4d}    "
                     f"TN={c['TN']:4d}  FP={c['FP']:4d}")
        lines.append(f"  Accuracy:        {m['accuracy']:.4f}")
        lines.append(f"  Precision:       {m['precision']:.4f}")
        lines.append(f"  Recall (TPR):    {m['recall']:.4f}")
        lines.append(f"  Specificity:     {m['specificity']:.4f}")
        lines.append(f"  FPR:             {m['false_positive_rate']:.4f}")
        lines.append(f"  F1:              {m['f1']:.4f}")
        lines.append("  Resumen por captura:")
        for cap, d in m["per_capture"].items():
            tag = "drone" if d["capture_type"] == "drone" else "nodrone"
            extra = f"d={d['distance_m']}m" if d["distance_m"] else f"w={d['wifi_level']}"
            verdict = "CONFIRMADO" if d["capture_confirmed"] else "no detectado"
            lines.append(
                f"    {cap:<40s} [{tag:<7s} {extra:<8s}] "
                f"{d['n_pred_present']}/{d['n_images']} imgs positivas "
                f"({d['pct_pred_present']:5.2f}%) -> {verdict}"
            )
    lines.append("")
    lines.append("=" * 78)
    return "\n".join(lines)


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(f"No existe {args.weights}")
    if not args.test_txt.exists():
        raise FileNotFoundError(f"No existe {args.test_txt}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Lee las rutas del split de test
    with open(args.test_txt, "r", encoding="utf-8") as f:
        image_paths = [ln.strip() for ln in f if ln.strip()]
    print(f"Test set: {len(image_paths)} imagenes")

    # Inferencia
    rows = run_inference(
        weights=args.weights,
        image_paths=image_paths,
        imgsz=args.imgsz,
        device=args.device,
        inference_conf=args.inference_conf,
        iou_nms=args.iou_nms,
    )

    # CSV por imagen
    csv_path = args.output_dir / "per_image_predictions.csv"
    write_per_image_csv(rows, csv_path)
    print(f"Guardado per-image CSV en: {csv_path}")

    # Sweep bidimensional: producto cartesiano (conf_thr, min_bursts).
    # Ordenamos por (conf asc, K asc) para que la tabla resumen vaya de la
    # configuracion mas permisiva (conf bajo, pocos bursts requeridos) a
    # la mas estricta (conf alto, muchos bursts requeridos).
    all_metrics = []
    for thr in sorted(args.conf):
        for k in sorted(args.min_drone_bursts):
            all_metrics.append(
                compute_presence_metrics(rows, conf_thr=thr, min_bursts=k)
            )

    # JSON
    json_path = args.output_dir / "presence_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "sweep_conf": sorted(args.conf),
                "sweep_min_drone_bursts": sorted(args.min_drone_bursts),
                "metrics": all_metrics,
            },
            f,
            indent=2,
        )
    print(f"Guardado JSON en: {json_path}")

    # Reporte legible
    report = format_report(all_metrics)
    txt_path = args.output_dir / "presence_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Guardado reporte en: {txt_path}")
    print()
    print(report)


if __name__ == "__main__":
    main()
