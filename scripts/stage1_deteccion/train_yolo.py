"""
train_yolo.py
=============

Entrena YOLOv8 (stage 1 del pipeline two-stage) sobre el dataset v3
auto-etiquetado por auto_label_bursts.py. El detector aprende a localizar
bursts en espectrogramas y a clasificarlos en:
  0 = interference  (bursts FHSS de BT o tramas WiFi cortas)
  1 = drone         (bursts FHSS de OcuSync del DJI MINI3)

Pre-requisitos:
  1. Ejecutar antes scripts/auto_label_bursts.py para generar los .txt YOLO
     y el data.yaml en data/mini3_detector_python_v3/yolo/.
  2. Anadir ultralytics a pyproject.toml:
         dependencies = [..., "ultralytics>=8.3"]
     y ejecutar 'uv sync'.

Uso tipico (GPU RTX 4070 Laptop, batch razonable):
    uv run python scripts/train_yolo.py
    uv run python scripts/train_yolo.py --model yolov8s.pt --epochs 80

Salidas (en artifacts-dir/name/):
    weights/best.pt          Mejor modelo (segun mAP val)
    weights/last.pt          Ultimo epoch
    results.csv              Metricas por epoch
    confusion_matrix.png     Matriz de confusion del detector
    P_curve.png, R_curve.png, F1_curve.png, PR_curve.png
    val_batch*.jpg           Ejemplos de predicciones en validacion
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_DATA_YAML = Path("data/mini3_detector_python_v3/yolo/data.yaml")
DEFAULT_ARTIFACTS = Path("data/mini3_detector_python_v3/artifacts/yolo")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--data-yaml",
        type=Path,
        default=DEFAULT_DATA_YAML,
        help="data.yaml generado por auto_label_bursts.py",
    )
    p.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS,
        help="Carpeta padre donde YOLO crea el subdirectorio del run.",
    )
    p.add_argument(
        "--name",
        default="run01",
        help="Nombre del run (subcarpeta dentro de artifacts-dir).",
    )
    p.add_argument(
        "--model",
        default="yolov8n.pt",
        help=(
            "Modelo base YOLO. Opciones razonables: "
            "yolov8n.pt (nano, ~3M params, mas rapido), "
            "yolov8s.pt (small, ~11M, mejor mAP), "
            "yolov8m.pt (medium, ~25M, slow pero mejor)."
        ),
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help=(
            "Tamano al que YOLO reescala las imagenes. 640 es el estandar; "
            "como nuestros PNGs son 1024x576, YOLO los letterboxa a 640x640."
        ),
    )
    p.add_argument("--batch", type=int, default=16)
    p.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping si no hay mejora de mAP en N epochs consecutivos.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device",
        default=None,
        help="cuda:0, cpu, etc. Auto-detect si no se indica.",
    )
    p.add_argument(
        "--no-eval-test",
        action="store_true",
        default=False,
        help="Omitir evaluacion en test al terminar.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit(
            "Falta la dependencia 'ultralytics'.\n"
            "Anadela a pyproject.toml en dependencies:\n"
            "    \"ultralytics>=8.3\",\n"
            "y ejecuta 'uv sync' para instalar."
        ) from e

    if not args.data_yaml.exists():
        raise FileNotFoundError(
            f"No existe {args.data_yaml}. Ejecuta primero "
            f"scripts/auto_label_bursts.py para generar el dataset YOLO."
        )

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    print(f"Modelo base: {args.model}")
    print(f"data.yaml:   {args.data_yaml.resolve()}")
    print(f"Artifacts:   {args.artifacts_dir.resolve()}")
    print(f"Run name:    {args.name}")
    print(f"Epochs:      {args.epochs} | imgsz: {args.imgsz} | batch: {args.batch}")
    print(f"Patience:    {args.patience}")
    print(f"Seed:        {args.seed}")
    print()

    model = YOLO(args.model)

    # -----------------------------------------------------------------------
    # Entrenamiento
    # -----------------------------------------------------------------------
    train_results = model.train(
        data=str(args.data_yaml.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        seed=args.seed,
        project=str(args.artifacts_dir.resolve()),
        name=args.name,
        exist_ok=True,
        device=args.device,
        verbose=True,
        save=True,
        save_period=10,
        # Augmentations conservadoras: los ejes tiempo/frecuencia tienen
        # significado fisico. Desactivamos flips y rotaciones; mantenemos
        # mosaic/mixup que son operaciones sobre imagenes completas.
        fliplr=0.0,
        flipud=0.0,
        degrees=0.0,
        scale=0.05,
        translate=0.02,
        shear=0.0,
        perspective=0.0,
    )

    run_dir = args.artifacts_dir / args.name
    best_weights = run_dir / "weights" / "best.pt"

    print(f"\nEntrenamiento completado.")
    print(f"Best weights:  {best_weights}")
    print(f"Logs y plots:  {run_dir}")

    # -----------------------------------------------------------------------
    # Evaluacion sobre el split de test
    # -----------------------------------------------------------------------
    if not args.no_eval_test:
        print("\nEvaluacion sobre el split de test...")
        best_model = YOLO(str(best_weights))
        test_metrics = best_model.val(
            data=str(args.data_yaml.resolve()),
            split="test",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=str(args.artifacts_dir.resolve()),
            name=f"{args.name}_test",
            exist_ok=True,
            verbose=True,
        )

        print("\nMetricas sobre test:")
        print(f"  mAP@50:        {test_metrics.box.map50:.4f}")
        print(f"  mAP@50-95:     {test_metrics.box.map:.4f}")
        print(f"  Precision:     {test_metrics.box.mp:.4f}")
        print(f"  Recall:        {test_metrics.box.mr:.4f}")

        # Metricas por clase
        if hasattr(test_metrics.box, "ap50") and len(test_metrics.box.ap50) > 0:
            print("\nMetricas por clase:")
            for i, cls_idx in enumerate(test_metrics.box.ap_class_index):
                cls_name = test_metrics.names.get(int(cls_idx), str(cls_idx))
                print(
                    f"  {cls_name:<15s} "
                    f"mAP@50={test_metrics.box.ap50[i]:.4f}  "
                    f"mAP@50-95={test_metrics.box.ap[i]:.4f}"
                )

        # Volcar metricas a JSON para la memoria
        results_json = run_dir / "test_metrics.json"
        with open(results_json, "w") as f:
            json.dump(
                {
                    "split": "test",
                    "imgsz": args.imgsz,
                    "model": args.model,
                    "best_weights": str(best_weights),
                    "mAP50": float(test_metrics.box.map50),
                    "mAP50_95": float(test_metrics.box.map),
                    "precision": float(test_metrics.box.mp),
                    "recall": float(test_metrics.box.mr),
                    "per_class": {
                        test_metrics.names.get(int(cls_idx), str(cls_idx)): {
                            "mAP50": float(test_metrics.box.ap50[i]),
                            "mAP50_95": float(test_metrics.box.ap[i]),
                        }
                        for i, cls_idx in enumerate(test_metrics.box.ap_class_index)
                    },
                },
                f,
                indent=2,
            )
        print(f"\nMetricas guardadas en {results_json}")


if __name__ == "__main__":
    main()
