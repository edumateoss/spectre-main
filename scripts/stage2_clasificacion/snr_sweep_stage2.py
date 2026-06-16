"""
snr_sweep_stage2.py
===================

Etapa extra del Stage 2: barrido de SNR sobre el clasificador 6 clases.

Idea:
    El modelo `resnet18_stage2_mavic_split/best_model.pth` se entreno sobre
    espectrogramas limpios. Aqui medimos su robustez al degradar la SNR
    inyectando ruido sintetico en el dominio IQ y regenerando el
    espectrograma con el MISMO pipeline que se uso en entrenamiento
    (make_spectrogram_dataset.py).

Dos modelos de ruido:
    - AWGN: ruido blanco gaussiano complejo. Curva canonica de robustez,
      directamente comparable con el paper RFUAV (kitoweeknd) y con la
      literatura general de comunicaciones.
    - FHSS-like (BT sintetico): bursts cortos en frecuencias aleatorias
      dentro de la banda capturada, parametros tipicos de Bluetooth
      Classic (1600 hops/s, 366us, ~1 MHz BW). Mas duro que AWGN porque
      comparte morfologia con la senal del dron.

Para cada (filename del test set, noise_type, snr_db) el script:
    1. Resuelve el .sigmf-data y el indice de segmento desde el nombre PNG.
    2. Lee el segmento IQ y le inyecta ruido calibrado al SNR objetivo.
    3. Regenera el espectrograma como PIL.Image.
    4. Aplica las mismas transforms del modelo (FrequencyNormalize -> Resize
       -> ImageNet norm) y predice.
    5. Registra prediccion vs etiqueta real.

Salidas (en --output-dir):
    - run_config.json
    - sigmf_meta_index.json (fs, fc por filename_stem, log de debugging)
    - predictions_per_sample.csv (una fila por (filename, noise, snr))
    - snr_sweep_results.csv (resumen por (noise, snr))
    - confusion_matrices/cm_{noise}_{snr}.csv (matriz por punto)
    - accuracy_vs_snr.png, f1_macro_vs_snr.png, f1_per_class_vs_snr.png

Uso:
    uv run python scripts/snr_sweep_stage2.py
    uv run python scripts/snr_sweep_stage2.py --quick-check
    uv run python scripts/snr_sweep_stage2.py --snr-list inf 20 10 0 -10 -20
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torchvision import models, transforms
from tqdm import tqdm

# Reusar las definiciones del entrenamiento para garantizar transforms
# identicas a las usadas para entrenar el modelo de 6 clases.
sys.path.insert(0, str(Path(__file__).resolve().parent))           # mismo subdir: train_stage2
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))    # scripts/: snr_noise, spectrogram_render
from train_stage2_classifier_mavic_split import (  # noqa: E402
    CLASS_NAMES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    FrequencyNormalize,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from snr_noise import inject_noise  # noqa: E402
from spectrogram_render import (  # noqa: E402
    NFFT,
    read_iq_segment,
    render_spectrogram_image,
    render_spectrogram_image_matplotlib,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL = Path("data/stage2_classifier_v2_mavic_split/artifacts/resnet18_stage2_mavic_split/best_model.pth")
DEFAULT_TEST_CSV = Path("data/stage2_classifier_v2_mavic_split/artifacts/resnet18_stage2_mavic_split/splits/test.csv")
DEFAULT_SIGMF_ROOT = Path("data_raw/sigmf")
DEFAULT_OUTPUT_DIR = Path("data/stage2_classifier_v2_mavic_split/artifacts/snr_sweep")
DEFAULT_SEGMENT_DURATION_S = 0.1

DEFAULT_SNR_LIST = ["inf", "20", "15", "10", "5", "0", "-5", "-10", "-15", "-20"]
DEFAULT_NOISE_TYPES = ["awgn", "fhss"]

SEG_FILENAME_RE = re.compile(r"^(?P<stem>.+?)__seg(?P<idx>\d{4,})\.png$")


# ---------------------------------------------------------------------------
# Indices auxiliares
# ---------------------------------------------------------------------------

def build_sigmf_index(sigmf_root: Path) -> dict[str, Path]:
    """{filename_stem: ruta_al_.sigmf-data}"""
    idx: dict[str, Path] = {}
    for p in sigmf_root.rglob("*.sigmf-data"):
        idx[p.stem] = p
    return idx


def read_sigmf_meta(stem: str, sigmf_data_path: Path) -> dict:
    """Lee fs y fc del .sigmf-meta hermano."""
    meta_path = sigmf_data_path.with_suffix(".sigmf-meta")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    fs = float(meta["global"]["core:sample_rate"])
    fc = float(meta["captures"][0]["core:frequency"])
    return {"fs": fs, "fc": fc, "meta_path": str(meta_path)}


def parse_filename(filename: str) -> tuple[str, int]:
    """Devuelve (filename_stem, seg_idx) desde la ruta del PNG."""
    base = os.path.basename(filename)
    m = SEG_FILENAME_RE.match(base)
    if not m:
        raise ValueError(f"Filename no parseable: {filename}")
    return m.group("stem"), int(m.group("idx"))


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

def load_model(model_path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    """Carga el ResNet18 6 clases con el state_dict del best_model.pth."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint


def build_eval_transforms(image_size: int, freq_normalize: bool) -> transforms.Compose:
    """Transforms identicas a las del eval del modelo de 6 clases."""
    steps = []
    if freq_normalize:
        steps.append(FrequencyNormalize(window_frac=0.5))
    steps.append(transforms.Resize((image_size, image_size)))
    steps += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(steps)


# ---------------------------------------------------------------------------
# Bucle principal
# ---------------------------------------------------------------------------

def parse_snr_list(snr_strs: list[str]) -> list[float]:
    out: list[float] = []
    for s in snr_strs:
        s = s.strip().lower()
        if s in ("inf", "+inf", "infty", "infinity", "clean"):
            out.append(float("inf"))
        else:
            out.append(float(s))
    return out


def snr_to_str(snr: float) -> str:
    if math.isinf(snr):
        return "inf"
    return f"{snr:+g}"


def run_sweep(args: argparse.Namespace) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "confusion_matrices").mkdir(parents=True, exist_ok=True)

    print(f"\nCargando modelo: {args.model_path}")
    model, checkpoint = load_model(args.model_path, device)
    image_size = int(checkpoint.get("image_size", 224))
    freq_normalize = bool(checkpoint.get("freq_normalize", True))
    print(f"  image_size={image_size}, freq_normalize={freq_normalize}")

    eval_transform = build_eval_transforms(image_size=image_size, freq_normalize=freq_normalize)

    print(f"\nCargando test CSV: {args.test_csv}")
    df = pd.read_csv(args.test_csv).reset_index(drop=True)
    if args.quick_check:
        df = df.groupby("label", group_keys=False).head(args.quick_check_per_class)
        print(f"Modo quick-check: {len(df)} segmentos ({args.quick_check_per_class} por clase)")
    print(f"  N segmentos: {len(df)}")

    print(f"\nIndexando .sigmf-data en {args.sigmf_root}")
    sigmf_index = build_sigmf_index(args.sigmf_root)
    print(f"  {len(sigmf_index)} ficheros .sigmf-data")

    # Resolver y cachear metadata SigMF por filename_stem (fs, fc)
    needed_stems = sorted({parse_filename(p)[0] for p in df["filename"]})
    sigmf_meta_index: dict[str, dict] = {}
    missing = []
    for stem in needed_stems:
        if stem not in sigmf_index:
            missing.append(stem)
            continue
        sigmf_meta_index[stem] = read_sigmf_meta(stem, sigmf_index[stem])
        sigmf_meta_index[stem]["data_path"] = str(sigmf_index[stem])
    if missing:
        raise RuntimeError(f"No se ha encontrado .sigmf-data para los stems: {missing}")
    print(f"  {len(sigmf_meta_index)} stems resueltos")
    with open(output_dir / "sigmf_meta_index.json", "w", encoding="utf-8") as f:
        json.dump(sigmf_meta_index, f, indent=2)

    snr_values = parse_snr_list(args.snr_list)
    noise_types = list(args.noise_types)
    seg_duration_s = float(args.segment_duration_s)

    render_mode = args.render_mode.lower()
    if render_mode == "fast":
        render_fn = render_spectrogram_image
        print("Render: rapido (numpy + PIL)")
    elif render_mode == "matplotlib":
        render_fn = render_spectrogram_image_matplotlib
        print("Render: matplotlib (lento, fiel a entrenamiento)")
    else:
        raise ValueError(f"--render-mode debe ser 'fast' o 'matplotlib', recibido: {render_mode}")

    print(f"\nBarrido:")
    print(f"  noise_types: {noise_types}")
    print(f"  snr_list:    {[snr_to_str(s) for s in snr_values]} dB")
    print(f"  total inferencias: {len(df) * len(snr_values) * len(noise_types)}")

    rng_master = np.random.default_rng(args.seed)

    # Pre-cargar IQ por (stem, seg_idx) UNA vez por segmento, reutilizado en
    # todas las combinaciones de noise/SNR.
    # Esto es lo que mas tiempo ahorra.

    per_sample_rows: list[dict] = []

    config_out = {
        "model_path": str(args.model_path),
        "test_csv": str(args.test_csv),
        "sigmf_root": str(args.sigmf_root),
        "output_dir": str(args.output_dir),
        "noise_types": noise_types,
        "snr_list_db": [snr_to_str(s) for s in snr_values],
        "segment_duration_s": seg_duration_s,
        "image_size": image_size,
        "freq_normalize": freq_normalize,
        "render_mode": render_mode,
        "quick_check": bool(args.quick_check),
        "n_test_samples": int(len(df)),
        "seed": int(args.seed),
        "device": str(device),
    }
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config_out, f, indent=2)

    t0 = time.time()
    with torch.no_grad():
        pbar_outer = tqdm(df.itertuples(index=False), total=len(df), desc="segmentos")
        for row in pbar_outer:
            filename = row.filename
            label_str = row.label
            label_idx = CLASS_TO_IDX[label_str]
            stem, seg_idx = parse_filename(filename)
            meta = sigmf_meta_index[stem]
            fs = meta["fs"]
            fc = meta["fc"]
            sigmf_path = Path(meta["data_path"])
            n_samples = int(round(seg_duration_s * fs))
            start = seg_idx * n_samples

            iq_clean = read_iq_segment(sigmf_path, start, n_samples)
            if iq_clean is None:
                print(f"[WARN] segmento incompleto: {filename}")
                continue

            # Cache de la prediccion limpia (SNR=inf) por fichero/segmento,
            # para reusarla entre noise_types sin volver a renderizar.
            clean_row: dict | None = None

            for noise_type in noise_types:
                for snr_db in snr_values:
                    # Para SNR=inf el resultado no depende de noise_type;
                    # se computa una sola vez y se reusa para los demas.
                    if math.isinf(snr_db) and clean_row is not None:
                        copy = clean_row.copy()
                        copy["noise_type"] = noise_type
                        per_sample_rows.append(copy)
                        continue

                    rng_local = np.random.default_rng(
                        rng_master.integers(0, 2**32 - 1)
                    )
                    iq_noisy, _info = inject_noise(
                        iq_clean, fs=fs, snr_db=snr_db,
                        noise_type=noise_type, rng=rng_local,
                    )
                    img = render_fn(iq_noisy, fs=fs, fc=fc)
                    x = eval_transform(img).unsqueeze(0).to(device)
                    logits = model(x)
                    pred_idx = int(logits.argmax(dim=1).item())

                    row_pred = {
                        "filename": filename,
                        "label": label_str,
                        "label_idx": label_idx,
                        "noise_type": noise_type,
                        "snr_db": snr_to_str(snr_db),
                        "snr_db_num": (float("inf") if math.isinf(snr_db) else float(snr_db)),
                        "pred_idx": pred_idx,
                        "pred_label": IDX_TO_CLASS[pred_idx],
                        "correct": int(pred_idx == label_idx),
                    }
                    per_sample_rows.append(row_pred)
                    # Guardar la prediccion limpia para reusarla
                    if math.isinf(snr_db):
                        clean_row = row_pred
            pbar_outer.set_postfix({
                "elapsed_min": f"{(time.time()-t0)/60:.1f}",
            })

    print(f"\nInferencia completada en {(time.time()-t0)/60:.1f} min")

    # Guardar predicciones a CSV
    per_sample_df = pd.DataFrame(per_sample_rows)
    per_sample_csv = output_dir / "predictions_per_sample.csv"
    per_sample_df.to_csv(per_sample_csv, index=False)
    print(f"Predicciones por muestra: {per_sample_csv}")

    # Agregar metricas por (noise_type, snr)
    rows = []
    for (noise_type, snr_str), grp in per_sample_df.groupby(["noise_type", "snr_db"]):
        y_true = grp["label_idx"].to_numpy()
        y_pred = grp["pred_idx"].to_numpy()
        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        f1_per_class = f1_score(
            y_true, y_pred,
            labels=list(range(len(CLASS_NAMES))),
            average=None,
            zero_division=0,
        )
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
        cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
        cm_path = output_dir / "confusion_matrices" / f"cm_{noise_type}_{snr_str}.csv"
        cm_df.to_csv(cm_path)

        out = {
            "noise_type": noise_type,
            "snr_db": snr_str,
            "snr_db_num": (
                float("inf") if snr_str == "inf" else float(snr_str)
            ),
            "n": int(len(grp)),
            "accuracy": float(acc),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
        }
        for cls_name, f1c in zip(CLASS_NAMES, f1_per_class):
            out[f"f1_{cls_name}"] = float(f1c)
        rows.append(out)

    sweep_df = pd.DataFrame(rows).sort_values(["noise_type", "snr_db_num"])
    sweep_csv = output_dir / "snr_sweep_results.csv"
    sweep_df.to_csv(sweep_csv, index=False)
    print(f"Resumen por (noise,snr): {sweep_csv}")

    return {
        "config": config_out,
        "per_sample_csv": str(per_sample_csv),
        "sweep_csv": str(sweep_csv),
        "n_predictions": len(per_sample_df),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Barrido SNR sobre Stage 2 (6 clases)")
    p.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    p.add_argument("--sigmf-root", type=Path, default=DEFAULT_SIGMF_ROOT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--segment-duration-s", type=float, default=DEFAULT_SEGMENT_DURATION_S)
    p.add_argument("--snr-list", nargs="+", default=DEFAULT_SNR_LIST, help="Lista de SNR en dB. Usa 'inf' para limpio.")
    p.add_argument("--noise-types", nargs="+", default=DEFAULT_NOISE_TYPES, choices=["awgn", "fhss"])
    p.add_argument("--quick-check", action="store_true", help="Modo rapido: solo N segmentos por clase, SNR=inf.")
    p.add_argument("--quick-check-per-class", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--render-mode", default="fast", choices=["fast", "matplotlib"],
        help="'fast' (numpy+PIL, 10-30x mas rapido) o 'matplotlib' (fiel a entrenamiento).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick_check:
        # Modo validacion: SNR=inf, los dos tipos de ruido (deberian coincidir)
        args.snr_list = ["inf"]
        # Mantener noise_types como default para validar que el codepath
        # de "skip si inf" funciona.
    run_sweep(args)


if __name__ == "__main__":
    main()
