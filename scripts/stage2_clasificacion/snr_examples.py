"""
snr_examples.py
===============

Genera rejillas de espectrogramas para *ver* visualmente como degrada cada
tipo de ruido (AWGN y FHSS-like) la senal a distintos SNR. Sirve para la
memoria del TFG y para sanity check de la inyeccion.

Para cada clase (drone Mini3, drone Mavic, drone F450, drone Hunter,
drone Mavic-novideo e interferencia), toma un segmento representativo del
test set y produce dos figuras:

    examples_<clase>_awgn.png    fila = SNR, columnas = limpio + niveles
    examples_<clase>_fhss.png

Cada figura es una rejilla 1 x N donde N = len(SNR_LIST) + 1 (la primera
columna es el espectrograma limpio).

Tambien produce una comparativa lado a lado AWGN vs FHSS para una clase
y un SNR concretos:
    examples_compare_<clase>_snr<N>.png

Uso tipico:
    uv run python scripts/snr_examples.py
    uv run python scripts/snr_examples.py --classes mini3 interference
    uv run python scripts/snr_examples.py --snr-list 10 0 -10
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from snr_noise import inject_noise  # noqa: E402
from spectrogram_render import (  # noqa: E402
    read_iq_segment,
    render_spectrogram_image,
)


SEG_FILENAME_RE = re.compile(r"^(?P<stem>.+?)__seg(?P<idx>\d{4,})\.png$")

DEFAULT_TEST_CSV = Path("data/stage2_classifier_v2_mavic_split/artifacts/resnet18_stage2_mavic_split/splits/test.csv")
DEFAULT_SIGMF_ROOT = Path("data_raw/sigmf")
DEFAULT_OUTPUT_DIR = Path("data/stage2_classifier_v2_mavic_split/artifacts/snr_sweep/examples")
DEFAULT_CLASSES = ["mini3", "interference", "mavic_video", "mavic_novideo", "hunter", "f450"]
DEFAULT_SNR_LIST = ["20", "10", "0", "-10", "-20"]


def parse_filename(filename: str) -> tuple[str, int]:
    base = os.path.basename(filename)
    m = SEG_FILENAME_RE.match(base)
    if not m:
        raise ValueError(f"Filename no parseable: {filename}")
    return m.group("stem"), int(m.group("idx"))


def build_sigmf_index(sigmf_root: Path) -> dict[str, Path]:
    return {p.stem: p for p in sigmf_root.rglob("*.sigmf-data")}


def read_sigmf_meta(sigmf_data_path: Path) -> tuple[float, float]:
    meta_path = sigmf_data_path.with_suffix(".sigmf-meta")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    fs = float(meta["global"]["core:sample_rate"])
    fc = float(meta["captures"][0]["core:frequency"])
    return fs, fc


def pick_sample_per_class(
    df: pd.DataFrame,
    sigmf_index: dict[str, Path],
    classes: list[str],
    seg_pref: int = 25,
) -> dict[str, dict]:
    """
    Selecciona un segmento por clase. Prefiere uno con indice cercano a
    seg_pref (evita los primeros donde a veces hay arranque del SDR).
    """
    out: dict[str, dict] = {}
    for cls in classes:
        sub = df[df["label"] == cls]
        if sub.empty:
            print(f"[WARN] sin muestras de clase {cls}")
            continue
        # Para cada filename del subset, calcular idx
        candidates = []
        for fn in sub["filename"].tolist():
            stem, idx = parse_filename(fn)
            if stem in sigmf_index:
                candidates.append((stem, idx, fn))
        if not candidates:
            print(f"[WARN] clase {cls} sin candidatos resueltos")
            continue
        # Ordenar por distancia a seg_pref
        candidates.sort(key=lambda x: abs(x[1] - seg_pref))
        stem, idx, fn = candidates[0]
        sigmf_path = sigmf_index[stem]
        fs, fc = read_sigmf_meta(sigmf_path)
        out[cls] = {
            "filename": fn,
            "stem": stem,
            "seg_idx": idx,
            "sigmf_path": str(sigmf_path),
            "fs": fs,
            "fc": fc,
        }
    return out


def render_with_noise(
    iq_clean: np.ndarray,
    fs: float,
    fc: float,
    snr_db: float,
    noise_type: str,
    rng: np.random.Generator,
) -> Image.Image:
    iq, _ = inject_noise(iq_clean, fs=fs, snr_db=snr_db, noise_type=noise_type, rng=rng)
    return render_spectrogram_image(iq, fs=fs, fc=fc)


def snr_label(snr: float) -> str:
    return "limpio" if math.isinf(snr) else f"SNR={snr:+g} dB"


def plot_grid_for_class(
    cls: str,
    info: dict,
    snr_list: list[float],
    noise_types: list[str],
    output_dir: Path,
    rng: np.random.Generator,
    seg_duration_s: float = 0.1,
) -> None:
    sigmf_path = Path(info["sigmf_path"])
    fs = info["fs"]; fc = info["fc"]
    n_samples = int(round(seg_duration_s * fs))
    start = info["seg_idx"] * n_samples
    iq_clean = read_iq_segment(sigmf_path, start, n_samples)
    if iq_clean is None:
        print(f"[WARN] segmento incompleto para {cls}")
        return

    # Por cada noise_type, hacer figura con [limpio, SNR1, SNR2, ...]
    for noise_type in noise_types:
        cols = [float("inf")] + snr_list
        n_cols = len(cols)
        fig, axes = plt.subplots(1, n_cols, figsize=(3.2 * n_cols, 3.0))
        if n_cols == 1:
            axes = [axes]

        for ax, snr in zip(axes, cols):
            img = render_with_noise(iq_clean, fs, fc, snr, noise_type, rng)
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(snr_label(snr), fontsize=10)

        fig.suptitle(f"Clase: {cls}   |   Ruido: {noise_type.upper()}", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        out = output_dir / f"examples_{cls}_{noise_type}.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"  guardado {out}")


def plot_comparison(
    cls: str,
    info: dict,
    snr: float,
    output_dir: Path,
    rng: np.random.Generator,
    seg_duration_s: float = 0.1,
) -> None:
    """3 paneles: limpio, AWGN al SNR dado, FHSS al SNR dado."""
    sigmf_path = Path(info["sigmf_path"])
    fs = info["fs"]; fc = info["fc"]
    n_samples = int(round(seg_duration_s * fs))
    start = info["seg_idx"] * n_samples
    iq_clean = read_iq_segment(sigmf_path, start, n_samples)
    if iq_clean is None:
        return

    img_clean = render_spectrogram_image(iq_clean, fs, fc)
    img_awgn = render_with_noise(iq_clean, fs, fc, snr, "awgn", rng)
    img_fhss = render_with_noise(iq_clean, fs, fc, snr, "fhss", rng)

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0))
    for ax, img, title in zip(
        axes,
        [img_clean, img_awgn, img_fhss],
        ["Limpio", f"AWGN @ SNR={snr:+g} dB", f"FHSS-like @ SNR={snr:+g} dB"],
    ):
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=10)
    fig.suptitle(f"Comparativa AWGN vs FHSS - clase {cls}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    snr_str = f"{snr:+g}".replace("+", "p").replace("-", "m")
    out = output_dir / f"examples_compare_{cls}_snr{snr_str}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  guardado {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Visualizar como degrada cada ruido la senal")
    p.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    p.add_argument("--sigmf-root", type=Path, default=DEFAULT_SIGMF_ROOT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    p.add_argument("--snr-list", nargs="+", default=DEFAULT_SNR_LIST,
                   help="Lista de SNR en dB para las rejillas (el limpio se anade siempre).")
    p.add_argument("--noise-types", nargs="+", default=["awgn", "fhss"], choices=["awgn", "fhss"])
    p.add_argument("--compare-snr", type=float, default=0.0,
                   help="SNR usado para la figura comparativa AWGN vs FHSS por clase.")
    p.add_argument("--seg-pref", type=int, default=25, help="Indice de segmento preferido por clase.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Salida: {args.output_dir}")

    snr_list = [float(s) for s in args.snr_list]
    print(f"SNRs: {snr_list}")

    df = pd.read_csv(args.test_csv)
    sigmf_index = build_sigmf_index(args.sigmf_root)
    print(f"SigMF en disco: {len(sigmf_index)}")

    samples = pick_sample_per_class(df, sigmf_index, args.classes, seg_pref=args.seg_pref)
    print("\nMuestras seleccionadas por clase:")
    for cls, info in samples.items():
        print(f"  {cls:<15s} -> {info['stem']} seg{info['seg_idx']:04d} (fs={info['fs']/1e6:.1f} MHz)")

    rng = np.random.default_rng(args.seed)

    print("\nGenerando rejillas:")
    for cls, info in samples.items():
        plot_grid_for_class(
            cls, info, snr_list, args.noise_types,
            args.output_dir, rng,
        )

    print(f"\nGenerando comparativas AWGN vs FHSS a SNR={args.compare_snr}:")
    for cls, info in samples.items():
        plot_comparison(cls, info, args.compare_snr, args.output_dir, rng)

    print(f"\nListo. Outputs en: {args.output_dir}")


if __name__ == "__main__":
    main()
