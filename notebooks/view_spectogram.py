from pathlib import Path
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt
from sigmf import sigmffile


def resolve_sigmf_paths(input_path: str):
    path = Path(input_path)

    if path.suffix == ".sigmf-meta":
        meta_path = path
        data_path = Path(str(path).replace(".sigmf-meta", ".sigmf-data"))
    elif path.suffix == ".sigmf-data":
        data_path = path
        meta_path = Path(str(path).replace(".sigmf-data", ".sigmf-meta"))
    else:
        meta_path = Path(str(path) + ".sigmf-meta")
        data_path = Path(str(path) + ".sigmf-data")

    if not meta_path.exists():
        raise FileNotFoundError(f"No encuentro el archivo meta: {meta_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"No encuentro el archivo data: {data_path}")

    return meta_path, data_path


def load_sigmf_capture(input_path: str):
    meta_path, data_path = resolve_sigmf_paths(input_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    global_meta = meta.get("global", {})
    captures = meta.get("captures", [])

    sample_rate = global_meta.get("core:sample_rate", None)
    datatype = global_meta.get("core:datatype", None)

    center_freq = None
    if captures:
        center_freq = captures[0].get("core:frequency", None)

    sigfile = sigmffile.fromfile(str(meta_path))
    samples = sigfile.read_samples()

    return samples, sample_rate, center_freq, datatype, meta_path, data_path


def plot_spectrogram(
    samples: np.ndarray,
    sample_rate: float,
    center_freq: float | None,
    output_path: str,
    nfft: int = 1024,
    noverlap: int = 768,
    max_seconds: float | None = None,
    db_min: float | None = None,
    db_max: float | None = None,
):
    if sample_rate is None:
        raise ValueError("No se ha encontrado core:sample_rate en el archivo SigMF meta.")

    if max_seconds is not None:
        max_samples = int(max_seconds * sample_rate)
        samples = samples[:max_samples]

    if len(samples) == 0:
        raise ValueError("La señal está vacía.")

    duration_s = len(samples) / sample_rate

    fig, ax = plt.subplots(figsize=(12, 6))

    pxx, freqs, bins, im = ax.specgram(
        samples,
        NFFT=nfft,
        Fs=sample_rate,
        noverlap=noverlap,
        scale="dB",
        mode="psd",
    )

    if db_min is not None or db_max is not None:
        im.set_clim(vmin=db_min, vmax=db_max)

    ax.set_xlabel("Tiempo (s)")
    if center_freq is not None:
        ax.set_ylabel(f"Frecuencia relativa (Hz)\nfc = {center_freq / 1e6:.3f} MHz")
        ax.set_title(f"Espectrograma SigMF | {duration_s:.3f} s | fc = {center_freq / 1e6:.3f} MHz")
    else:
        ax.set_ylabel("Frecuencia relativa (Hz)")
        ax.set_title(f"Espectrograma SigMF | {duration_s:.3f} s")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Potencia/Frecuencia (dB)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"\nEspectrograma guardado en: {output_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Generar espectrograma a partir de una captura SigMF")
    parser.add_argument(
        "input_path",
        type=str,
        help="Ruta al nombre base o al .sigmf-meta/.sigmf-data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="PNG de salida. Si no se indica, se guarda junto al archivo de entrada.",
    )
    parser.add_argument(
        "--nfft",
        type=int,
        default=1024,
        help="Tamaño FFT",
    )
    parser.add_argument(
        "--noverlap",
        type=int,
        default=768,
        help="Solape entre ventanas",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=1.0,
        help="Tiempo máximo a visualizar",
    )
    parser.add_argument(
        "--db-min",
        type=float,
        default=None,
        help="Límite inferior en dB",
    )
    parser.add_argument(
        "--db-max",
        type=float,
        default=None,
        help="Límite superior en dB",
    )

    args = parser.parse_args()

    samples, sample_rate, center_freq, datatype, meta_path, data_path = load_sigmf_capture(args.input_path)

    if args.output is None:
        base_name = meta_path.name.replace(".sigmf-meta", "")
        output_path = str(meta_path.parent / f"{base_name}_spectrogram.png")
    else:
        output_path = args.output

    print("\n=== Información de la captura ===")
    print(f"Meta file   : {meta_path}")
    print(f"Data file   : {data_path}")
    print(f"Datatype    : {datatype}")
    print(f"Sample rate : {sample_rate} Sa/s")
    if center_freq is not None:
        print(f"Center freq : {center_freq} Hz ({center_freq / 1e6:.3f} MHz)")
    print(f"Samples     : {len(samples)}")
    if sample_rate is not None:
        print(f"Duration    : {len(samples) / sample_rate:.6f} s")

    plot_spectrogram(
        samples=samples,
        sample_rate=sample_rate,
        center_freq=center_freq,
        output_path=output_path,
        nfft=args.nfft,
        noverlap=args.noverlap,
        max_seconds=args.max_seconds,
        db_min=args.db_min,
        db_max=args.db_max,
    )


if __name__ == "__main__":
    main()