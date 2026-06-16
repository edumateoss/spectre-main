from pathlib import Path
import numpy as np


def read_sigmf_cf32(path, max_samples=None):
    """
    Lee un .sigmf-data en formato cf32_le:
    float32 intercalado I,Q,I,Q...
    """
    raw = np.fromfile(path, dtype=np.float32)

    if len(raw) % 2 != 0:
        raw = raw[:-1]

    iq = raw[0::2] + 1j * raw[1::2]

    if max_samples is not None:
        iq = iq[:max_samples]

    return iq.astype(np.complex64)


def write_sigmf_cf32(path, iq):
    """
    Guarda IQ complejo como cf32_le:
    float32 intercalado I,Q,I,Q...
    """
    out = np.empty(iq.size * 2, dtype=np.float32)
    out[0::2] = np.real(iq)
    out[1::2] = np.imag(iq)
    out.tofile(path)


def add_awgn_complex(iq, target_snr_db, seed=1234):
    """
    Añade AWGN complejo para obtener un SNR objetivo aproximado.
    """
    rng = np.random.default_rng(seed)

    signal_power = np.mean(np.abs(iq) ** 2)
    snr_linear = 10 ** (target_snr_db / 10)
    noise_power = signal_power / snr_linear

    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(iq.shape) + 1j * rng.standard_normal(iq.shape)
    )

    return (iq + noise).astype(np.complex64)


def main():
    input_file = Path("Archives/dron_USRP.sigmf-data")
    output_dir = Path("Archives/dron_snr_versions")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Para probar rápido usa 2 o 5 millones de muestras.
    # Luego puedes poner max_samples=None para procesar todo.
    iq = read_sigmf_cf32(input_file, max_samples=5_000_000)

    snr_levels = [20, 10, 0, -5, -10, -15, -20]

    for snr in snr_levels:
        iq_noisy = add_awgn_complex(iq, target_snr_db=snr, seed=1234)
        out_file = output_dir / f"dron_{snr}dB.sigmf-data"
        write_sigmf_cf32(out_file, iq_noisy)
        print(f"Guardado: {out_file}")


if __name__ == "__main__":
    main()