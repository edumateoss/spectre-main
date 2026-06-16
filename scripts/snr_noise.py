"""
snr_noise.py
============

Modulo de inyeccion de ruido sintetico sobre senal IQ compleja, calibrado
en SNR objetivo (dB). Pensado para el barrido de robustez del Stage 2.

Provee dos modelos de ruido:

1. AWGN: ruido blanco gaussiano complejo, modelo canonico del paper RFUAV.
   La potencia se ajusta para cumplir un SNR objetivo respecto a la potencia
   media del IQ original.

2. FHSS-like (interferencia tipo Bluetooth): bursts cortos de tonos
   modulados en frecuencias aleatorias dentro del ancho de banda capturado.
   Simula una interferencia FHSS que comparte morfologia con la senal del
   dron, lo que hace el problema de clasificacion mas dificil que con AWGN.
   Parametros tipicos de Bluetooth Classic:
       - 1600 hops/s
       - BW por hop = 1 MHz
       - duracion de slot = 366 us

Convenciones:
    - IQ complejo de tipo numpy.complex64.
    - SNR en dB. Se acepta snr_db = numpy.inf para no inyectar ruido.
    - La potencia "de senal" es la potencia media del IQ original
      (np.mean(|x|**2)) tras quitar el componente DC.
"""
from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
# Utilidades generales
# ---------------------------------------------------------------------------

def signal_power(iq: np.ndarray) -> float:
    """Potencia media en escala lineal de una senal IQ compleja."""
    if iq.size == 0:
        return 0.0
    return float(np.mean(np.abs(iq) ** 2))


def measured_snr_db(iq_clean: np.ndarray, noise: np.ndarray) -> float:
    """SNR medido en dB entre senal limpia y vector de ruido inyectado."""
    p_s = signal_power(iq_clean)
    p_n = signal_power(noise)
    if p_n <= 0:
        return float("inf")
    return 10.0 * math.log10(p_s / p_n)


def _target_noise_power(p_signal: float, snr_db: float) -> float:
    """Potencia de ruido necesaria para alcanzar el SNR objetivo en dB."""
    if not math.isfinite(snr_db):
        return 0.0
    return p_signal / (10.0 ** (snr_db / 10.0))


# ---------------------------------------------------------------------------
# 1. Ruido AWGN complejo
# ---------------------------------------------------------------------------

def add_awgn(
    iq: np.ndarray,
    snr_db: float,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Anade ruido blanco gaussiano complejo a una senal IQ.

    n[k] ~ CN(0, sigma**2) con sigma**2 calibrada para cumplir snr_db.

    Devuelve:
        iq_noisy: senal contaminada (complex64)
        info:     diccionario con potencias y SNR medido para diagnostico.
    """
    if rng is None:
        rng = np.random.default_rng()

    iq = np.asarray(iq, dtype=np.complex64)
    p_signal = signal_power(iq)

    if not math.isfinite(snr_db):
        # SNR infinito: no inyectamos nada
        return iq.copy(), {
            "p_signal": p_signal,
            "p_noise": 0.0,
            "snr_db_target": float("inf"),
            "snr_db_measured": float("inf"),
            "noise_type": "awgn",
        }

    p_noise = _target_noise_power(p_signal, snr_db)
    # Ruido complejo con varianza total p_noise y varianza por componente p_noise/2
    sigma = math.sqrt(p_noise / 2.0)
    n_re = rng.standard_normal(iq.shape).astype(np.float32) * sigma
    n_im = rng.standard_normal(iq.shape).astype(np.float32) * sigma
    noise = (n_re + 1j * n_im).astype(np.complex64)

    iq_noisy = iq + noise

    info = {
        "p_signal": p_signal,
        "p_noise": signal_power(noise),
        "snr_db_target": float(snr_db),
        "snr_db_measured": measured_snr_db(iq, noise),
        "noise_type": "awgn",
    }
    return iq_noisy, info


# ---------------------------------------------------------------------------
# 2. Ruido tipo FHSS / Bluetooth sintetico
# ---------------------------------------------------------------------------

def add_fhss_interference(
    iq: np.ndarray,
    fs: float,
    snr_db: float,
    rng: np.random.Generator | None = None,
    hop_rate: float = 1600.0,
    burst_duration_s: float = 366e-6,
    hop_bw_hz: float = 1e6,
    band_frac: float = 0.9,
) -> tuple[np.ndarray, dict]:
    """
    Inyecta interferencia FHSS sintetica al estilo de Bluetooth Classic.

    Modelo:
      - Se generan ceil(hop_rate * duracion) bursts.
      - Cada burst tiene duracion burst_duration_s, frecuencia central
        elegida uniformemente en [-band_frac/2, +band_frac/2] * fs,
        y portadora compleja exp(j 2pi f_c t) sin modulacion adicional.
        Esto produce, en el espectrograma, un rectangulo estrecho
        (~hop_bw_hz por la duracion del burst) en una posicion aleatoria
        del plano tiempo-frecuencia.
      - La energia total del proceso se calibra a la potencia objetivo
        para cumplir snr_db.

    Parametros:
        fs: sample rate Hz.
        snr_db: SNR objetivo respecto a la potencia del IQ original.
        hop_rate: hops/s (BT Classic ~ 1600).
        burst_duration_s: duracion de cada burst (~366 us en BT).
        hop_bw_hz: anchura de cada burst en frecuencia (informativa,
            se aproxima por el sinc del pulso rectangular de duracion
            burst_duration_s ~ 1/burst_duration_s Hz; con 366us sale ~2.7 kHz,
            pero al ser senal modulada el lobulo efectivo ronda ~1 MHz).
        band_frac: fraccion del ancho de banda capturado donde pueden caer
            las portadoras (0.9 evita pegar bursts contra los bordes del
            espectro).

    Devuelve:
        iq_noisy: senal con interferencia inyectada (complex64).
        info: diccionario diagnostico.
    """
    if rng is None:
        rng = np.random.default_rng()

    iq = np.asarray(iq, dtype=np.complex64)
    N = iq.shape[0]
    p_signal = signal_power(iq)

    if not math.isfinite(snr_db):
        return iq.copy(), {
            "p_signal": p_signal,
            "p_noise": 0.0,
            "snr_db_target": float("inf"),
            "snr_db_measured": float("inf"),
            "noise_type": "fhss",
            "n_bursts": 0,
        }

    duration_s = N / fs
    n_bursts = max(1, int(math.ceil(hop_rate * duration_s)))
    burst_samples = max(1, int(round(burst_duration_s * fs)))

    interference = np.zeros(N, dtype=np.complex64)

    # Posiciones de inicio aleatorias. Permitimos que se solapen,
    # como ocurre en un entorno real con varios emisores BT activos.
    starts = rng.integers(0, max(1, N - burst_samples + 1), size=n_bursts)
    # Frecuencias centrales aleatorias dentro de la banda
    half_bw = (band_frac / 2.0) * fs
    f_centers = rng.uniform(-half_bw, half_bw, size=n_bursts)

    # Fases iniciales aleatorias para que no sumen coherentemente
    phases = rng.uniform(0.0, 2.0 * math.pi, size=n_bursts)

    # Amplitud unitaria por burst antes de calibrar potencia global
    t_burst = np.arange(burst_samples, dtype=np.float32) / fs

    for k in range(n_bursts):
        s = int(starts[k])
        e = s + burst_samples
        if e > N:
            e = N
            length = e - s
            tk = t_burst[:length]
        else:
            length = burst_samples
            tk = t_burst

        # Anadir un pequeno chirp para emular GFSK simplificado: f_c +/- 250 kHz
        # con sentido aleatorio. Esto evita bursts perfectamente monotonales.
        f_dev = float(rng.choice([-250e3, 250e3]))
        # Frecuencia instantanea lineal del symbol: f(t) = f_c + f_dev * (2*t/T - 1)
        f_inst = f_centers[k] + f_dev * (2.0 * tk / burst_duration_s - 1.0)
        phase_inst = phases[k] + 2.0 * math.pi * np.cumsum(f_inst) / fs

        # Ventana de Hann sobre la duracion del burst para evitar saltos
        # bruscos en los bordes (suaviza el espectro).
        w = np.hanning(length).astype(np.float32) if length > 1 else np.array([1.0], dtype=np.float32)

        burst = (w * np.exp(1j * phase_inst).astype(np.complex64))
        interference[s:e] += burst

    # Calibrar potencia total al SNR objetivo
    p_inter_raw = signal_power(interference)
    if p_inter_raw <= 0:
        return iq.copy(), {
            "p_signal": p_signal,
            "p_noise": 0.0,
            "snr_db_target": float(snr_db),
            "snr_db_measured": float("inf"),
            "noise_type": "fhss",
            "n_bursts": n_bursts,
        }
    p_target = _target_noise_power(p_signal, snr_db)
    scale = math.sqrt(p_target / p_inter_raw)
    interference = (interference * scale).astype(np.complex64)

    iq_noisy = iq + interference

    info = {
        "p_signal": p_signal,
        "p_noise": signal_power(interference),
        "snr_db_target": float(snr_db),
        "snr_db_measured": measured_snr_db(iq, interference),
        "noise_type": "fhss",
        "n_bursts": n_bursts,
        "burst_samples": burst_samples,
    }
    return iq_noisy, info


# ---------------------------------------------------------------------------
# 3. API unificada
# ---------------------------------------------------------------------------

def inject_noise(
    iq: np.ndarray,
    fs: float,
    snr_db: float,
    noise_type: str,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    """Punto de entrada unico. noise_type ∈ {"awgn", "fhss"}."""
    noise_type = noise_type.lower()
    if noise_type == "awgn":
        return add_awgn(iq, snr_db=snr_db, rng=rng)
    if noise_type in ("fhss", "bluetooth", "bt"):
        return add_fhss_interference(iq, fs=fs, snr_db=snr_db, rng=rng)
    raise ValueError(f"noise_type no reconocido: {noise_type!r}")


# ---------------------------------------------------------------------------
# Auto-test minimo (calibracion de SNR)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    fs = 100e6  # 100 MS/s, similar al sample rate de las capturas reales
    N = int(0.1 * fs)
    # Senal de prueba: tono complejo con potencia unitaria
    t = np.arange(N) / fs
    x = (np.exp(1j * 2 * np.pi * 1e6 * t)).astype(np.complex64)

    print("Test de calibracion AWGN:")
    for snr in [20, 10, 0, -10, -20]:
        _, info = add_awgn(x, snr_db=snr, rng=rng)
        print(f"  target={snr:+4d} dB | medido={info['snr_db_measured']:+.2f} dB")

    print("\nTest de calibracion FHSS-like:")
    for snr in [20, 10, 0, -10, -20]:
        _, info = add_fhss_interference(x, fs=fs, snr_db=snr, rng=rng)
        print(
            f"  target={snr:+4d} dB | medido={info['snr_db_measured']:+.2f} dB "
            f"| n_bursts={info['n_bursts']}"
        )
