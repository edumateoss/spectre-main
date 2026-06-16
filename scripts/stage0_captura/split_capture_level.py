"""
Split del dataset por capture_id (no por PNG) para evitar fuga de informacion
entre train / val / test.

Genera unicamente CSVs de metadatos en data/mini3_detector_python_v1/metadata/:
    - train.csv, val.csv, test.csv          (una fila por PNG)
    - train_capture_ids.csv, val_..., test_...   (una fila por captura)

No copia los PNG a carpetas train/val/test. Los DataLoaders leen los PNG
desde su ubicacion original en "all/<label>/..." usando el "filename" del CSV.

Uso:
    python scripts/split_capture_level.py \
        --metadata-csv data/mini3_detector_python_v1/metadata/spectrograms.csv \
        --output-dir   data/mini3_detector_python_v1/metadata \
        --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15 \
        --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--metadata-csv",
        default="data/mini3_detector_python_v1/metadata/spectrograms.csv",
        help="CSV maestro generado por make_spectrogram_dataset.py",
    )
    p.add_argument(
        "--output-dir",
        default="data/mini3_detector_python_v1/metadata",
        help="Carpeta donde se escriben train.csv, val.csv, test.csv y los *_capture_ids.csv",
    )
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--stratify-by",
        default="label",
        choices=["label", "label_antenna", "label_distance"],
        help=(
            "Variable de estratificacion a nivel de captura. "
            "Por defecto 'label' (mas estable con pocas capturas)."
        ),
    )
    return p.parse_args()


def build_strata(capture_df: pd.DataFrame, stratify_by: str) -> pd.Series | None:
    """
    Devuelve una Series con la columna de estratificacion, o None si no se puede
    estratificar (alguna clase aparece menos de 2 veces).
    """
    if stratify_by == "label":
        strata = capture_df["label"].astype(str)
    elif stratify_by == "label_antenna":
        strata = (
            capture_df["label"].astype(str)
            + "__"
            + capture_df["antenna"].astype(str)
        )
    elif stratify_by == "label_distance":
        strata = (
            capture_df["label"].astype(str)
            + "__"
            + capture_df["distance"].astype(str)
        )
    else:
        raise ValueError(f"stratify_by no soportado: {stratify_by}")

    counts = strata.value_counts()
    if counts.min() < 2:
        print(
            f"[WARN] No se puede estratificar por '{stratify_by}' porque "
            f"alguna clase tiene <2 capturas. Cayendo a estratificacion por label."
        )
        if stratify_by != "label":
            return build_strata(capture_df, "label")
        return None
    return strata


def safe_split(
    df: pd.DataFrame,
    test_size: float,
    seed: int,
    strata: pd.Series | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """train_test_split tolerante a fallos de estratificacion."""
    try:
        return train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            stratify=strata,
        )
    except ValueError as e:
        print(f"[WARN] Estratificacion fallida ({e}). Reintentando sin estratificar.")
        return train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            stratify=None,
        )


def report_split(name: str, df_pngs: pd.DataFrame) -> None:
    n_caps = df_pngs["capture_id"].nunique()
    n_png = len(df_pngs)
    counts = df_pngs["label"].value_counts().to_dict()
    print(
        f"  {name:5s}: {n_caps:3d} capturas | {n_png:5d} PNGs | "
        f"drone={counts.get('drone', 0)}  interference={counts.get('interference', 0)}"
    )


def main() -> None:
    args = parse_args()

    total = round(args.train_ratio + args.val_ratio + args.test_ratio, 6)
    if total != 1.0:
        raise ValueError(
            f"train + val + test debe sumar 1.0 (actual: {total})"
        )

    metadata_csv = Path(args.metadata_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not metadata_csv.exists():
        raise FileNotFoundError(f"No existe {metadata_csv}")

    df = pd.read_csv(metadata_csv)

    required_cols = {"filename", "label", "capture_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en {metadata_csv}: {sorted(missing)}")

    print(f"\nCSV maestro: {metadata_csv}")
    print(f"  Filas (PNGs): {len(df)}")
    print(f"  Capturas unicas: {df['capture_id'].nunique()}")
    print(f"  Distribucion por label:\n{df['label'].value_counts().to_string()}")

    # ------------------------------------------------------------
    # 1. Tabla a nivel de captura (una fila por captura)
    # ------------------------------------------------------------
    group_cols = ["capture_id", "label"]
    optional_cols = ["antenna", "distance", "angle", "environment", "gain", "platform"]
    for c in optional_cols:
        if c in df.columns:
            group_cols.append(c)

    capture_df = (
        df.groupby("capture_id", as_index=False)
        .first()[group_cols]
        .reset_index(drop=True)
    )
    print(f"\nCapturas para repartir: {len(capture_df)}")

    # ------------------------------------------------------------
    # 2. Split jerarquico: primero train vs (val+test), luego val vs test
    # ------------------------------------------------------------
    strata = build_strata(capture_df, args.stratify_by)

    train_caps, temp_caps = safe_split(
        capture_df,
        test_size=(1.0 - args.train_ratio),
        seed=args.seed,
        strata=strata,
    )

    temp_ratio = args.val_ratio + args.test_ratio
    val_share = args.val_ratio / temp_ratio  # cuanto del bloque temporal va a val
    temp_strata = build_strata(temp_caps.reset_index(drop=True), args.stratify_by)

    val_caps, test_caps = safe_split(
        temp_caps,
        test_size=(1.0 - val_share),
        seed=args.seed,
        strata=temp_strata,
    )

    splits = {
        "train": train_caps,
        "val": val_caps,
        "test": test_caps,
    }

    # ------------------------------------------------------------
    # 3. Sanity check: ningun capture_id en mas de un split
    # ------------------------------------------------------------
    train_ids = set(train_caps["capture_id"])
    val_ids = set(val_caps["capture_id"])
    test_ids = set(test_caps["capture_id"])

    overlap_tv = train_ids & val_ids
    overlap_tt = train_ids & test_ids
    overlap_vt = val_ids & test_ids
    if overlap_tv or overlap_tt or overlap_vt:
        raise RuntimeError(
            f"Leakage detectado: train/val={overlap_tv}, "
            f"train/test={overlap_tt}, val/test={overlap_vt}"
        )

    # ------------------------------------------------------------
    # 4. Volcar a CSV: capture_ids + PNGs por split
    # ------------------------------------------------------------
    print("\nResultados del split:")
    for name, caps in splits.items():
        caps_sorted = caps.sort_values("capture_id").reset_index(drop=True)
        caps_csv = output_dir / f"{name}_capture_ids.csv"
        caps_sorted.to_csv(caps_csv, index=False)

        df_split = df[df["capture_id"].isin(caps["capture_id"])].copy()
        df_split = df_split.sort_values(["capture_id", "filename"]).reset_index(drop=True)
        png_csv = output_dir / f"{name}.csv"
        df_split.to_csv(png_csv, index=False)

        report_split(name, df_split)

    print(f"\nCSVs generados en: {output_dir}")
    print("Listo. El siguiente paso es entrenar el detector binario.")


if __name__ == "__main__":
    main()
