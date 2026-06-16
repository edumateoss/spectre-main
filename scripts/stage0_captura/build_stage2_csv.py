"""
build_stage2_csv.py
===================

Reconstruye el CSV maestro del Stage 2 a partir del CSV existente y de
los PNGs reales en disco. Hace cuatro cosas:

1. **Deduplica filas y verifica PNG en disco.** Auditoria 2026-05-26
   detecta 19 PNGs de hunter_4_conwifi (segs 0000-0018) que aparecen
   DUPLICADOS en el CSV (cada uno dos veces). Se conserva la primera
   ocurrencia y se descarta la duplicada. Adicionalmente, cualquier
   fila cuyo PNG no exista en disco se elimina.

2. **Anyade clase mini3 reusando el dataset v3.** 1800 PNGs en 12
   capturas (5 m / 10 m, WiFi L1/L2/L3, 2 caps por condicion).

3. **Anyade clase interference reusando el dataset v3.** Para evitar
   duplicar PNGs, las filas de interference apuntan a rutas relativas
   a data/mini3_detector_python_v3/all/interference/ via la columna
   `source_dataset`.

4. **Anyade subgroup video/novideo a mavic** para evitar fugas de
   informacion en el split. Mavic_6 se parte en dos capture_ids
   sinteticos:
      - mavic_6_novideo : segs 0-70 (sin transmision de video activa)
      - mavic_6_video   : segs 71-113 (transmision de video activa)
   El resto de capturas de mavic se etiquetan:
      - mavic_4, mavic_5 : novideo
      - mavic_10*        : video

Output:

- spectrograms_stage2_clean.csv (CSV definitivo enriquecido)

El CSV original no se sobrescribe; se conserva como backup.

Uso:
    uv run python scripts/build_stage2_csv.py
    uv run python scripts/build_stage2_csv.py --no-mini3
    uv run python scripts/build_stage2_csv.py --no-interference
    uv run python scripts/build_stage2_csv.py --interference-cap-limit 3
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_CSV_IN = Path("data/stage2_classifier_v1/metadata/spectrograms_stage2.csv")
DEFAULT_CSV_OUT = Path("data/stage2_classifier_v1/metadata/spectrograms_stage2_clean.csv")

V3_DRONE = Path("data/mini3_detector_python_v3/all/drone")
V3_INTERFERENCE = Path("data/mini3_detector_python_v3/all/interference")

HEADER = [
    "filename",
    "label",
    "subgroup",
    "wifi_level",
    "condition",
    "antenna",
    "distance_m",
    "angle",
    "environment",
    "fc",
    "fs",
    "gain",
    "capture_id",
    "platform",
    "source_dataset",
]

MAVIC_VIDEO_CAPTURES = {
    "mavic_10",
    "mavic_10_+10",
    "mavic_10_+10_1",
    "mavic_10_-10_1",
    "mavic_10_-10_2",
    "mavic_10_captura1",
}
MAVIC_NOVIDEO_CAPTURES = {"mavic_4", "mavic_5"}
MAVIC_6_TRANSITION_SEG = 71


def classify_mavic_row(filename: str, capture_id: str) -> tuple[str, str]:
    """Devuelve (subgroup, capture_id_final) para una fila de mavic."""
    if capture_id == "mavic_6":
        try:
            seg_str = filename.rsplit("__seg", 1)[1].rsplit(".", 1)[0]
            seg = int(seg_str)
        except (IndexError, ValueError):
            return "novideo", "mavic_6"
        if seg < MAVIC_6_TRANSITION_SEG:
            return "novideo", "mavic_6_novideo"
        return "video", "mavic_6_video"
    if capture_id in MAVIC_VIDEO_CAPTURES:
        return "video", capture_id
    if capture_id in MAVIC_NOVIDEO_CAPTURES:
        return "novideo", capture_id
    return "novideo", capture_id


def parse_mini3_capture(cap: str) -> tuple[str, str]:
    """De mini3_drone_d{D}_w{W}_s{S}_cap{NN} extrae distance y wifi level."""
    dist = wifi = ""
    for p in cap.split("_"):
        if p.startswith("d") and p[1:].isdigit():
            dist = p[1:]
        elif p.startswith("w") and p[1:].isdigit():
            wifi = p[1:]
    return dist, wifi


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv-in", type=Path, default=DEFAULT_CSV_IN)
    p.add_argument("--csv-out", type=Path, default=DEFAULT_CSV_OUT)
    p.add_argument(
        "--no-mini3",
        action="store_true",
        help="No anyadir clase mini3 (Stage 2 sin mini3).",
    )
    p.add_argument(
        "--no-interference",
        action="store_true",
        help="No anyadir clase interference (Stage 2 sin interferencia).",
    )
    p.add_argument(
        "--interference-cap-limit",
        type=int,
        default=None,
        help="Limite de capturas de interference (None = 9 disponibles).",
    )
    p.add_argument(
        "--mini3-cap-limit",
        type=int,
        default=None,
        help="Limite de capturas de mini3 (None = 12 disponibles).",
    )
    p.add_argument(
        "--stage2-root",
        type=Path,
        default=Path("data/stage2_classifier_v1"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.csv_in.exists():
        raise FileNotFoundError(f"No existe {args.csv_in}")

    # 1) Leer CSV original
    print(f"Leyendo {args.csv_in}...")
    with open(args.csv_in, encoding="utf-8") as f:
        rows_in = list(csv.DictReader(f))
    print(f"  Filas leidas: {len(rows_in)}")

    listdir_cache: dict[Path, set[str]] = {}

    def png_exists(rel: str) -> bool:
        full = args.stage2_root / rel
        d = full.parent
        if d not in listdir_cache:
            listdir_cache[d] = set(os.listdir(d)) if d.exists() else set()
        return full.name in listdir_cache[d]

    rows_with_png: list[dict] = []
    seen_filenames: set[str] = set()
    duplicates: list[dict] = []
    orphans: list[dict] = []
    for r in rows_in:
        fn = r["filename"]
        if not png_exists(fn):
            orphans.append(r)
            continue
        if fn in seen_filenames:
            duplicates.append(r)
            continue
        seen_filenames.add(fn)
        rows_with_png.append(r)
    print(
        f"  Unicas con PNG: {len(rows_with_png)} | "
        f"duplicadas: {len(duplicates)} | huerfanas: {len(orphans)}"
    )
    if duplicates:
        c = Counter(r["capture_id"] for r in duplicates)
        for cap, n in sorted(c.items()):
            print(f"    [dup] {cap}: {n}")
    if orphans:
        c = Counter(r["capture_id"] for r in orphans)
        for cap, n in sorted(c.items()):
            print(f"    [orph] {cap}: {n}")

    # 2) Reescribir filas existentes (f450, hunter, mavic)
    out_rows: list[dict] = []
    for r in rows_with_png:
        new = {col: "" for col in HEADER}
        new["filename"] = str(args.stage2_root / r["filename"]).replace("\\", "/")
        for k in (
            "label", "wifi_level", "condition", "antenna", "distance_m",
            "angle", "environment", "fc", "fs", "gain", "platform",
        ):
            new[k] = r.get(k, "")
        new["source_dataset"] = "stage2"
        if r["label"] == "mavic":
            sg, capid = classify_mavic_row(r["filename"], r["capture_id"])
            new["subgroup"] = sg
            new["capture_id"] = capid
        else:
            new["subgroup"] = "none"
            new["capture_id"] = r["capture_id"]
        out_rows.append(new)
    print(f"\nFilas stage2 tras dedup + subgroup: {len(out_rows)}")

    # 3) Anyadir mini3 desde v3
    if not args.no_mini3:
        print(f"\nAnyadiendo clase mini3 desde {V3_DRONE}...")
        if not V3_DRONE.exists():
            print(f"  AVISO: {V3_DRONE} no existe, saltando mini3")
        else:
            png_by_cap: dict[str, list[str]] = defaultdict(list)
            for fn in sorted(os.listdir(V3_DRONE)):
                if not fn.endswith(".png"):
                    continue
                cap = fn.rsplit("__seg", 1)[0]
                png_by_cap[cap].append(fn)

            cap_list = sorted(png_by_cap.keys())
            if args.mini3_cap_limit is not None:
                cap_list = cap_list[: args.mini3_cap_limit]
                print(f"  Limitando a {len(cap_list)} capturas: {cap_list}")
            else:
                print(f"  Incluyendo todas las {len(cap_list)} capturas")

            n_added = 0
            for cap in cap_list:
                dist, wifi = parse_mini3_capture(cap)
                for fn in png_by_cap[cap]:
                    rel = str(V3_DRONE / fn).replace("\\", "/")
                    out_rows.append({
                        "filename": rel,
                        "label": "mini3",
                        "subgroup": "none",
                        "wifi_level": wifi,
                        "condition": "clean",
                        "antenna": "directional",
                        "distance_m": dist,
                        "angle": "",
                        "environment": "outdoor",
                        "fc": "2450000000.0",
                        "fs": "40000000.0",
                        "gain": "10",
                        "capture_id": cap,
                        "platform": "drone",
                        "source_dataset": "v3",
                    })
                    n_added += 1
            print(f"  Filas mini3 anyadidas: {n_added}")

    # 4) Anyadir interference desde v3
    if not args.no_interference:
        print(f"\nAnyadiendo clase interference desde {V3_INTERFERENCE}...")
        if not V3_INTERFERENCE.exists():
            print(f"  AVISO: {V3_INTERFERENCE} no existe, saltando interference")
        else:
            png_by_cap = defaultdict(list)
            for fn in sorted(os.listdir(V3_INTERFERENCE)):
                if not fn.endswith(".png"):
                    continue
                cap = fn.rsplit("__seg", 1)[0]
                png_by_cap[cap].append(fn)

            cap_list = sorted(png_by_cap.keys())
            if args.interference_cap_limit is not None:
                cap_list = cap_list[: args.interference_cap_limit]
                print(f"  Limitando a {len(cap_list)} capturas: {cap_list}")
            else:
                print(f"  Incluyendo todas las {len(cap_list)} capturas")

            n_added = 0
            for cap in cap_list:
                wifi = ""
                for p in cap.split("_"):
                    if p.startswith("w") and p[1:].isdigit():
                        wifi = p[1:]
                        break
                for fn in png_by_cap[cap]:
                    rel = str(V3_INTERFERENCE / fn).replace("\\", "/")
                    out_rows.append({
                        "filename": rel,
                        "label": "interference",
                        "subgroup": "none",
                        "wifi_level": wifi,
                        "condition": "clean",
                        "antenna": "directional",
                        "distance_m": "",
                        "angle": "",
                        "environment": "indoor",
                        "fc": "",
                        "fs": "40000000.0",
                        "gain": "10",
                        "capture_id": cap,
                        "platform": "wifi",
                        "source_dataset": "v3",
                    })
                    n_added += 1
            print(f"  Filas interference anyadidas: {n_added}")

    # 5) Escribir CSV
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"\nCSV escrito en: {args.csv_out}")

    # 6) Resumen
    print("\n=== Resumen final ===")
    print(f"Total filas: {len(out_rows)}")
    print("\nPor label:")
    for lbl, n in sorted(Counter(r["label"] for r in out_rows).items()):
        print(f"  {lbl:<15s} {n:>5d}")
    print("\nPor (label, subgroup):")
    for (lbl, sg), n in sorted(Counter((r["label"], r["subgroup"]) for r in out_rows).items()):
        print(f"  {lbl:<15s} {sg:<10s} {n:>5d}")
    print("\nCapture_ids por label:")
    caps_by_label: dict[str, set[str]] = defaultdict(set)
    for r in out_rows:
        caps_by_label[r["label"]].add(r["capture_id"])
    for lbl in sorted(caps_by_label):
        print(f"  {lbl}: {len(caps_by_label[lbl])} capturas")


if __name__ == "__main__":
    main()
