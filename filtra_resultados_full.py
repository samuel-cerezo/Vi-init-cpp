#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Genera tres CSV (-before, -stable, -after) con la MISMA cantidad de filas,
filtrando ref_idx que cumplan TODAS las condiciones:
  - >= N filas 'Before'
  - >= 1 fila 'Stable'
  - >= 1 fila 'After'

Para cada ref_idx válido:
  - -before: primera fila 'Before' (orden temporal)
  - -stable: primera fila 'Stable' (orden temporal)
  - -after : última  fila 'After'  (orden temporal)

Uso:
    python filtra_resultados_full.py /ruta/al/archivo.csv --min-before 3
"""

import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Filtra por ref_idx (Before>=N, Stable>=1, After>=1) y genera 3 CSV emparejados.")
    parser.add_argument("input_csv", type=str, help="Ruta al CSV original.")
    parser.add_argument("--min-before", type=int, default=3, help="Mínimo de filas 'Before' por ref_idx. (default: 3)")
    args = parser.parse_args()

    input_path = Path(args.input_csv).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {input_path}")

    # === Lectura y saneo mínimo ===
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    required = {"ref_idx", "status"}
    if not required.issubset(df.columns):
        raise ValueError(f"El CSV debe contener las columnas: {', '.join(sorted(required))}")

    # Columnas para orden temporal
    order_cols = []
    if "t_frames" in df.columns: order_cols.append("t_frames")
    if "n_frames" in df.columns: order_cols.append("n_frames")
    if not order_cols:
        df = df.reset_index().rename(columns={"index": "_row_index"})
        order_cols = ["_row_index"]

    # Normalización de status
    status_series = df["status"].fillna("").astype(str).str.strip()
    try:
        df["status_norm"] = status_series.str.casefold()
    except AttributeError:
        df["status_norm"] = status_series.str.lower()

    # === Conteos por estado ===
    mask_before = df["status_norm"].eq("before")
    mask_stable = df["status_norm"].eq("stable")
    mask_after  = df["status_norm"].eq("after")

    cnt_before = df[mask_before].groupby("ref_idx").size().rename("count_before")
    cnt_stable = df[mask_stable].groupby("ref_idx").size().rename("count_stable")
    cnt_after  = df[mask_after ].groupby("ref_idx").size().rename("count_after")

    refs_before_ok = set(cnt_before[cnt_before >= args.min_before].index.tolist())
    refs_stable_ok = set(cnt_stable.index.tolist())  # >=1
    refs_after_ok  = set(cnt_after.index.tolist())   # >=1

    # === Intersección (criterio final) ===
    refs_valid = refs_before_ok & refs_stable_ok & refs_after_ok

    # Resumen previo
    total_refs   = df["ref_idx"].nunique()
    n_before_ok  = len(refs_before_ok)
    n_stable_ok  = len(refs_stable_ok)
    n_after_ok   = len(refs_after_ok)
    n_valid      = len(refs_valid)

    # Subconjunto final
    dfv = df[df["ref_idx"].isin(refs_valid)].copy()

    # Mantener orden de columnas original en salida
    output_cols = [c for c in df.columns if c not in ("status_norm", "_row_index")]

    # Helpers: primera/última por estado
    def first_by_status(g, status_str):
        tmp = g[g["status_norm"].eq(status_str)]
        if tmp.empty:  # no debería ocurrir tras intersección, defensa
            return None
        return tmp.sort_values(order_cols).iloc[[0]][output_cols]

    def last_by_status(g, status_str):
        tmp = g[g["status_norm"].eq(status_str)]
        if tmp.empty:
            return None
        return tmp.sort_values(order_cols).iloc[[-1]][output_cols]

    rows_before, rows_stable, rows_after = [], [], []

    # Procesar por ref_idx válido
    for rid, g in dfv.groupby("ref_idx", sort=True):
        b = first_by_status(g, "before")
        s = first_by_status(g, "stable")
        a = last_by_status (g, "after")

        # Todos deben existir por construcción; añadimos defensa por si acaso
        if b is None or s is None or a is None:
            continue

        rows_before.append(b)
        rows_stable.append(s)
        rows_after.append(a)

    # Concatenar
    df_before_out = pd.concat(rows_before, ignore_index=True) if rows_before else pd.DataFrame(columns=output_cols)
    df_stable_out = pd.concat(rows_stable, ignore_index=True) if rows_stable else pd.DataFrame(columns=output_cols)
    df_after_out  = pd.concat(rows_after , ignore_index=True) if rows_after  else pd.DataFrame(columns=output_cols)

    # Escritura
    base = input_path.stem
    out_before = input_path.with_name(f"{base}-before.csv")
    out_stable = input_path.with_name(f"{base}-stable.csv")
    out_after  = input_path.with_name(f"{base}-after.csv")

    df_before_out.to_csv(out_before, index=False)
    df_stable_out.to_csv(out_stable, index=False)
    df_after_out.to_csv(out_after, index=False)

    # Resumen final
    print("=== Resumen (intersección: Before>=N ∩ Stable>=1 ∩ After>=1) ===")
    print(f"Archivo de entrada: {input_path}")
    print(f"Umbral min Before: {args.min_before}")
    print(f"ref_idx totales: {total_refs}")
    print(f"ref_idx con >= {args.min_before} 'Before': {n_before_ok}")
    print(f"ref_idx con >= 1 'Stable'    : {n_stable_ok}")
    print(f"ref_idx con >= 1 'After'     : {n_after_ok}")
    print(f"ref_idx válidos (intersección): {n_valid}")
    print()
    print("=== Salidas (mismo número de filas) ===")
    print(f"-before: {out_before}   (filas: {len(df_before_out)})")
    print(f"-stable: {out_stable}   (filas: {len(df_stable_out)})")
    print(f"-after : {out_after}    (filas: {len(df_after_out)})")


if __name__ == "__main__":
    main()


#    df["status_norm"] = df["status"].astype(str).str.strip().str.casefold()


# usage:
#   /usr/bin/env python3  filtra_resultados_full.py /Users/samucerezo/dev/src/Vi-init-cpp/results/full_MH01.csv --min-before 3
