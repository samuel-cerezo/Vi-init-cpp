#!/usr/bin/env python3
"""
Plot gravity components g_w_x, g_w_y, g_w_z vs t_frames for a chosen ref_idx,
and compute a FINAL-row summary ALWAYS across ALL ref_idx.

Optionally, when plotting a ref, also create ONE combined PNG by vertically stacking
the three individual plots (no subplots used).

Usage:
    # List available ref_idx
    python plot_gravity_by_ref.py --csv /path/to/file.csv --list

    # Plot gravity for one ref_idx (only this one is plotted)
    python plot_gravity_by_ref.py --csv /path/to/file.csv --ref 2001

    # Plot and also save a combined PNG
    python plot_gravity_by_ref.py --csv /path/to/file.csv --ref 2001 --combined

    # Compute the FINAL-row summary across ALL ref_idx
    python plot_gravity_by_ref.py --csv /path/to/file.csv --final-summary

Outputs:
    - g_w_x_vs_t_frames_ref<ref>.png
    - g_w_y_vs_t_frames_ref<ref>.png
    - g_w_z_vs_t_frames_ref<ref>.png
    - gravity_components_ref<ref>_COMBINED.png   [only with --combined]
    - final_rows_summary.csv  (FINAL row per ref_idx + a MEAN row)
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RMSE_COLS = ["rmse_g_deg", "rmse_v_ms", "rmse_ba_ms2", "rmse_bg_rads", "t_frames"]

def choose_final_row(gdf: pd.DataFrame) -> pd.Series:
    """Choose the final/stabilized row for a given ref group."""
    if "n_frames" in gdf.columns:
        idx = gdf["n_frames"].idxmax()
        return gdf.loc[idx]
    elif "t_frames" in gdf.columns:
        idx = gdf["t_frames"].idxmax()
        return gdf.loc[idx]
    else:
        return gdf.tail(1).iloc[0]

def make_plots_g(fdf: pd.DataFrame, ref_val: int, outdir: Path):
    fdf = fdf.sort_values("t_frames")
    saved = []
    for col in ["g_w_x", "g_w_y", "g_w_z"]:
        plt.figure(figsize=(9,5))
        plt.plot(fdf["t_frames"].values, fdf[col].values, linewidth=4)
        plt.xlabel("time [sec]")
        plt.ylabel(f"{col} [m/s²]")
        #plt.title(f"Evolución de {col} vs t_frames (ref_idx = {ref_val})")
        plt.tight_layout()
        out = outdir / f"{col}_ref{ref_val}.png"
        plt.savefig(out, dpi=600, bbox_inches="tight")
        print(f"Saved: {out}")
        saved.append(out)
    return saved

def make_plots_b(fdf: pd.DataFrame, ref_val: int, outdir: Path):
    fdf = fdf.sort_values("t_frames")
    saved = []
    for col in ["ba_w_x","ba_w_y","ba_w_z"]:
        plt.figure(figsize=(9,5))
        plt.plot(fdf["t_frames"].values, fdf[col].values, linewidth=4)
        plt.xlabel("time [sec]")
        plt.ylabel(f"{col} [m/s²]")
        #plt.title(f"Evolución de {col} vs t_frames (ref_idx = {ref_val})")
        plt.tight_layout()
        out = outdir / f"{col}_ref{ref_val}.png"
        plt.savefig(out, dpi=600, bbox_inches="tight")
        print(f"Saved: {out}")
        saved.append(out)
    return saved
def combine_pngs_vertically(png_paths, out_path: Path):
    try:
        from PIL import Image
    except Exception as e:
        print(f"Warning: PIL not available, cannot create combined PNG. {e}", file=sys.stderr)
        return None
    images = [Image.open(p) for p in png_paths]
    widths, heights = zip(*(im.size for im in images))
    max_w = max(widths)
    total_h = sum(heights)
    combined = Image.new("RGB", (max_w, total_h), (255, 255, 255))
    y = 0
    for im in images:
        # Paste centered
        x = (max_w - im.size[0]) // 2
        combined.paste(im, (x, y))
        y += im.size[1]
    combined.save(out_path)
    print(f"Saved: {out_path}")
    return out_path

def compute_final_rows_summary(df: pd.DataFrame, outdir: Path) -> Path:
    rows = []
    for ref, gdf in df.groupby("ref_idx"):
        final_row = choose_final_row(gdf)
        row_out = {"ref_idx": ref}
        for c in RMSE_COLS:
            if c in final_row.index:
                row_out[c] = final_row[c]
        rows.append(row_out)
    summary = pd.DataFrame(rows).sort_values("ref_idx")

    # Append a mean row
    mean_row = {"ref_idx": "MEAN"}
    for c in RMSE_COLS:
        if c in summary.columns:
            mean_row[c] = summary[c].mean(skipna=True)
    summary = pd.concat([summary, pd.DataFrame([mean_row])], ignore_index=True)

    out_csv = outdir / "final_rows_summary.csv"
    summary.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    return out_csv

def main():
    ap = argparse.ArgumentParser(description="Plot gravity by ref_idx and compute FINAL-row summary (all refs).")
    ap.add_argument("--csv", required=True, help="Path to CSV file")
    ap.add_argument("--ref", type=int, help="ref_idx value to plot (only this ref is plotted)")
    ap.add_argument("--list", action="store_true", help="List unique ref_idx values and exit")
    ap.add_argument("--combined", action="store_true", help="After plotting, join the three PNGs into one combined PNG")
    ap.add_argument("--final-summary", action="store_true", dest="final_summary", help="Compute FINAL-row summary across ALL ref_idx")
    ap.add_argument("--outdir", default=None, help="Directory to save outputs (default: alongside CSV)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)

    outdir = Path(args.outdir) if args.outdir else csv_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    if args.list:
        if "ref_idx" not in df.columns:
            print("Error: CSV must have 'ref_idx' column.", file=sys.stderr)
            sys.exit(1)
        vals = sorted(df["ref_idx"].dropna().unique().tolist())
        print("Available ref_idx values:")
        for v in vals:
            print(v)
        return

    if args.ref is not None:
        for c in ["ref_idx","t_frames","g_w_x","g_w_y","g_w_z"]:
            if c not in df.columns:
                print(f"Error: Missing required column for plotting: {c}", file=sys.stderr)
                sys.exit(1)
        fdf = df[df["ref_idx"] == args.ref].copy()
        if fdf.empty:
            vals = sorted(df["ref_idx"].dropna().unique().tolist())
            print(f"Error: No rows found for ref_idx={args.ref}.", file=sys.stderr)
            print(f"Hint: available ref_idx values include: {vals[:20]}{' ...' if len(vals) > 20 else ''}", file=sys.stderr)
            sys.exit(2)
        saved_g = make_plots_g(fdf, args.ref, outdir)
        saved_b = make_plots_b(fdf, args.ref, outdir)
       
        if args.combined and len(saved_g) == 3 and len(saved_b) == 3:
            combined_path = outdir / f"gravity_components_ref{args.ref}_COMBINED.png"
            combine_pngs_vertically(saved_g, combined_path)
            combined_path = outdir / f"bias_components_ref{args.ref}_COMBINED.png"
            combine_pngs_vertically(saved_b, combined_path)

    if args.final_summary:
        for c in ["ref_idx"] + RMSE_COLS:
            if c not in df.columns:
                print(f"Error: Missing required column for summary: {c}", file=sys.stderr)
                sys.exit(1)
        compute_final_rows_summary(df, outdir)

    if (args.ref is None) and (not args.final_summary) and (not args.list):
        print("Nothing to do. Use --ref to plot, --final-summary to compute the summary, --combined to join PNGs, or --list.", file=sys.stderr)
        sys.exit(0)

if __name__ == "__main__":
    main()


#   Usage:
#       /usr/bin/env python3 "/Users/samucerezo/dev/src/Vi-init-cpp/rmse_analysis.py" --csv /Users/samucerezo/dev/src/Vi-init-cpp/results/full_MH01_20251002_094742.csv --combined --final-summary --ref 14