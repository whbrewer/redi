#!/usr/bin/env python3
"""
AIDRIN before/after analysis for TaiESM1 2m_temperature climate data.

Raw:     one NetCDF chunk (tas field, lat×lon×time)
AI-ready: corresponding REDI NPZ (regridded + normalized)

Metrics: completeness, outliers, class_imbalance (by season), duplicity.

Usage:
    source /opt/venvs/drai/bin/activate
    python3 scripts/run_aidrin_climate.py [--chunk YSTR] [--sample N]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

# ─── Paths ────────────────────────────────────────────────────────────────────
RAW_DIR   = "/opt/drai/datasets/CMIP6/TaiESM1/raw/2m_temperature"
REDI_DIR  = "/home/w1b/drai/apps/redi/output"
OUT_JSON  = "/home/w1b/drai/apps/redi/validate_results/aidrin_climate.json"

DEFAULT_CHUNK = "185001010000-186001010000"

SEASONS = {0: "DJF", 1: "MAM", 2: "JJA", 3: "SON"}  # month→season index

# ─── Data loading ──────────────────────────────────────────────────────────────

def month_to_season(month: int) -> int:
    """Return season index 0-3 for a given month (1-12)."""
    return (month % 12) // 3  # DJF=0, MAM=1, JJA=2, SON=3


def load_raw(chunk: str, sample: int) -> "pd.DataFrame":
    import pandas as pd
    raw_nc = f"{RAW_DIR}/2m_temperature_{chunk}_raw.nc"
    if not Path(raw_nc).exists():
        sys.exit(f"Raw NetCDF not found: {raw_nc}")

    print(f"  Loading raw NetCDF: {raw_nc}")
    ds = xr.open_dataset(raw_nc)
    tas = ds["tas"]  # (time, lat, lon)

    T, LAT, LON = tas.shape
    total = T * LAT * LON
    print(f"  Raw shape: {tas.shape}  ({total:,} total values)")

    # Sample time steps to keep manageable
    rng = np.random.default_rng(42)
    if sample and T * LAT * LON > sample:
        # Sample random (t, lat, lon) indices
        t_idx   = rng.integers(0, T,   size=sample)
        lat_idx = rng.integers(0, LAT, size=sample)
        lon_idx = rng.integers(0, LON, size=sample)

        times = tas.time.values[t_idx]
        months = [int(str(t)[:7].split("-")[1]) for t in times]
        values = tas.values[t_idx, lat_idx, lon_idx]

        df = pd.DataFrame({
            "lat_idx":   lat_idx,
            "lon_idx":   lon_idx,
            "time_idx":  t_idx,
            "value":     values.astype(np.float64),
            "season":    [month_to_season(m) for m in months],
            "hemisphere": np.sign(
                ds.lat.values[lat_idx]).astype(int),  # -1/0/+1
        })
    else:
        # Use all values (rare for full data, useful for small test chunks)
        arr = tas.values.reshape(-1)
        months_all = np.repeat(
            [int(str(t)[:7].split("-")[1]) for t in tas.time.values],
            LAT * LON
        )
        lats = np.tile(np.repeat(np.arange(LAT), LON), T)
        df = pd.DataFrame({
            "lat_idx":   lats,
            "time_idx":  np.repeat(np.arange(T), LAT * LON),
            "value":     arr.astype(np.float64),
            "season":    [month_to_season(m) for m in months_all],
        })

    fill_val = ds["tas"].attrs.get("_FillValue", None) or ds["tas"].attrs.get("missing_value", None)
    if fill_val is not None:
        df.loc[np.isclose(df["value"], fill_val, rtol=1e-4), "value"] = np.nan

    n_nan = df["value"].isna().sum()
    print(f"  Sample: {len(df):,} rows  NaN: {n_nan:,} ({100*n_nan/len(df):.2f}%)")
    ds.close()
    return df


def load_ai_ready(chunk: str, sample: int) -> "pd.DataFrame":
    import pandas as pd
    # Find REDI output dir for this chunk
    pattern = f"taiesm1_2m_temp_{chunk}"
    candidates = sorted(Path(REDI_DIR).glob(f"{pattern}*/*/data.npz"))
    if not candidates:
        sys.exit(f"No REDI output found for chunk {chunk} in {REDI_DIR}/{pattern}*/")

    npz_path = candidates[-1]  # most recent
    print(f"  Loading AI-ready NPZ: {npz_path}")
    d = np.load(npz_path)
    X = d["X"]  # (time, lat, lon) normalized
    T, LAT, LON = X.shape
    print(f"  AI-ready shape: {X.shape}  ({T*LAT*LON:,} total values)")

    rng = np.random.default_rng(42)
    if sample and T * LAT * LON > sample:
        t_idx   = rng.integers(0, T,   size=sample)
        lat_idx = rng.integers(0, LAT, size=sample)
        lon_idx = rng.integers(0, LON, size=sample)
        values  = X[t_idx, lat_idx, lon_idx]
        # Approximate months from time index (10-year chunk = 14601 days / ~365.25)
        months = ((t_idx / T * 12).astype(int) % 12) + 1

        df = pd.DataFrame({
            "lat_idx":   lat_idx,
            "lon_idx":   lon_idx,
            "time_idx":  t_idx,
            "value":     values.astype(np.float64),
            "season":    [month_to_season(m) for m in months],
            "hemisphere": np.where(lat_idx < LAT // 2, -1, 1),
        })
    else:
        arr = X.reshape(-1)
        df = pd.DataFrame({"value": arr.astype(np.float64)})

    n_nan = df["value"].isna().sum()
    print(f"  Sample: {len(df):,} rows  NaN: {n_nan:,} ({100*n_nan/len(df):.2f}%)")
    return df


# ─── AIDRIN metrics ───────────────────────────────────────────────────────────

RAW_METRICS = [
    ("completeness",    {}),
    ("duplicity",       {}),
    ("outliers",        {}),
    ("class_imbalance", {"target_column": "season"}),
]

AI_METRICS = [
    ("completeness",    {}),
    ("duplicity",       {}),
    ("outliers",        {}),
    ("class_imbalance", {"target_column": "season"}),
]


def run_metric(metric: str, csv_path: str, extra: dict) -> dict | None:
    config = {"file_path": csv_path, "metrics": [metric], **extra}
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as cfg:
        json.dump(config, cfg)
        cfg_path = cfg.name

    try:
        result = subprocess.run(
            ["aidrin", "batch", "--no-viz", cfg_path],
            capture_output=True, text=True, timeout=180,
        )
        os.unlink(cfg_path)
        if result.returncode != 0:
            print(f"    [error] {metric}: {result.stderr[:200].strip()}")
            return None
        stdout = result.stdout
        json_start = stdout.rfind('\n{')
        if json_start >= 0:
            stdout = stdout[json_start + 1:]
        else:
            key_pos = stdout.find(f'"{metric}"')
            if key_pos > 0:
                json_start = stdout.rfind('{', 0, key_pos)
                if json_start >= 0:
                    stdout = stdout[json_start:]
        return json.loads(stdout)
    except subprocess.TimeoutExpired:
        os.unlink(cfg_path)
        print(f"    [timeout] {metric}")
        return None
    except json.JSONDecodeError:
        print(f"    [parse-error] {metric}: {result.stdout[:200]}")
        return None
    except Exception as e:
        print(f"    [exception] {metric}: {e}")
        return None


def run_all(label: str, df: "pd.DataFrame", metrics: list[tuple]) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as tmp:
        df.to_csv(tmp.name, index=False)
        csv_path = tmp.name

    results = {}
    for metric, extra in metrics:
        print(f"  [{label}] {metric} ...", end=" ", flush=True)
        out = run_metric(metric, csv_path, extra)
        if out is not None:
            results[metric] = out
            print("OK")
        else:
            print("SKIP")

    os.unlink(csv_path)
    return results


# ─── Summarise ────────────────────────────────────────────────────────────────

def _scalar(d, *keys, default="—"):
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
        if d == default:
            return default
    if isinstance(d, float):
        return round(d, 4)
    return d


def summarise(results: dict) -> dict:
    s = {}

    r = results.get("completeness", {})
    s["completeness"] = _scalar(r, "completeness", "Overall Completeness")

    r = results.get("duplicity", {})
    s["duplicate_ratio"] = _scalar(
        r, "duplicity", "Duplicity scores", "Overall duplicity of the dataset")

    r = results.get("outliers", {})
    per_col = _scalar(r, "outliers", "Outlier scores")
    if isinstance(per_col, dict):
        ratios = [v for v in per_col.values() if isinstance(v, (int, float))]
        s["avg_outlier_ratio"] = round(sum(ratios) / len(ratios), 4) if ratios else "—"
        s["max_outlier_ratio"] = round(max(ratios), 4) if ratios else "—"
    else:
        s["avg_outlier_ratio"] = s["max_outlier_ratio"] = "—"

    r = results.get("class_imbalance", {})
    imb = r.get("class_imbalance", {}).get("Imbalance degree", {})
    s["class_imbalance"] = (
        round(float(imb["Imbalance Degree score"]), 4)
        if "Imbalance Degree score" in imb else "—"
    )

    return s


def fmt_delta(raw_val, ai_val, direction):
    try:
        r, a = float(raw_val), float(ai_val)
    except (ValueError, TypeError):
        return ""
    diff = a - r
    if abs(diff) < 1e-6:
        symbol = "="
    elif direction == "↑":
        symbol = "▲" if diff > 0 else "▼"
    else:
        symbol = "▲" if diff < 0 else "▼"
    return f"{diff:+.4f} {symbol}"


METRICS_DISPLAY = [
    ("completeness",      "Completeness",          "Data Quality", "↑"),
    ("duplicate_ratio",   "Duplicate ratio",        "Data Quality", "↓"),
    ("avg_outlier_ratio", "Avg outlier ratio",      "Data Quality", "↓"),
    ("max_outlier_ratio", "Max outlier ratio",      "Data Quality", "↓"),
    ("class_imbalance",   "Class imbalance (season)", "Fairness",   "↓"),
]


def print_table(raw_summ: dict, ai_summ: dict, chunk: str):
    W = 85
    print("\n" + "=" * W)
    print("AIDRIN DATA QUALITY ASSESSMENT: TaiESM1 CLIMATE — RAW vs. REDI AI-READY")
    print("=" * W)
    print(f"  Chunk:    {chunk}")
    print(f"  Raw:      TaiESM1 2m_temperature NetCDF (192×288 grid, full resolution)")
    print(f"  AI-ready: REDI NPZ (regridded 128×256 ClimaX grid, normalized)\n")

    header = f"{'Metric':<28} {'Category':<14} {'Dir':>3}  {'Raw':>10}  {'AI-ready':>10}  {'Δ':>14}"
    print(header)
    print("-" * W)
    for key, label, cat, direction in METRICS_DISPLAY:
        r = raw_summ.get(key, "—")
        a = ai_summ.get(key, "—")
        delta = fmt_delta(r, a, direction)
        rs = f"{r:.4f}" if isinstance(r, float) else str(r)
        as_ = f"{a:.4f}" if isinstance(a, float) else str(a)
        print(f"  {label:<26} {cat:<14} {direction:>3}  {rs:>10}  {as_:>10}  {delta:>14}")
    print("=" * W)
    print("  ▲ = improvement after REDI   ▼ = degradation   = unchanged")
    print("  class_imbalance: labels = season (DJF/MAM/JJA/SON)")


def write_latex(raw_summ: dict, ai_summ: dict, out_path: str):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{AIDRIN data quality assessment for TaiESM1 2m\_temperature climate data "
        r"before (raw NetCDF, 192$\times$288 grid) and after REDI processing "
        r"(regridded 128$\times$256 ClimaX grid, normalized). "
        r"Class imbalance uses season labels (DJF/MAM/JJA/SON).}",
        r"\label{tab:aidrin_climate}",
        r"\begin{tabular}{llrrl}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{Category} & \textbf{Raw} & \textbf{AI-ready} & \textbf{$\Delta$} \\",
        r"\midrule",
    ]
    for key, label, cat, direction in METRICS_DISPLAY:
        r = raw_summ.get(key, "—")
        a = ai_summ.get(key, "—")
        delta = fmt_delta(r, a, direction)
        rs = f"{r:.3f}" if isinstance(r, float) else str(r)
        as_ = f"{a:.3f}" if isinstance(a, float) else str(a)
        try:
            rv, av = float(r), float(a)
            diff = av - rv
            improved = (direction == "↑" and diff > 1e-6) or \
                       (direction == "↓" and diff < -1e-6)
            if improved:
                as_ = r"\textbf{" + as_ + "}"
                delta = r"\textbf{" + delta + "}"
        except (ValueError, TypeError):
            pass
        lines.append(f"  {label} & {cat} & {rs} & {as_} & {delta} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    Path(out_path).write_text("\n".join(lines) + "\n")
    print(f"\nLaTeX table saved → {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", default=DEFAULT_CHUNK,
                        help=f"Year-string chunk to use (default: {DEFAULT_CHUNK})")
    parser.add_argument("--sample", type=int, default=200_000,
                        help="Max grid-point samples per dataset (default: 200000)")
    args = parser.parse_args()

    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/4] Loading raw NetCDF data (chunk {args.chunk}) ...")
    raw_df = load_raw(args.chunk, args.sample)

    print(f"\n[2/4] Loading REDI AI-ready NPZ ...")
    ai_df = load_ai_ready(args.chunk, args.sample)

    print(f"\n[3/4] Running {len(RAW_METRICS)} AIDRIN metrics on raw data ...")
    raw_results = run_all("raw", raw_df, RAW_METRICS)

    print(f"\n[4/4] Running {len(AI_METRICS)} AIDRIN metrics on AI-ready data ...")
    ai_results = run_all("ai", ai_df, AI_METRICS)

    output = {
        "chunk": args.chunk,
        "sample_size": args.sample,
        "raw_shape": "14601×192×288",
        "ai_ready_shape": "14601×128×256",
        "raw": raw_results,
        "ai_ready": ai_results,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nFull results → {OUT_JSON}")

    raw_summ = summarise(raw_results)
    ai_summ  = summarise(ai_results)
    print_table(raw_summ, ai_summ, args.chunk)

    latex_path = OUT_JSON.replace(".json", ".tex")
    write_latex(raw_summ, ai_summ, latex_path)


if __name__ == "__main__":
    main()
