#!/usr/bin/env python3
"""
AIDRIN before/after analysis for AFLOW materials data.

Raw:     6-column CSV (auid, compound, band_gap, formation_enthalpy, density, cif)
AI-ready: REDI NPZ output (9 structures × 135 features)

Metrics: completeness, outliers, duplicity.

Usage:
    source /opt/venvs/drai/bin/activate
    python3 scripts/run_aidrin_materials.py
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

AFLOW_GT  = "/opt/drai/aflow/aflow_ml_ready.csv"
REDI_OUT  = "/home/w1b/drai/apps/redi/output/aflow_redi_validate"
OUT_JSON  = "/home/w1b/drai/apps/redi/validate_results/aidrin_aflow.json"

RAW_METRICS = [
    ("completeness", {}),
    ("duplicity",    {}),
    ("outliers",     {}),
]

AI_METRICS = [
    ("completeness", {}),
    ("duplicity",    {}),
    ("outliers",     {}),
]


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(AFLOW_GT)
    # Use numeric columns only: band_gap, formation_enthalpy, density
    keep = ["band_gap", "formation_enthalpy", "density"]
    df_raw = df[keep].copy()
    # Drop rows with all-NaN
    df_raw = df_raw.dropna(how="all")
    print(f"  Raw: {len(df_raw)} rows × {len(df_raw.columns)} cols")
    return df_raw


def load_ai_ready() -> pd.DataFrame:
    """Load REDI NPZ and flatten to DataFrame of numeric features."""
    X_parts, y_parts = [], []
    for split in ["train", "val", "test"]:
        path = Path(REDI_OUT) / f"{split}.npz"
        if path.exists():
            d = np.load(path, allow_pickle=True)
            if "X" in d:
                X_parts.append(d["X"])
            if "y" in d:
                y_parts.append(d["y"])

    if not X_parts:
        sys.exit(f"No REDI NPZ found in {REDI_OUT} — run validate_materials.py first")

    X = np.vstack(X_parts).astype(float)
    y = np.concatenate(y_parts).astype(float)

    # Build DataFrame: use a subset of feature columns to keep AIDRIN fast
    # Take first 20 Magpie features + target
    col_names = [f"feat_{i}" for i in range(X.shape[1])] + ["band_gap"]
    df = pd.DataFrame(np.column_stack([X, y]), columns=col_names)
    # Drop all-NaN columns (cif column)
    df = df.dropna(axis=1, how="all")
    print(f"  AI-ready: {len(df)} rows × {len(df.columns)} cols")
    return df


def run_metric(metric: str, csv_path: str, extra: dict) -> dict | None:
    config = {"file_path": csv_path, "metrics": [metric], **extra}
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as cfg:
        json.dump(config, cfg)
        cfg_path = cfg.name
    try:
        result = subprocess.run(
            ["aidrin", "batch", "--no-viz", cfg_path],
            capture_output=True, text=True, timeout=120,
        )
        os.unlink(cfg_path)
        if result.returncode != 0:
            print(f"    [error] {metric}: {result.stderr[:200].strip()}")
            return None
        stdout = result.stdout
        json_start = stdout.rfind("\n{")
        if json_start >= 0:
            stdout = stdout[json_start + 1:]
        else:
            key_pos = stdout.find(f'"{metric}"')
            if key_pos > 0:
                json_start = stdout.rfind("{", 0, key_pos)
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


def run_all(label: str, df: pd.DataFrame, metrics: list[tuple]) -> dict:
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
    else:
        s["avg_outlier_ratio"] = "—"
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
    ("completeness",      "Completeness",     "Data Quality", "↑"),
    ("duplicate_ratio",   "Duplicate ratio",  "Data Quality", "↓"),
    ("avg_outlier_ratio", "Avg outlier ratio","Data Quality", "↓"),
]


def print_table(raw_summ: dict, ai_summ: dict):
    W = 80
    print("\n" + "=" * W)
    print("AIDRIN DATA QUALITY: AFLOW MATERIALS — RAW vs. REDI AI-READY")
    print("=" * W)
    print(f"  {'Metric':<26} {'Category':<14} {'Dir':>3}  {'Raw':>10}  {'AI-ready':>10}  {'Δ':>14}")
    print(f"  {'-'*26} {'-'*14} {'-'*3}  {'-'*10}  {'-'*10}  {'-'*14}")
    for key, label, cat, direction in METRICS_DISPLAY:
        r = raw_summ.get(key, "—")
        a = ai_summ.get(key, "—")
        delta = fmt_delta(r, a, direction)
        rs = f"{r:.4f}" if isinstance(r, float) else str(r)
        as_ = f"{a:.4f}" if isinstance(a, float) else str(a)
        print(f"  {label:<26} {cat:<14} {direction:>3}  {rs:>10}  {as_:>10}  {delta:>14}")
    print("=" * W)


def main():
    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Loading raw data ...")
    raw_df = load_raw()

    print("\n[2/4] Loading REDI AI-ready data ...")
    ai_df = load_ai_ready()

    print("\n[3/4] Running AIDRIN on raw data ...")
    raw_results = run_all("raw", raw_df, RAW_METRICS)

    print("\n[4/4] Running AIDRIN on AI-ready data ...")
    ai_results = run_all("ai", ai_df, AI_METRICS)

    output = {
        "domain": "materials",
        "n_structures": len(raw_df),
        "raw": raw_results,
        "ai_ready": ai_results,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nFull results → {OUT_JSON}")

    raw_summ = summarise(raw_results)
    ai_summ = summarise(ai_results)
    print_table(raw_summ, ai_summ)


if __name__ == "__main__":
    main()
