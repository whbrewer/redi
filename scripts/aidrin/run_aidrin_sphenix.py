#!/usr/bin/env python3
"""
AIDRIN before/after analysis for sPHENIX nuclear physics data.

Each row = one detector event (array) summarized as statistics:
  Raw:     sum_hits, max_adc, mean_adc, nonzero_fraction (per event, 1310 events)
  AI-ready: same stats from REDI output arrays

Usage:
    source /opt/venvs/drai/bin/activate
    python3 scripts/run_aidrin_sphenix.py
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

SPHENIX_DIR = "/opt/drai/sphenix/outer/12-2_0-0"
REDI_OUT    = "/home/w1b/drai/apps/redi/output/sphenix_redi_validate"
OUT_JSON    = "/home/w1b/drai/apps/redi/validate_results/aidrin_sphenix.json"

N_SAMPLE = 200   # arrays to sample for AIDRIN


def event_stats(arr: np.ndarray) -> dict:
    """Summarize one (192, 249, 16) detector array as scalar stats."""
    flat = arr.astype(float).ravel()
    nonzero = flat[flat > 0]
    return {
        "sum_hits": float(flat.sum()),
        "max_adc": float(flat.max()),
        "mean_adc": float(flat.mean()),
        "nonzero_fraction": float(len(nonzero) / len(flat)),
        "mean_nonzero_adc": float(nonzero.mean()) if len(nonzero) > 0 else 0.0,
    }


def load_raw(sample: int) -> pd.DataFrame:
    import glob
    files = sorted(glob.glob(f"{SPHENIX_DIR}/*.npy"))
    rng = np.random.default_rng(42)
    files = rng.choice(files, size=min(sample, len(files)), replace=False).tolist()
    rows = [event_stats(np.load(f)) for f in files]
    df = pd.DataFrame(rows)
    print(f"  Raw: {len(df)} events × {len(df.columns)} cols")
    return df


def load_ai_ready(sample: int) -> pd.DataFrame:
    rows = []
    for split in ["train", "val", "test"]:
        path = Path(REDI_OUT) / f"{split}.npz"
        if not path.exists():
            continue
        X = np.load(path)["X"]
        for i in range(min(sample - len(rows), len(X))):
            rows.append(event_stats(X[i]))
        if len(rows) >= sample:
            break
    if not rows:
        sys.exit(f"No REDI output in {REDI_OUT} — run validate_sphenix.py first")
    df = pd.DataFrame(rows)
    print(f"  AI-ready: {len(df)} events × {len(df.columns)} cols")
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
                js = stdout.rfind("{", 0, key_pos)
                if js >= 0:
                    stdout = stdout[js:]
        return json.loads(stdout)
    except subprocess.TimeoutExpired:
        os.unlink(cfg_path)
        print(f"    [timeout] {metric}")
        return None
    except (json.JSONDecodeError, Exception) as e:
        print(f"    [error] {metric}: {e}")
        return None


def run_all(label: str, df: pd.DataFrame, metrics: list) -> dict:
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


METRICS = [
    ("completeness", {}),
    ("duplicity",    {}),
    ("outliers",     {}),
]


def _scalar(d, *keys, default="—"):
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
        if d == default:
            return default
    return round(d, 4) if isinstance(d, float) else d


def summarise(results: dict) -> dict:
    s = {}
    s["completeness"] = _scalar(
        results.get("completeness", {}), "completeness", "Overall Completeness")
    s["duplicate_ratio"] = _scalar(
        results.get("duplicity", {}), "duplicity", "Duplicity scores",
        "Overall duplicity of the dataset")
    per_col = _scalar(results.get("outliers", {}), "outliers", "Outlier scores")
    if isinstance(per_col, dict):
        ratios = [v for v in per_col.values() if isinstance(v, (int, float))]
        s["avg_outlier_ratio"] = round(sum(ratios) / len(ratios), 4) if ratios else "—"
    else:
        s["avg_outlier_ratio"] = "—"
    return s


def fmt_delta(rv, av, direction):
    try:
        r, a = float(rv), float(av)
    except (ValueError, TypeError):
        return ""
    diff = a - r
    if abs(diff) < 1e-6:
        return f"{diff:+.4f} ="
    symbol = ("▲" if diff > 0 else "▼") if direction == "↑" else ("▲" if diff < 0 else "▼")
    return f"{diff:+.4f} {symbol}"


METRICS_DISPLAY = [
    ("completeness",      "Completeness",     "Data Quality", "↑"),
    ("duplicate_ratio",   "Duplicate ratio",  "Data Quality", "↓"),
    ("avg_outlier_ratio", "Avg outlier ratio","Data Quality", "↓"),
]


def print_table(raw_summ: dict, ai_summ: dict):
    W = 80
    print("\n" + "=" * W)
    print("AIDRIN DATA QUALITY: sPHENIX — RAW vs. REDI AI-READY")
    print("=" * W)
    print(f"  {'Metric':<26} {'Category':<14} {'Dir':>3}  {'Raw':>10}  {'AI-ready':>10}  {'Δ':>14}")
    print(f"  {'-'*26} {'-'*14} {'-'*3}  {'-'*10}  {'-'*10}  {'-'*14}")
    for key, label, cat, direction in METRICS_DISPLAY:
        r, a = raw_summ.get(key, "—"), ai_summ.get(key, "—")
        delta = fmt_delta(r, a, direction)
        rs = f"{r:.4f}" if isinstance(r, float) else str(r)
        as_ = f"{a:.4f}" if isinstance(a, float) else str(a)
        print(f"  {label:<26} {cat:<14} {direction:>3}  {rs:>10}  {as_:>10}  {delta:>14}")
    print("=" * W)
    print("  Each row = one detector event (sum/max/mean/nonzero stats per event)")


def main():
    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Loading raw event stats ...")
    raw_df = load_raw(N_SAMPLE)

    print("\n[2/4] Loading REDI AI-ready event stats ...")
    ai_df = load_ai_ready(N_SAMPLE)

    print("\n[3/4] Running AIDRIN on raw ...")
    raw_results = run_all("raw", raw_df, METRICS)

    print("\n[4/4] Running AIDRIN on AI-ready ...")
    ai_results = run_all("ai", ai_df, METRICS)

    output = {
        "domain": "nuclear_physics",
        "dataset": "sPHENIX outer TPC 12-2_0-0",
        "n_events": N_SAMPLE,
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
