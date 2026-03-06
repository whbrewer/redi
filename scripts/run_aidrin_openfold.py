#!/usr/bin/env python3
"""
AIDRIN before/after analysis for OpenFold / Life Sciences (6KWC).

Each amino acid position is one row:
  Raw:     position, aa_char, aa_type (polar/nonpolar/charged/special), aa_index
  AI-ready: position, encoded_int (REDI's integer encoding)

Metrics: completeness, duplicity, outliers.

Usage:
    source /opt/venvs/drai/bin/activate
    python3 scripts/run_aidrin_openfold.py
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

FASTA_PATH = "/opt/drai/openfold/examples/monomer/fasta_dir/6kwc.fasta"
REDI_OUT   = "/home/w1b/drai/apps/redi/output/openfold_redi_validate"
OUT_JSON   = "/home/w1b/drai/apps/redi/validate_results/aidrin_openfold.json"

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
VOCAB       = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}
VOCAB_REV   = {v: k for k, v in VOCAB.items()}
VOCAB_REV[0] = "-"

# Rough AA type classification
AA_TYPE = {
    "A": 0, "G": 0, "I": 0, "L": 0, "M": 0, "F": 0, "V": 0, "W": 0,  # nonpolar
    "S": 1, "T": 1, "C": 1, "Y": 1, "N": 1, "Q": 1,                    # polar
    "D": 2, "E": 2, "K": 2, "R": 2, "H": 2,                             # charged
    "P": 3,                                                               # special
}


def parse_fasta(path: str) -> str:
    seq = ""
    with open(path) as f:
        for line in f:
            if not line.startswith(">"):
                seq += line.strip().upper()
    return seq


def load_raw(seq: str) -> pd.DataFrame:
    rows = []
    for i, aa in enumerate(seq):
        rows.append({
            "position": i,
            "aa_index": VOCAB.get(aa, 0),
            "aa_type": AA_TYPE.get(aa, -1),
        })
    df = pd.DataFrame(rows)
    print(f"  Raw: {len(df)} rows × {len(df.columns)} cols")
    return df


def load_ai_ready() -> pd.DataFrame:
    h5_path = Path(REDI_OUT) / "data.h5"
    if not h5_path.exists():
        sys.exit(f"No HDF5 at {h5_path} — run validate_openfold.py first")
    with h5py.File(h5_path, "r") as f:
        X = None
        for split in ["train", "val", "test"]:
            if split in f and "X" in f[split] and len(f[split]["X"]) > 0:
                X = f[split]["X"][0]  # first (only) sequence
                break
    if X is None:
        sys.exit("No X data found in HDF5")

    # X is the full padded array (length seq_len); strip trailing zeros
    X_trimmed = X[X != 0]
    rows = [{"position": i, "encoded": int(v)} for i, v in enumerate(X_trimmed)]
    df = pd.DataFrame(rows)
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


def run_all(label: str, df: pd.DataFrame,
            metrics: list[tuple]) -> dict:
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
    print("AIDRIN DATA QUALITY: OPENFOLD 6KWC — RAW vs. REDI AI-READY")
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
    print("  Each row = one amino acid position (191 positions total)")


def main():
    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)

    seq = parse_fasta(FASTA_PATH)
    print(f"Sequence: {len(seq)} AA")

    print("\n[1/4] Loading raw data (per-position) ...")
    raw_df = load_raw(seq)

    print("\n[2/4] Loading REDI AI-ready data ...")
    ai_df = load_ai_ready()

    print("\n[3/4] Running AIDRIN on raw data ...")
    raw_results = run_all("raw", raw_df, RAW_METRICS)

    print("\n[4/4] Running AIDRIN on AI-ready data ...")
    ai_results = run_all("ai", ai_df, AI_METRICS)

    output = {
        "domain": "life_sciences",
        "protein": "6KWC",
        "seq_length": len(seq),
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
