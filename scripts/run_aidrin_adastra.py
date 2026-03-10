#!/usr/bin/env python3
"""
Full 15-metric AIDRIN before/after analysis: Adastra MI250 HPC raw vs. REDI AI-ready.

Demonstrates data quality, fairness, and privacy improvements that REDI
provides across all AIDRIN metric categories.

Usage:
    source /opt/venvs/drai/bin/activate
    python3 scripts/run_aidrin_adastra.py [--no-privacy] [--sample N]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# ─── Paths ────────────────────────────────────────────────────────────────────
RAW_PATH  = "/opt/data/adastraMI250/AdastaJobsMI250_15days.parquet"
NPZ_PATH  = "/home/w1b/drai/apps/redi/output/adastra_mi250_raps/raps_jobs.npz"
OUT_JSON  = "/home/w1b/drai/apps/redi/validate_results/aidrin_adastra.json"

EPSILON   = 1.0          # differential-privacy noise budget
TRACE_COLS = {"cpu_trace", "gpu_trace", "ntx_trace", "nrx_trace",
              "scheduled_nodes", "args"}

# ─── Metric configs ────────────────────────────────────────────────────────────
# RAW columns of interest:
#   quasi-identifiers : user_id, group_id     (numeric IDs)
#   sensitive target  : wait_seconds          (queue latency)
#   id column         : job_id
#   categorical       : job_state
#   numerical         : time_limit, num_nodes_req, run_time, wait_seconds
#   y_true (binary)   : completed (1=COMPLETED, 0=other)
#   sensitive attr    : group_id  (project/group)
#
# AI-READY columns of interest:
#   quasi-identifiers : account, name         (project code + hashed job name)
#   sensitive target  : wait_seconds
#   id column         : id
#   categorical       : account, end_state
#   numerical         : nodes_required, time_limit, wait_seconds, priority
#   y_true (binary)   : completed
#   sensitive attr    : account

RAW_METRICS = [
    ("completeness",        {}),
    ("duplicity",           {}),
    ("outliers",            {}),
    ("correlations",        {"columns": ["time_limit", "num_nodes_req",
                                         "run_time", "wait_seconds"]}),
    ("feature_relevance",   {"cat_columns":  ["job_state"],
                             "num_columns":  ["time_limit", "num_nodes_req",
                                               "run_time"],
                             "target_column": "wait_seconds"}),
    ("class_imbalance",     {"target_column": "completed"}),
    ("statistical_rates",   {"y_true_column": "completed",
                             "sensitive_attribute_column": "group_id"}),
    ("representation_rate", {"columns": ["job_state"]}),
    ("k_anonymity",         {"quasi_identifiers": ["user_id", "group_id"]}),
    ("l_diversity",         {"quasi_identifiers": ["user_id", "group_id"],
                             "sensitive_column": "wait_seconds"}),
    ("t_closeness",         {"quasi_identifiers": ["user_id", "group_id"],
                             "sensitive_column": "wait_seconds"}),
    ("entropy_risk",        {"quasi_identifiers": ["user_id", "group_id"]}),
    ("single_attribute_risk",   {"id_column":   "job_id",
                                  "eval_columns": ["user_id", "group_id"]}),
    ("multiple_attribute_risk", {"id_column":   "job_id",
                                  "eval_columns": ["user_id", "group_id"]}),
    ("differential_privacy",    {"columns": ["time_limit", "num_nodes_req",
                                              "run_time"],
                                  "epsilon": EPSILON}),
]

AI_METRICS = [
    ("completeness",        {}),
    ("duplicity",           {}),
    ("outliers",            {}),
    ("correlations",        {"columns": ["nodes_required", "time_limit",
                                         "wait_seconds", "priority"]}),
    ("feature_relevance",   {"cat_columns":  ["account", "end_state"],
                             "num_columns":  ["nodes_required", "time_limit",
                                               "priority"],
                             "target_column": "wait_seconds"}),
    ("class_imbalance",     {"target_column": "completed"}),
    ("statistical_rates",   {"y_true_column": "completed",
                             "sensitive_attribute_column": "account"}),
    ("representation_rate", {"columns": ["account", "end_state"]}),
    ("k_anonymity",         {"quasi_identifiers": ["account", "name"]}),
    ("l_diversity",         {"quasi_identifiers": ["account", "name"],
                             "sensitive_column": "wait_seconds"}),
    ("t_closeness",         {"quasi_identifiers": ["account", "name"],
                             "sensitive_column": "wait_seconds"}),
    ("entropy_risk",        {"quasi_identifiers": ["account", "name"]}),
    ("single_attribute_risk",   {"id_column":   "id",
                                  "eval_columns": ["account", "name"]}),
    ("multiple_attribute_risk", {"id_column":   "id",
                                  "eval_columns": ["account", "name"]}),
    ("differential_privacy",    {"columns": ["nodes_required", "time_limit",
                                              "wait_seconds", "priority"],
                                  "epsilon": EPSILON}),
]


# ─── Data loading ──────────────────────────────────────────────────────────────

def load_raw(sample: int | None = None) -> pd.DataFrame:
    """Load Adastra MI250 parquet, add derived columns."""
    df = pd.read_parquet(RAW_PATH)
    print(f"  Raw: {len(df):,} rows × {len(df.columns)} cols "
          f"({df.job_id.nunique():,} unique jobs)")

    # Compute wait_seconds
    df["wait_seconds"] = (df["start_time"] - df["submit_time"]).dt.total_seconds()

    # Binary target: 1 = COMPLETED, 0 = everything else
    df["completed"] = (df["job_state"] == "COMPLETED").astype(int)

    # Drop timezone-aware datetime columns
    dt_cols = df.select_dtypes(include=["datetime64[ns, UTC]", "datetimetz"]).columns.tolist()
    df = df.drop(columns=dt_cols)

    # Retain only columns needed for metrics
    keep = {
        "job_id", "user_id", "group_id", "job_state",
        "time_limit", "num_nodes_req", "run_time",
        "wait_seconds", "completed",
    }
    df = df[[c for c in df.columns if c in keep]]

    # Convert numeric quasi-IDs to string so aidrin treats them as categorical
    for col in ["user_id", "group_id"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Fill NaN in quasi-ID columns
    for col in ["user_id", "group_id"]:
        if col in df.columns:
            df[col] = df[col].fillna("<missing>")

    if sample and len(df) > sample:
        df = df.sample(n=sample, random_state=42)
        print(f"  (sampled to {sample:,} rows for speed)")

    return df


def load_ai_ready() -> pd.DataFrame:
    """Load REDI raps_jobs.npz → flat DataFrame, add derived columns."""
    data = np.load(NPZ_PATH, allow_pickle=True)
    jobs = data["jobs"]
    rows = []
    for j in jobs:
        row = {}
        for k, v in j.items():
            if k in TRACE_COLS:
                continue
            if isinstance(v, (list, np.ndarray)):
                continue
            row[k] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    print(f"  AI-ready: {len(df):,} rows × {len(df.columns)} cols")

    # Derived columns
    if "start_time" in df.columns and "submit_time" in df.columns:
        df["wait_seconds"] = df["start_time"] - df["submit_time"]
    else:
        df["wait_seconds"] = float("nan")

    # Binary target
    df["completed"] = (df.get("end_state", "") == "COMPLETED").astype(int)

    # Booleans → int
    for col in df.select_dtypes(include="bool").columns:
        df[col] = df[col].astype(int)

    # Fill NaN in string quasi-ID columns
    for col in ["account", "name"]:
        if col in df.columns:
            df[col] = df[col].fillna("<missing>")

    # Retain only columns referenced in AI_METRICS
    keep = {
        "id", "name", "account", "end_state",
        "nodes_required", "time_limit", "wait_seconds",
        "priority", "completed",
    }
    df = df[[c for c in df.columns if c in keep]]

    return df


# ─── Running aidrin ────────────────────────────────────────────────────────────

def run_metric(metric: str, csv_path: str, extra: dict) -> dict | None:
    """Run a single aidrin metric via `aidrin batch` JSON config."""
    config = {
        "file_path": csv_path,
        "metrics": [metric],
        **extra,
    }
    with tempfile.NamedTemporaryFile(
        suffix=".json", mode="w", delete=False
    ) as cfg_file:
        json.dump(config, cfg_file)
        cfg_path = cfg_file.name

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


def run_all(label: str, df: pd.DataFrame, metrics: list[tuple]) -> dict:
    """Write df to CSV, run every metric, return results dict."""
    with tempfile.NamedTemporaryFile(
        suffix=".csv", mode="w", delete=False
    ) as tmp:
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


# ─── Summarisation ─────────────────────────────────────────────────────────────

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

    # DATA QUALITY
    r = results.get("completeness", {})
    s["completeness"] = _scalar(r, "completeness", "Overall Completeness")

    r = results.get("duplicity", {})
    s["duplicate_ratio"] = _scalar(r, "duplicity",
                                    "Duplicity scores",
                                    "Overall duplicity of the dataset")

    r = results.get("outliers", {})
    per_col = _scalar(r, "outliers", "Outlier scores")
    if isinstance(per_col, dict):
        ratios = [v for v in per_col.values() if isinstance(v, (int, float))]
        s["avg_outlier_ratio"] = round(sum(ratios) / len(ratios), 4) if ratios else "—"
        s["max_outlier_ratio"] = round(max(ratios), 4) if ratios else "—"
    else:
        s["avg_outlier_ratio"] = s["max_outlier_ratio"] = "—"

    # CORRELATION
    r = results.get("feature_relevance", {})
    pearson = r.get("feature_relevance", {}).get("Pearson Correlation to Target", {})
    if pearson:
        best_feat = max(pearson, key=lambda k: abs(pearson[k]) if isinstance(pearson[k], float) else 0)
        s["top_feature"] = best_feat
        s["top_feat_corr"] = round(abs(pearson[best_feat]), 4) if isinstance(pearson[best_feat], float) else "—"
    else:
        s["top_feature"] = s["top_feat_corr"] = "—"

    # FAIRNESS
    r = results.get("class_imbalance", {})
    imb_block = r.get("class_imbalance", {}).get("Imbalance degree", {})
    s["class_imbalance"] = round(float(imb_block["Imbalance Degree score"]), 4) \
        if "Imbalance Degree score" in imb_block else "—"

    r = results.get("representation_rate", {})
    ratios = r.get("representation_rate", {}).get("Probability ratios", {})
    if ratios:
        vals = [v for v in ratios.values() if isinstance(v, (int, float))]
        s["max_repr_ratio"] = round(max(vals), 4) if vals else "—"
        s["min_repr_rate"] = round(min(vals), 4) if vals else "—"
    else:
        s["max_repr_ratio"] = s["min_repr_rate"] = "—"

    # PRIVACY
    for metric in ("k_anonymity", "l_diversity", "t_closeness"):
        r = results.get(metric, {}).get(metric, {})
        score = r.get("k-Value") or r.get("l-Value") or r.get("t-Value")
        if score is None:
            stats = r.get("descriptive_statistics", {})
            score = stats.get("min")
        s[metric] = round(float(score), 4) if score is not None else "—"
        stats = r.get("descriptive_statistics", {})
        s[f"{metric}_mean"] = round(float(stats["mean"]), 4) if "mean" in stats else "—"

    r = results.get("entropy_risk", {}).get("entropy_risk", {})
    stats = r.get("descriptive_statistics", {})
    s["entropy_risk"] = round(float(stats["mean"]), 4) if "mean" in stats else "—"
    s["entropy_risk_min"] = round(float(r.get("Entropy-Value", float("nan"))), 4) \
        if "Entropy-Value" in r else "—"

    r = results.get("single_attribute_risk", {})
    col_stats = r.get("single_attribute_risk", {}).get(
        "Descriptive statistics of the risk scores", {})
    if col_stats:
        means = [v["mean"] for v in col_stats.values()
                 if isinstance(v, dict) and "mean" in v]
        s["single_attribute_risk"] = round(sum(means) / len(means), 4) if means else "—"
    else:
        s["single_attribute_risk"] = "—"

    r = results.get("multiple_attribute_risk", {})
    mar = r.get("multiple_attribute_risk", {})
    score = mar.get("Dataset Risk Score")
    if score is None:
        score = mar.get("Descriptive statistics of the risk scores", {}).get("mean")
    s["multiple_attribute_risk"] = round(float(score), 4) if score is not None else "—"

    return s


# ─── Pretty printing ───────────────────────────────────────────────────────────

METRICS_DISPLAY = [
    ("completeness",           "Completeness",             "Data Quality", "↑"),
    ("duplicate_ratio",        "Duplicate ratio",          "Data Quality", "↓"),
    ("avg_outlier_ratio",      "Avg outlier ratio",        "Data Quality", "↓"),
    ("max_outlier_ratio",      "Max outlier ratio",        "Data Quality", "↓"),
    ("top_feat_corr",          "Top feature correlation",  "Correlation",  "↑"),
    ("class_imbalance",        "Class imbalance degree",   "Fairness",     "↓"),
    ("max_repr_ratio",         "Max representation ratio", "Fairness",     "↓"),
    ("k_anonymity",            "k-anonymity (min k)",      "Privacy",      "↑"),
    ("k_anonymity_mean",       "k-anonymity (mean k)",     "Privacy",      "↑"),
    ("l_diversity",            "l-diversity (min l)",      "Privacy",      "↑"),
    ("t_closeness",            "t-closeness (min t)",      "Privacy",      "↑"),
    ("entropy_risk",           "Entropy risk (mean bits)", "Privacy",      "↑"),
    ("single_attribute_risk",  "Single-attr risk",         "Privacy",      "↓"),
    ("multiple_attribute_risk","Multi-attr risk",          "Privacy",      "↓"),
]


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


def print_table(raw_summ: dict, ai_summ: dict):
    W = 90
    print("\n" + "=" * W)
    print("AIDRIN DATA QUALITY ASSESSMENT: ADASTRA MI250 — RAW vs. REDI AI-READY")
    print("=" * W)
    print(f"  Raw:      Adastra MI250 SLURM snapshot (15 days), {raw_summ.get('_nrows', '?')} jobs")
    print(f"  AI-ready: REDI raps_jobs.npz\n")

    header = f"{'Metric':<28} {'Category':<14} {'Dir':>3}  {'Raw':>10}  {'AI-ready':>10}  {'Δ':>14}"
    print(header)
    print("-" * W)

    cur_cat = None
    for key, label, cat, direction in METRICS_DISPLAY:
        if cat != cur_cat:
            if cur_cat is not None:
                print()
            cur_cat = cat

        r = raw_summ.get(key, "—")
        a = ai_summ.get(key, "—")
        delta = fmt_delta(r, a, direction)
        rs = f"{r:.4f}" if isinstance(r, float) else str(r)
        as_ = f"{a:.4f}" if isinstance(a, float) else str(a)
        print(f"  {label:<26} {cat:<14} {direction:>3}  {rs:>10}  {as_:>10}  {delta:>14}")

    print("=" * W)
    print("  ▲ = improvement after REDI   ▼ = degradation   = unchanged")
    print("  Raw quasi-IDs: [user_id, group_id]   AI-ready quasi-IDs: [account, name]")
    print("  k/l/t: higher min = stronger group privacy guarantee")
    print("  Entropy risk mean bits: higher = larger equivalence classes = harder to isolate a record")


# ─── LaTeX table ──────────────────────────────────────────────────────────────

def write_latex(raw_summ: dict, ai_summ: dict, out_path: str):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{AIDRIN data readiness assessment for Adastra MI250.}",
        r"\label{tab:aidrin_adastra}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrrl}",
        r"\toprule",
        r"\textbf{Category} & \textbf{Metric} & \textbf{Raw} & \textbf{Ready} \\",
        r"\midrule",
    ]

    cur_cat = None
    for key, label, cat, direction in METRICS_DISPLAY:
        r = raw_summ.get(key, "—")
        a = ai_summ.get(key, "—")
        delta = fmt_delta(r, a, direction)
        rs = f"{r:.3f}" if isinstance(r, float) else str(r)
        as_ = f"{a:.3f}" if isinstance(a, float) else str(a)

        if cat != cur_cat:
            if cur_cat is not None:
                lines.append(r"\midrule")
            cur_cat = cat

        try:
            rv, av = float(r), float(a)
            diff = av - rv
            improved = (direction == "↑" and diff > 1e-6) or \
                       (direction == "↓" and diff < -1e-6)
            if improved:
                as_ = r"\textbf{" + as_ + "}"
        except (ValueError, TypeError):
            pass

        lines.append(f"  {cat} & {label} & {rs} & {as_} \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    Path(out_path).write_text("\n".join(lines) + "\n")
    print(f"\nLaTeX table saved → {out_path}")


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None,
                        help="Optional row limit (default: all 15,285 jobs)")
    parser.add_argument("--no-privacy", action="store_true",
                        help="Skip slow privacy metrics")
    args = parser.parse_args()

    skip = set()
    if args.no_privacy:
        skip = {"k_anonymity", "l_diversity", "t_closeness", "entropy_risk",
                "single_attribute_risk", "multiple_attribute_risk"}

    raw_metrics = [(m, e) for m, e in RAW_METRICS if m not in skip]
    ai_metrics  = [(m, e) for m, e in AI_METRICS  if m not in skip]

    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Loading raw Adastra MI250 data ...")
    raw_df = load_raw(sample=args.sample)
    raw_df.attrs["_nrows"] = len(raw_df)

    print("\n[2/4] Loading REDI AI-ready data ...")
    ai_df = load_ai_ready()

    print(f"\n[3/4] Running {len(raw_metrics)} metrics on raw data ...")
    raw_results = run_all("raw", raw_df, raw_metrics)

    print(f"\n[4/4] Running {len(ai_metrics)} metrics on AI-ready data ...")
    ai_results = run_all("ai", ai_df, ai_metrics)

    output = {"raw": raw_results, "ai_ready": ai_results}
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nFull results → {OUT_JSON}")

    raw_summ = summarise(raw_results)
    raw_summ["_nrows"] = len(raw_df)
    ai_summ  = summarise(ai_results)
    print_table(raw_summ, ai_summ)

    latex_path = OUT_JSON.replace(".json", ".tex")
    write_latex(raw_summ, ai_summ, latex_path)


if __name__ == "__main__":
    main()
