#!/usr/bin/env python3
"""
Materials timing benchmark: run REDI on AFLOW raw CSV and parse per-step timings.

Usage:
    source /opt/venvs/drai/bin/activate
    python3 scripts/bench_materials_timing.py [--runs N]
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

AFLOW_GT = "/opt/drai/aflow/aflow_ml_ready.csv"
RAW_CSV  = "/tmp/aflow_raw_timing.csv"
OUT_JSON = "/home/w1b/drai/apps/redi/validate_results/bench_aflow_timing.json"

STAGE_MAP = {
    "detect_format":         "ingest",
    "load_materials_csv":    "ingest",
    "load_structure":        "ingest",
    "clean_materials":       "preprocess",
    "featurize_compositions":"transform",
    "normalize_materials":   "transform",
    "structure_to_graph":    "structure",
    "materials_to_numpy":    "structure",
    "train_val_test_split":  "output",
    "save_npz":              "output",
    "write_checksums":       "output",
    "save_metadata":         "output",
}

STEP_RE = re.compile(r"(\w+)\s+.*?\((\d+)ms\)")


def prepare_raw_csv():
    import pandas as pd
    df = pd.read_csv(AFLOW_GT)
    raw_cols = ["auid", "compound", "band_gap", "formation_enthalpy", "density", "cif"]
    df[raw_cols].to_csv(RAW_CSV, index=False)


def parse_timings(output: str) -> dict[str, int]:
    timings = {}
    for line in output.splitlines():
        clean = re.sub(r"\x1b\[[0-9;]*m", "", line)
        m = STEP_RE.search(clean)
        if m:
            step, ms = m.group(1), int(m.group(2))
            timings[step] = timings.get(step, 0) + ms
    return timings


def aggregate_stages(step_timings: dict[str, int]) -> dict[str, int]:
    stages = {"ingest": 0, "preprocess": 0, "transform": 0, "structure": 0, "output": 0}
    for step, ms in step_timings.items():
        stage = STAGE_MAP.get(step)
        if stage:
            stages[stage] += ms
    return stages


def run_once(run_idx: int):
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"  [run {run_idx+1}] redi run {RAW_CSV} ...", flush=True)
        t0 = time.time()
        result = subprocess.run(
            ["redi", "run", RAW_CSV, "--domain", "materials",
             "-o", tmpdir],
            capture_output=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        total_ms = int((time.time() - t0) * 1000)

    step_t = parse_timings(result.stdout)
    stage_t = aggregate_stages(step_t)
    return step_t, stage_t, total_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of timing runs to average (default: 3)")
    args = parser.parse_args()

    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    prepare_raw_csv()

    import pandas as pd
    df_raw = pd.read_csv(RAW_CSV)
    n_structures = len(df_raw)

    print(f"\nMaterials Timing Benchmark  ({n_structures} structures, {args.runs} runs)")
    print()

    all_stages, all_totals = [], []
    for i in range(args.runs):
        _, stage_t, total_ms = run_once(i)
        all_stages.append(stage_t)
        all_totals.append(total_ms)
        print(f"  Run {i+1}: {total_ms/1000:.3f}s  "
              + "  ".join(f"{k}={v}ms" for k, v in stage_t.items() if v > 0))

    avg_stages = {
        k: int(sum(r[k] for r in all_stages) / len(all_stages))
        for k in all_stages[0]
    }
    avg_total = int(sum(all_totals) / len(all_totals))
    throughput = n_structures / (avg_total / 1000)

    result = {
        "case": "hydragnn_aflow",
        "domain": "materials",
        "n_structures": n_structures,
        "n_features": 135,
        "runs": args.runs,
        "stages_ms": avg_stages,
        "total_ms": avg_total,
        "total_s": round(avg_total / 1000, 3),
        "throughput_structs_per_s": round(throughput, 2),
        "throughput_label": "structs/s",
    }
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved → {OUT_JSON}")

    W = 60
    print("\n" + "=" * W)
    print("MATERIALS TIMING SUMMARY")
    print("=" * W)
    print(f"  {'Stage':<14}  {'ms':>6}  {'%':>6}")
    print(f"  {'-'*14}  {'-'*6}  {'-'*6}")
    for stage in ["ingest", "preprocess", "transform", "structure", "output"]:
        ms = avg_stages[stage]
        pct = 100 * ms / avg_total if avg_total else 0
        print(f"  {stage:<14}  {ms:>6,}  {pct:>5.1f}%")
    print(f"  {'TOTAL':<14}  {avg_total:>6,}  {'100.0%':>6}")
    print(f"\n  Throughput: {throughput:.2f} structs/s")
    print("=" * W)


if __name__ == "__main__":
    main()
