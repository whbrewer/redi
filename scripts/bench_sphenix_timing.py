#!/usr/bin/env python3
"""
sPHENIX timing benchmark: run REDI on the 12-2_0-0 subdir and parse step timings.

Usage:
    source /opt/venvs/drai/bin/activate
    python3 scripts/bench_sphenix_timing.py [--runs N]
"""

import argparse
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

SPHENIX_DIR = "/opt/drai/sphenix/outer/12-2_0-0"
OUT_JSON    = "/home/w1b/drai/apps/redi/validate_results/bench_sphenix_timing.json"

STAGE_MAP = {
    "detect_format":       "ingest",
    "load_npy":            "ingest",
    "validate_input":      "preprocess",
    "compute_array_stats": "preprocess",
    "train_val_test_split":"output",
    "save_npz":            "output",
    "write_checksums":     "output",
    "save_metadata":       "output",
}

STEP_RE = re.compile(r"(\w+)\s+.*?\((\d+)ms\)")


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
    in_glob = f"{SPHENIX_DIR}/*.npy"
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"  [run {run_idx+1}] redi run {in_glob} --array ...", flush=True)
        t0 = time.time()
        result = subprocess.run(
            ["redi", "run", in_glob, "--array", "-o", tmpdir],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        total_ms = int((time.time() - t0) * 1000)
    step_t = parse_timings(result.stdout)
    stage_t = aggregate_stages(step_t)
    return step_t, stage_t, total_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()
    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)

    import glob
    n_files = len(glob.glob(f"{SPHENIX_DIR}/*.npy"))
    print(f"\nsPHENIX Timing Benchmark  ({n_files} events, {args.runs} run(s))")
    print()

    all_stages, all_totals = [], []
    for i in range(args.runs):
        _, stage_t, total_ms = run_once(i)
        all_stages.append(stage_t)
        all_totals.append(total_ms)
        print(f"  Run {i+1}: {total_ms/1000:.1f}s  "
              + "  ".join(f"{k}={v}ms" for k, v in stage_t.items() if v > 0))

    avg_stages = {k: int(sum(r[k] for r in all_stages) / len(all_stages))
                  for k in all_stages[0]}
    avg_total = int(sum(all_totals) / len(all_totals))
    throughput = n_files / (avg_total / 1000)

    result = {
        "case": "sphenix_outer_12-2_0-0",
        "domain": "nuclear_physics",
        "n_events": n_files,
        "array_shape": [192, 249, 16],
        "runs": args.runs,
        "stages_ms": avg_stages,
        "total_ms": avg_total,
        "total_s": round(avg_total / 1000, 3),
        "throughput_events_per_s": round(throughput, 1),
        "throughput_label": "events/s",
    }
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved → {OUT_JSON}")

    W = 58
    print("\n" + "=" * W)
    print(f"sPHENIX TIMING  ({n_files} events × 192×249×16)")
    print("=" * W)
    print(f"  {'Stage':<14}  {'ms':>8}  {'%':>6}")
    print(f"  {'-'*14}  {'-'*8}  {'-'*6}")
    for stage in ["ingest", "preprocess", "transform", "structure", "output"]:
        ms = avg_stages[stage]
        pct = 100 * ms / avg_total if avg_total else 0
        print(f"  {stage:<14}  {ms:>8,}  {pct:>5.1f}%")
    print(f"  {'TOTAL':<14}  {avg_total:>8,}  100.0%")
    print(f"\n  Throughput: {throughput:.1f} events/s")
    print("=" * W)


if __name__ == "__main__":
    main()
