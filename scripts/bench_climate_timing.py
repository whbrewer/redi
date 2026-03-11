#!/usr/bin/env python3
"""
Climate timing benchmark: run REDI on one TaiESM1 2m_temperature chunk,
parse per-step timings from stdout, and output a JSON summary.

Per-step timings are parsed from the REDI run output lines like:
  ✅ step_name (1234ms)

Stage mapping (steps → REDI stages):
  Ingest:    load_climate
  Preprocess: regrid_climate
  Transform:  normalize_climate, select_variables, compute_anomalies
  Structure:  climate_to_numpy, train_val_test_split
  Output:     save_climate

Usage:
    source /opt/venvs/drai/bin/activate
    python3 scripts/bench_climate_timing.py [--chunk YSTR] [--runs N]
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

RAW_DIR  = "/opt/drai/datasets/CMIP6/TaiESM1/raw/2m_temperature"
OUT_JSON = "/home/w1b/drai/apps/redi/validate_results/bench_climate_timing.json"
BASE_CFG = "/home/w1b/drai/apps/redi/examples/climate_taiesm1_validate.yaml"

DEFAULT_CHUNK = "185001010000-186001010000"

STAGE_MAP = {
    "load_climate":        "ingest",
    "regrid_climate":      "preprocess",
    "normalize_climate":   "transform",
    "select_variables":    "transform",
    "compute_anomalies":   "transform",
    "climate_to_numpy":    "structure",
    "train_val_test_split": "structure",
    "save_climate":        "output",
    "save_metadata":       "output",
}

# Regex: captures step name and ms value from lines like
#   ✅ step_name (1234ms)  or  ⏭️  step_name → skipped ... (0ms)
STEP_RE = re.compile(r"(\w+)\s+.*?\((\d+)ms\)")


def parse_timings(output: str) -> dict[str, int]:
    """Extract step → ms mapping from REDI stdout."""
    timings = {}
    for line in output.splitlines():
        # Strip ANSI escape codes
        clean = re.sub(r"\x1b\[[0-9;]*m", "", line)
        m = STEP_RE.search(clean)
        if m:
            step, ms = m.group(1), int(m.group(2))
            if ms > 0:
                timings[step] = timings.get(step, 0) + ms
    return timings


def aggregate_stages(step_timings: dict[str, int]) -> dict[str, int]:
    """Sum step timings into pipeline stages."""
    stages = {"ingest": 0, "preprocess": 0, "transform": 0, "structure": 0, "output": 0}
    for step, ms in step_timings.items():
        stage = STAGE_MAP.get(step)
        if stage:
            stages[stage] += ms
    return stages


def run_once(chunk: str, run_idx: int) -> tuple[dict, dict, int]:
    """Run REDI on the chunk; return (step_timings, stage_timings, total_ms)."""
    raw_nc = f"{RAW_DIR}/2m_temperature_{chunk}_raw.nc"
    if not Path(raw_nc).exists():
        sys.exit(f"Raw NetCDF not found: {raw_nc}")

    # Write a temp config pointing at this chunk
    import yaml
    with open(BASE_CFG) as f:
        cfg = yaml.safe_load(f)
    cfg["input_path"] = raw_nc

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg["output_dir"] = tmpdir
        cfg_path = f"/tmp/redi_climate_bench_{run_idx}.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f)

        print(f"  [run {run_idx+1}] redi run -c {cfg_path} ...", flush=True)
        t0 = time.time()
        result = subprocess.run(
            ["redi", "run", "-c", cfg_path],
            capture_output=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        total_ms = int((time.time() - t0) * 1000)
        Path(cfg_path).unlink(missing_ok=True)

    if result.returncode != 0:
        print(f"  WARNING: redi exited with code {result.returncode}")

    step_timings = parse_timings(result.stdout)
    stage_timings = aggregate_stages(step_timings)
    return step_timings, stage_timings, total_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", default=DEFAULT_CHUNK,
                        help=f"Year-string chunk (default: {DEFAULT_CHUNK})")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of timing runs to average (default: 1)")
    args = parser.parse_args()

    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)

    raw_nc = f"{RAW_DIR}/2m_temperature_{args.chunk}_raw.nc"
    size_bytes = Path(raw_nc).stat().st_size if Path(raw_nc).exists() else 0
    size_gib = size_bytes / (1024**3)

    print(f"\nClimate Timing Benchmark")
    print(f"  Chunk:   {args.chunk}")
    print(f"  Input:   {raw_nc}  ({size_gib:.2f} GiB)")
    print(f"  Runs:    {args.runs}")
    print()

    all_stages = []
    all_totals = []

    for i in range(args.runs):
        step_t, stage_t, total_ms = run_once(args.chunk, i)
        all_stages.append(stage_t)
        all_totals.append(total_ms)
        print(f"  Run {i+1}: total={total_ms/1000:.1f}s  "
              + "  ".join(f"{k}={v}ms" for k, v in stage_t.items() if v > 0))

    # Average across runs
    avg_stages = {
        k: int(sum(r[k] for r in all_stages) / len(all_stages))
        for k in all_stages[0]
    }
    avg_total = int(sum(all_totals) / len(all_totals))

    # Throughput: timesteps / total_s  (14601 timesteps per 10-year chunk)
    timesteps = 14601
    grid_points = timesteps * 128 * 256  # ClimaX grid after regrid
    throughput_pts_s = grid_points / (avg_total / 1000)

    result = {
        "case": "climax_taiesm1_2m_temp",
        "domain": "climate",
        "chunk": args.chunk,
        "input_size_gib": round(size_gib, 3),
        "input_shape": "14601×192×288",
        "output_shape": "14601×128×256",
        "runs": args.runs,
        "stages_ms": avg_stages,
        "total_ms": avg_total,
        "total_s": round(avg_total / 1000, 3),
        "throughput_grid_pts_per_s": int(throughput_pts_s),
        "throughput_label": "grid-pts/s",
    }

    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved → {OUT_JSON}")

    # Print summary table
    W = 70
    print("\n" + "=" * W)
    print("CLIMATE TIMING SUMMARY  (TaiESM1 2m_temperature, one 10-year chunk)")
    print("=" * W)
    print(f"  Input:  {raw_nc}")
    print(f"  Size:   {size_gib:.2f} GiB   Shape: 14601×192×288")
    print(f"  Output: 14601×128×256  (ClimaX grid, normalized)\n")
    headers = ["Stage", "ms", "%"]
    print(f"  {'Stage':<14}  {'ms':>8}  {'%':>6}")
    print(f"  {'-'*14}  {'-'*8}  {'-'*6}")
    for stage in ["ingest", "preprocess", "transform", "structure", "output"]:
        ms = avg_stages[stage]
        pct = 100 * ms / avg_total if avg_total else 0
        print(f"  {stage:<14}  {ms:>8,}  {pct:>5.1f}%")
    print(f"  {'TOTAL':<14}  {avg_total:>8,}  {'100.0%':>6}")
    print(f"\n  Throughput: {throughput_pts_s:,.0f} grid-pts/s")
    print("=" * W)


if __name__ == "__main__":
    main()
