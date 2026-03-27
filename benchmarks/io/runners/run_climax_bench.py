#!/usr/bin/env python3
"""
ClimaX / TaiESM1 pipeline stage timing benchmark.

Runs REDI on N NetCDF chunks from the Lustre TaiESM1 dataset, parses per-step
timings from stdout, and writes a JSON summary for the SC26 paper plot.

Usage:
    source ~/venv/redi/bin/activate
    python benchmarks/runners/run_climax_bench.py [--n-files N] [--out FILE]
"""

import argparse
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

import yaml

# ── Data locations ─────────────────────────────────────────────────────────────
NC_DIR  = "/lustre/orion/stf053/world-shared/vga/projects/ClimaX/data/TaiESM1/raw/2m_temperature"
CFG_TPL = str(Path(__file__).parent.parent / "configs" / "bench_climax.yaml")
OUT_JSON = str(Path(__file__).parent.parent / "results" / "bench_climax.json")

# ── Step → stage mapping ───────────────────────────────────────────────────────
STAGE_MAP = {
    "load_climate":         "ingest",
    "detect_format":        "ingest",
    "regrid_climate":       "preprocess",
    "normalize_climate":    "transform",
    "select_variables":     "transform",
    "compute_anomalies":    "transform",
    "climate_to_numpy":     "structure",
    "train_val_test_split": "structure",
    "save_climate":         "output",
    "save_metadata":        "output",
    "write_checksums":      "output",
}

STEP_RE = re.compile(r"(\w+)\s+.*?\((\d+)ms\)")


def parse_timings(output: str) -> dict:
    timings = {}
    for line in output.splitlines():
        clean = re.sub(r"\x1b\[[0-9;]*m", "", line)
        m = STEP_RE.search(clean)
        if m:
            step, ms = m.group(1), int(m.group(2))
            if ms > 0:
                timings[step] = timings.get(step, 0) + ms
    return timings


def aggregate_stages(step_timings: dict) -> dict:
    stages = {"ingest": 0, "preprocess": 0, "transform": 0, "structure": 0, "output": 0}
    for step, ms in step_timings.items():
        stage = STAGE_MAP.get(step)
        if stage:
            stages[stage] += ms
    return stages


def run_one_file(nc_path: str, cfg_tpl: dict, run_idx: int) -> tuple:
    """Run REDI on a single NC file; return (step_timings, total_ms, file_size_gib)."""
    size_bytes = Path(nc_path).stat().st_size if Path(nc_path).exists() else 0
    size_gib = size_bytes / (1024 ** 3)

    cfg = dict(cfg_tpl)
    cfg["input_path"] = nc_path

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg["output_dir"] = tmpdir
        tmp_cfg = f"/tmp/redi_climax_bench_{run_idx}.yaml"
        with open(tmp_cfg, "w") as f:
            yaml.dump(cfg, f)

        print(f"  [file {run_idx+1}] {Path(nc_path).name}  ({size_gib:.1f} GiB)", flush=True)
        t0 = time.time()
        result = subprocess.run(
            ["redi", "run", "-c", tmp_cfg],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        total_ms = int((time.time() - t0) * 1000)
        Path(tmp_cfg).unlink(missing_ok=True)

    if result.returncode != 0:
        print(f"    WARNING: redi exited {result.returncode}")
        print(result.stdout[-2000:])

    step_t = parse_timings(result.stdout)
    return step_t, total_ms, size_gib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-files", type=int, default=5,
                        help="Number of 10-year NC chunks to process (default: 5)")
    parser.add_argument("--out", default=OUT_JSON, help="Output JSON path")
    args = parser.parse_args()

    nc_files = sorted(Path(NC_DIR).glob("*.nc"))
    if not nc_files:
        raise SystemExit(f"No .nc files found in {NC_DIR}")
    nc_files = nc_files[: args.n_files]

    print(f"\nClimaX/TaiESM1 Stage Timing Benchmark")
    print(f"  NC dir:    {NC_DIR}")
    print(f"  Files:     {len(nc_files)} × 10-year 2m_temperature chunks")
    print(f"  Config:    {CFG_TPL}")
    print()

    with open(CFG_TPL) as f:
        cfg_tpl = yaml.safe_load(f)

    all_step_t: dict = {}
    total_ms = 0
    total_gib = 0.0

    for i, nc_path in enumerate(nc_files):
        step_t, file_ms, size_gib = run_one_file(str(nc_path), cfg_tpl, i)
        total_ms += file_ms
        total_gib += size_gib
        for k, v in step_t.items():
            all_step_t[k] = all_step_t.get(k, 0) + v
        stage_t = aggregate_stages(step_t)
        print(f"    {file_ms/1000:.1f}s  " +
              "  ".join(f"{k}={v/1000:.1f}s" for k, v in stage_t.items() if v > 0))

    stage_totals = aggregate_stages(all_step_t)

    # Throughput: grid-points processed (128×256 ClimaX grid, 14601 steps per file)
    timesteps_per_file = 14601
    grid_pts_per_file = timesteps_per_file * 128 * 256
    total_grid_pts = grid_pts_per_file * len(nc_files)
    throughput = total_grid_pts / (total_ms / 1000) if total_ms > 0 else 0

    result = {
        "case": "climax_taiesm1",
        "domain": "climate",
        "n_files": len(nc_files),
        "total_input_gib": round(total_gib, 2),
        "input_shape_per_file": f"14601×192×288",
        "output_shape_per_file": "14601×128×256",
        "stages_ms": stage_totals,
        "total_ms": total_ms,
        "total_s": round(total_ms / 1000, 1),
        "throughput_grid_pts_per_s": int(throughput),
        "throughput_label": "grid-pts/s",
        "step_timings_ms": all_step_t,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults → {args.out}")

    W = 70
    print("\n" + "=" * W)
    print(f"CLIMAX TIMING SUMMARY  ({len(nc_files)} TaiESM1 2m_temp files)")
    print("=" * W)
    print(f"  Input:  {total_gib:.1f} GiB total")
    print(f"  {'Stage':<14}  {'s':>8}  {'%':>6}")
    print(f"  {'-'*14}  {'-'*8}  {'-'*6}")
    for stage in ["ingest", "preprocess", "transform", "structure", "output"]:
        ms = stage_totals[stage]
        pct = 100 * ms / total_ms if total_ms else 0
        print(f"  {stage:<14}  {ms/1000:>8.1f}  {pct:>5.1f}%")
    print(f"  {'TOTAL':<14}  {total_ms/1000:>8.1f}  100.0%")
    print(f"\n  Throughput: {throughput:,.0f} grid-pts/s")
    print("=" * W)


if __name__ == "__main__":
    main()
