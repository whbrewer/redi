#!/usr/bin/env python3
"""
ClimaX / CMCC parallel speedup benchmark.

Available: 165 CMCC annual temperature NC files.
Fixed workload: first 32 files.
Worker sweep:   1, 2, 4, 8, 16, 32.

Each worker calls `redi run` on one NC file independently.
Speedup is measured as T_serial / T_N.

Usage:
    source ~/venv/redi/bin/activate
    python benchmarks/parallel/runners/run_climax_speedup.py
    python benchmarks/parallel/runners/run_climax_speedup.py --workers 8
    python benchmarks/parallel/runners/run_climax_speedup.py --sweep
    python benchmarks/parallel/runners/run_climax_speedup.py --n-files 16 --sweep
"""

import argparse
import concurrent.futures
import json
import subprocess
import tempfile
import time
from pathlib import Path

import yaml

NC_DIR  = "/lustre/orion/stf053/world-shared/vga/projects/ClimaX/data/CMCC/raw/temperature"
CFG_TPL = str(Path(__file__).parent.parent.parent / "io" / "configs" / "bench_climax.yaml")
OUT_DIR = Path(__file__).parent.parent / "results"

WORKER_SWEEP    = [1, 2, 4, 8, 16, 32]
DEFAULT_N_FILES = 32


def run_one(args_tuple):
    """Worker function: run redi on a single NC file. Returns (path, total_ms)."""
    nc_path, cfg_tpl, run_idx = args_tuple
    cfg = dict(cfg_tpl)
    cfg["input_path"] = str(nc_path)
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg["output_dir"] = tmpdir
        tmp_cfg = f"/tmp/redi_climax_speedup_{run_idx}.yaml"
        with open(tmp_cfg, "w") as f:
            yaml.dump(cfg, f)
        t0 = time.time()
        result = subprocess.run(
            ["redi", "run", "-c", tmp_cfg],
            capture_output=True, text=True,
        )
        elapsed_ms = int((time.time() - t0) * 1000)
        Path(tmp_cfg).unlink(missing_ok=True)
    ok = result.returncode == 0
    return str(nc_path), elapsed_ms, ok


def run_sweep(nc_files, cfg_tpl, workers):
    """Run all files with the given worker count; return total wall time in ms."""
    tasks = [(str(f), cfg_tpl, i) for i, f in enumerate(nc_files)]
    t0 = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run_one, tasks))
    total_ms = int((time.time() - t0) * 1000)
    n_ok = sum(1 for _, _, ok in results if ok)
    return total_ms, n_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (omit to run full sweep)")
    parser.add_argument("--n-files", type=int, default=DEFAULT_N_FILES,
                        help=f"Number of NC files in fixed workload (default: {DEFAULT_N_FILES})")
    parser.add_argument("--sweep", action="store_true",
                        help="Run full worker sweep: 1,2,4,8,16,32")
    parser.add_argument("--out", default=str(OUT_DIR / "speedup_climax.json"),
                        help="Output JSON path")
    args = parser.parse_args()

    nc_files = sorted(Path(NC_DIR).glob("*.nc"))[: args.n_files]
    if not nc_files:
        raise SystemExit(f"No .nc files found in {NC_DIR}")

    with open(CFG_TPL) as f:
        cfg_tpl = yaml.safe_load(f)

    print(f"\nClimaX / CMCC Parallel Speedup Benchmark")
    print(f"  Files:   {len(nc_files)} NC files (of 165 available)")
    print(f"  NC dir:  {NC_DIR}")
    print()

    worker_list = WORKER_SWEEP if (args.sweep or args.workers is None) else [args.workers]
    worker_list = [w for w in worker_list if w <= len(nc_files)]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing results if present (for incremental runs)
    out_path = Path(args.out)
    existing = {}
    if out_path.exists():
        with open(out_path) as f:
            prev = json.load(f)
        if prev.get("n_files") == len(nc_files):
            existing = {r["workers"]: r for r in prev.get("runs", [])}

    serial_ms = existing.get(1, {}).get("total_ms")
    runs = list(existing.values())

    for w in worker_list:
        print(f"  workers={w:2d} ...", end="  ", flush=True)
        total_ms, n_ok = run_sweep(nc_files, cfg_tpl, w)
        if w == 1:
            serial_ms = total_ms
        speedup = round(serial_ms / total_ms, 3) if serial_ms else None
        efficiency = round(speedup / w, 3) if speedup else None

        run = {
            "workers": w,
            "n_files": len(nc_files),
            "n_ok": n_ok,
            "total_ms": total_ms,
            "total_s": round(total_ms / 1000, 1),
            "speedup": speedup,
            "parallel_efficiency": efficiency,
        }
        existing[w] = run
        runs = sorted(existing.values(), key=lambda r: r["workers"])
        print(f"{total_ms/1000:.1f}s  speedup={speedup}")

    result = {
        "case": "climax_cmcc",
        "domain": "climate",
        "n_files": len(nc_files),
        "runs": runs,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults → {out_path}")

    W = 58
    print("\n" + "=" * W)
    print(f"CLIMAX/CMCC SPEEDUP  ({len(nc_files)} files)")
    print("=" * W)
    print(f"  {'Workers':>7}  {'Time (s)':>8}  {'Speedup':>8}  {'Efficiency':>10}")
    print(f"  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*10}")
    for r in runs:
        sp  = f"{r['speedup']:.2f}"  if r['speedup']  else "—"
        eff = f"{r['parallel_efficiency']:.2f}" if r['parallel_efficiency'] else "—"
        print(f"  {r['workers']:>7}  {r['total_s']:>8.1f}  {sp:>8}  {eff:>10}")
    print("=" * W)


if __name__ == "__main__":
    main()
