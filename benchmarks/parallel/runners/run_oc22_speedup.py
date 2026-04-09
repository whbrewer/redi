#!/usr/bin/env python3
"""
OC22 / HydraGNN parallel speedup benchmark.

Parallelises at the LMDB shard level: each worker runs `redi run <shard>`
independently via the redi-hydragnn plugin.  Per-shard work is capped at
max_samples structures so the 1-worker baseline is feasible.

Fixed workload: 32 shards × 5,000 structures each = 160,000 total.
Worker sweep:   1, 2, 4, 8, 16, 32.

Usage:
    source ~/venv/redi/bin/activate
    python benchmarks/parallel/runners/run_oc22_speedup.py
    python benchmarks/parallel/runners/run_oc22_speedup.py --workers 8
    python benchmarks/parallel/runners/run_oc22_speedup.py --sweep
    python benchmarks/parallel/runners/run_oc22_speedup.py --n-shards 32 --max-samples-per-shard 5000 --sweep
"""

import argparse
import concurrent.futures
import json
import subprocess
import tempfile
import time
from pathlib import Path

import yaml

LMDB_ROOT = ("/lustre/orion/gen053/world-shared/HydraGNN/data/oc22"
             "/s2ef_total_train_val_test_lmdbs/data/oc22/s2ef-total/train")
OUT_DIR   = Path(__file__).parent.parent / "results"

WORKER_SWEEP               = [1, 2, 4, 8, 16, 32]
DEFAULT_N_SHARDS           = 32
DEFAULT_MAX_SAMPLES_PER_SHARD = 5_000


def run_one(args_tuple):
    """Worker: run redi on a single LMDB shard with a cap on structures."""
    shard_path, max_samples, worker_idx = args_tuple

    cfg = {
        "domain":      "hydragnn",
        "max_samples": max_samples,
        "num_workers": 1,       # parallelism is at shard level, not within shard
        "output_format": "npz",
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg["output_dir"] = tmpdir
        tmp_cfg = f"/tmp/redi_oc22_shard_{worker_idx}.yaml"
        with open(tmp_cfg, "w") as f:
            yaml.dump(cfg, f)

        t0 = time.time()
        result = subprocess.run(
            ["redi", "run", str(shard_path), "--domain", "hydragnn",
             "-c", tmp_cfg, "--style", "simple"],
            capture_output=True, text=True,
        )
        elapsed_ms = int((time.time() - t0) * 1000)
        Path(tmp_cfg).unlink(missing_ok=True)

    ok = result.returncode == 0
    return str(shard_path), elapsed_ms, ok


def run_sweep(shards, max_samples_per_shard, workers):
    """Run all shards with the given worker count; return (wall_ms, n_ok)."""
    tasks = [(s, max_samples_per_shard, i) for i, s in enumerate(shards)]
    t0 = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run_one, tasks))
    wall_ms = int((time.time() - t0) * 1000)
    n_ok = sum(1 for _, _, ok in results if ok)
    return wall_ms, n_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-shards", type=int, default=DEFAULT_N_SHARDS,
                        help=f"Number of LMDB shards (default: {DEFAULT_N_SHARDS})")
    parser.add_argument("--max-samples-per-shard", type=int,
                        default=DEFAULT_MAX_SAMPLES_PER_SHARD,
                        help=f"Structure cap per shard (default: {DEFAULT_MAX_SAMPLES_PER_SHARD:,})")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (omit to run full sweep)")
    parser.add_argument("--sweep", action="store_true",
                        help="Run full worker sweep: 1,2,4,8,16,32")
    parser.add_argument("--data-dir", default=LMDB_ROOT,
                        help="OC22 LMDB train split directory (default: OLCF Lustre path)")
    parser.add_argument("--out", default=str(OUT_DIR / "speedup_oc22.json"),
                        help="Output JSON path")
    args = parser.parse_args()

    shards = sorted(
        p for p in Path(args.data_dir).iterdir()
        if p.suffix == ".lmdb" and not p.name.endswith("-lock")
    )[: args.n_shards]

    if not shards:
        raise SystemExit(f"No .lmdb shards found in {args.data_dir}")

    total_structures = len(shards) * args.max_samples_per_shard

    print(f"\nOC22 / HydraGNN Parallel Speedup Benchmark")
    print(f"  Shards:      {len(shards)}  ×  {args.max_samples_per_shard:,} structures each")
    print(f"  Total work:  {total_structures:,} structures")
    print(f"  LMDB root:   {args.data_dir}")
    print()

    worker_list = WORKER_SWEEP if (args.sweep or args.workers is None) else [args.workers]
    worker_list = [w for w in worker_list if w <= len(shards)]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)

    # Load existing results only if same workload
    existing = {}
    if out_path.exists():
        with open(out_path) as f:
            prev = json.load(f)
        if (prev.get("n_shards") == len(shards) and
                prev.get("max_samples_per_shard") == args.max_samples_per_shard):
            existing = {r["workers"]: r for r in prev.get("runs", [])}

    serial_ms = existing.get(1, {}).get("total_ms")

    for w in worker_list:
        print(f"  workers={w:2d} ...", end="  ", flush=True)
        wall_ms, n_ok = run_sweep(shards, args.max_samples_per_shard, w)

        if w == 1:
            serial_ms = wall_ms
        speedup    = round(serial_ms / wall_ms, 3) if serial_ms else None
        efficiency = round(speedup / w, 3)          if speedup  else None
        throughput = round(n_ok * args.max_samples_per_shard / (wall_ms / 1000), 1) if wall_ms > 0 else 0

        run = {
            "workers":                  w,
            "n_shards":                 len(shards),
            "max_samples_per_shard":    args.max_samples_per_shard,
            "n_ok":                     n_ok,
            "total_s":                  round(wall_ms / 1000, 1),
            "total_ms":                 wall_ms,
            "throughput_structs_per_s": throughput,
            "speedup":                  speedup,
            "parallel_efficiency":      efficiency,
        }
        existing[w] = run
        print(f"{wall_ms/1000:.1f}s  speedup={speedup}  ({throughput:.0f} structs/s)")

    runs = sorted(existing.values(), key=lambda r: r["workers"])
    result = {
        "case":                 "oc22_hydragnn",
        "domain":               "hydragnn",
        "n_shards":             len(shards),
        "max_samples_per_shard": args.max_samples_per_shard,
        "runs":                 runs,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults → {out_path}")

    W = 58
    print("\n" + "=" * W)
    print(f"OC22 SPEEDUP  ({len(shards)} shards × {args.max_samples_per_shard:,} structs)")
    print("=" * W)
    print(f"  {'Workers':>7}  {'Time (s)':>8}  {'Speedup':>8}  {'Efficiency':>10}")
    print(f"  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*10}")
    for r in runs:
        sp  = f"{r['speedup']:.2f}"            if r["speedup"]            else "—"
        eff = f"{r['parallel_efficiency']:.2f}" if r["parallel_efficiency"] else "—"
        print(f"  {r['workers']:>7}  {r['total_s']:>8.1f}  {sp:>8}  {eff:>10}")
    print("=" * W)


if __name__ == "__main__":
    main()
