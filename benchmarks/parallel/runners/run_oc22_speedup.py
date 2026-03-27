#!/usr/bin/env python3
"""
OC22 / HydraGNN parallel speedup benchmark.

Available: 41 LMDB shards in the train split (~1.5 GiB each).
Fixed workload: first 32 shards.
Worker sweep:   1, 2, 4, 8, 16, 32.

Each worker calls `redi run` on one LMDB shard independently via the
materials domain LMDB pipeline (LoadOC22LMDB → PreprocessOC22 →
NormalizeOC22 → AssembleOC22Tensors → SaveNPZ).

Usage:
    source ~/venv/redi/bin/activate
    python benchmarks/parallel/runners/run_oc22_speedup.py
    python benchmarks/parallel/runners/run_oc22_speedup.py --workers 8
    python benchmarks/parallel/runners/run_oc22_speedup.py --sweep
    python benchmarks/parallel/runners/run_oc22_speedup.py --n-shards 16 --sweep
"""

import argparse
import concurrent.futures
import json
import subprocess
import tempfile
import time
from pathlib import Path

LMDB_ROOT = ("/lustre/orion/gen053/world-shared/HydraGNN/data/oc22"
             "/s2ef_total_train_val_test_lmdbs/data/oc22/s2ef-total/train")
OUT_DIR   = Path(__file__).parent.parent / "results"

WORKER_SWEEP    = [1, 2, 4, 8, 16, 32]
DEFAULT_N_SHARDS = 32


def run_one(args_tuple):
    """Worker function: run redi on a single LMDB shard."""
    lmdb_path, run_idx = args_tuple
    with tempfile.TemporaryDirectory() as tmpdir:
        t0 = time.time()
        result = subprocess.run(
            ["redi", "run", str(lmdb_path), "--domain", "materials",
             "-o", tmpdir, "--style", "simple"],
            capture_output=True, text=True,
        )
        elapsed_ms = int((time.time() - t0) * 1000)
    ok = result.returncode == 0
    return str(lmdb_path), elapsed_ms, ok


def run_sweep(shards, workers):
    """Run all shards with the given worker count; return total wall time in ms."""
    tasks = [(str(s), i) for i, s in enumerate(shards)]
    t0 = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run_one, tasks))
    total_ms = int((time.time() - t0) * 1000)
    n_ok = sum(1 for _, _, ok in results if ok)
    return total_ms, n_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-shards", type=int, default=DEFAULT_N_SHARDS,
                        help=f"Number of LMDB shards in fixed workload (default: {DEFAULT_N_SHARDS})")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (omit to run full sweep)")
    parser.add_argument("--sweep", action="store_true",
                        help="Run full worker sweep: 1,2,4,8,16,32")
    parser.add_argument("--out", default=str(OUT_DIR / "speedup_oc22.json"),
                        help="Output JSON path")
    args = parser.parse_args()

    shards = sorted(
        p for p in Path(LMDB_ROOT).iterdir()
        if p.suffix == ".lmdb" and not p.name.endswith("-lock")
    )[: args.n_shards]

    if not shards:
        raise SystemExit(f"No .lmdb shards found in {LMDB_ROOT}")

    total_gib = sum(s.stat().st_size for s in shards) / (1024 ** 3)

    print(f"\nOC22 / HydraGNN Parallel Speedup Benchmark")
    print(f"  Shards:   {len(shards)} LMDB shards  ({total_gib:.0f} GiB total)")
    print(f"  LMDB dir: {LMDB_ROOT}")
    print()

    worker_list = WORKER_SWEEP if (args.sweep or args.workers is None) else [args.workers]
    worker_list = [w for w in worker_list if w <= len(shards)]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.out)
    existing = {}
    if out_path.exists():
        with open(out_path) as f:
            prev = json.load(f)
        if prev.get("n_shards") == len(shards):
            existing = {r["workers"]: r for r in prev.get("runs", [])}

    serial_ms = existing.get(1, {}).get("total_ms")

    for w in worker_list:
        print(f"  workers={w:2d} ...", end="  ", flush=True)
        total_ms, n_ok = run_sweep(shards, w)
        if w == 1:
            serial_ms = total_ms
        speedup    = round(serial_ms / total_ms, 3) if serial_ms else None
        efficiency = round(speedup / w, 3) if speedup else None
        throughput = round(n_ok / (total_ms / 1000), 3) if total_ms > 0 else 0

        run = {
            "workers": w,
            "n_shards": len(shards),
            "n_ok": n_ok,
            "total_ms": total_ms,
            "total_s": round(total_ms / 1000, 1),
            "throughput_shards_per_s": throughput,
            "speedup": speedup,
            "parallel_efficiency": efficiency,
        }
        existing[w] = run
        print(f"{total_ms/1000:.1f}s  speedup={speedup}")

    runs = sorted(existing.values(), key=lambda r: r["workers"])
    result = {
        "case": "oc22_hydragnn",
        "domain": "materials",
        "n_shards": len(shards),
        "total_input_gib": round(total_gib, 1),
        "runs": runs,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults → {out_path}")

    W = 58
    print("\n" + "=" * W)
    print(f"OC22 SPEEDUP  ({len(shards)} LMDB shards, {total_gib:.0f} GiB)")
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
