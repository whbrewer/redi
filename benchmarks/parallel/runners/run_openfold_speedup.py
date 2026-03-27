#!/usr/bin/env python3
"""
OpenFold / OpenProteinSet parallel speedup benchmark.

Available: 131,487 protein directories under open_protein_set/pdb/.
Fixed workload: first N protein directories (default: 128).
Worker sweep:   1, 2, 4, 8, 16, 32.

Unlike the IO benchmark (which calls `redi run` on the full OPS root and uses
internal threading), this runner parallelises at the protein-directory level:
each worker calls `redi run` on a single protein directory.

Usage:
    source ~/venv/redi/bin/activate
    python benchmarks/parallel/runners/run_openfold_speedup.py
    python benchmarks/parallel/runners/run_openfold_speedup.py --workers 8
    python benchmarks/parallel/runners/run_openfold_speedup.py --sweep
    python benchmarks/parallel/runners/run_openfold_speedup.py --n-proteins 256 --sweep
"""

import argparse
import concurrent.futures
import json
import subprocess
import tempfile
import time
from pathlib import Path

OPS_PDB_DIR = "/lustre/orion/gen053/world-shared/openfold/data/open_protein_set/pdb"
OUT_DIR     = Path(__file__).parent.parent / "results"

WORKER_SWEEP    = [1, 2, 4, 8, 16, 32]
DEFAULT_N_PROTEINS = 128


def run_one(args_tuple):
    """Worker function: run redi on a single protein directory."""
    prot_dir, run_idx = args_tuple
    with tempfile.TemporaryDirectory() as tmpdir:
        t0 = time.time()
        result = subprocess.run(
            ["redi", "run", str(prot_dir), "--domain", "life_sciences",
             "-o", tmpdir, "--style", "simple"],
            capture_output=True, text=True,
        )
        elapsed_ms = int((time.time() - t0) * 1000)
    ok = result.returncode == 0
    return str(prot_dir), elapsed_ms, ok


def run_sweep(prot_dirs, workers):
    """Run all protein dirs with the given worker count; return total wall time in ms."""
    tasks = [(str(d), i) for i, d in enumerate(prot_dirs)]
    t0 = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run_one, tasks))
    total_ms = int((time.time() - t0) * 1000)
    n_ok = sum(1 for _, _, ok in results if ok)
    return total_ms, n_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-proteins", type=int, default=DEFAULT_N_PROTEINS,
                        help=f"Number of protein dirs in fixed workload (default: {DEFAULT_N_PROTEINS})")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (omit to run full sweep)")
    parser.add_argument("--sweep", action="store_true",
                        help="Run full worker sweep: 1,2,4,8,16,32")
    parser.add_argument("--out", default=str(OUT_DIR / "speedup_openfold.json"),
                        help="Output JSON path")
    args = parser.parse_args()

    prot_dirs = sorted(Path(OPS_PDB_DIR).iterdir())[: args.n_proteins]
    if not prot_dirs:
        raise SystemExit(f"No protein directories found in {OPS_PDB_DIR}")

    print(f"\nOpenFold Parallel Speedup Benchmark")
    print(f"  Proteins:  {len(prot_dirs):,} dirs")
    print(f"  OPS dir:   {OPS_PDB_DIR}")
    print()

    worker_list = WORKER_SWEEP if (args.sweep or args.workers is None) else [args.workers]
    worker_list = [w for w in worker_list if w <= len(prot_dirs)]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.out)
    existing = {}
    if out_path.exists():
        with open(out_path) as f:
            prev = json.load(f)
        if prev.get("n_proteins") == len(prot_dirs):
            existing = {r["workers"]: r for r in prev.get("runs", [])}

    serial_ms = existing.get(1, {}).get("total_ms")

    for w in worker_list:
        print(f"  workers={w:2d} ...", end="  ", flush=True)
        total_ms, n_ok = run_sweep(prot_dirs, w)
        if w == 1:
            serial_ms = total_ms
        speedup = round(serial_ms / total_ms, 3) if serial_ms else None
        efficiency = round(speedup / w, 3) if speedup else None
        throughput = round(n_ok / (total_ms / 1000), 2) if total_ms > 0 else 0

        run = {
            "workers": w,
            "n_proteins": len(prot_dirs),
            "n_ok": n_ok,
            "total_ms": total_ms,
            "total_s": round(total_ms / 1000, 1),
            "throughput_proteins_per_s": throughput,
            "speedup": speedup,
            "parallel_efficiency": efficiency,
        }
        existing[w] = run
        print(f"{total_ms/1000:.1f}s  speedup={speedup}  ({throughput:.2f} proteins/s)")

    runs = sorted(existing.values(), key=lambda r: r["workers"])
    result = {
        "case": "openfold_ops",
        "domain": "life_sciences",
        "n_proteins": len(prot_dirs),
        "runs": runs,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults → {out_path}")

    W = 58
    print("\n" + "=" * W)
    print(f"OPENFOLD SPEEDUP  ({len(prot_dirs):,} proteins)")
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
