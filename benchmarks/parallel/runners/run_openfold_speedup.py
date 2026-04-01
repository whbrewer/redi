#!/usr/bin/env python3
"""
OpenFold / OpenProteinSet parallel speedup benchmark.

Measures wall-clock time for processing N proteins while varying the number
of internal workers passed to `redi run --workers W`.  This matches the same
code path used in the pipeline stage-timing benchmark (Fig. 3) and avoids
the per-protein subprocess overhead of the old per-directory approach.

Fixed workload: 4096 proteins (default).
Worker sweep:   1, 2, 4, 8, 16, 32.

Usage:
    source ~/venv/redi/bin/activate
    python benchmarks/parallel/runners/run_openfold_speedup.py
    python benchmarks/parallel/runners/run_openfold_speedup.py --workers 8
    python benchmarks/parallel/runners/run_openfold_speedup.py --sweep
    python benchmarks/parallel/runners/run_openfold_speedup.py --n-proteins 4096 --sweep
"""

import argparse
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

import yaml

OPS_ROOT = "/lustre/orion/gen053/world-shared/openfold/data"
CFG_TPL  = str(Path(__file__).parent.parent.parent / "pipeline" / "configs" / "bench_openfold.yaml")
OUT_DIR  = Path(__file__).parent.parent / "results"

WORKER_SWEEP       = [1, 2, 4, 8, 16, 32]
DEFAULT_N_PROTEINS = 4096


def run_redi(n_proteins: int, num_workers: int) -> tuple[float, int]:
    """
    Run `redi run OPS_ROOT --workers num_workers --max-proteins n_proteins`.
    Returns (wall_time_s, n_processed).
    """
    with open(CFG_TPL) as f:
        cfg = yaml.safe_load(f)

    cfg["max_proteins"] = n_proteins
    cfg["num_workers"]  = num_workers
    cfg.pop("provenance", None)   # skip provenance overhead for benchmark

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg["output_dir"] = tmpdir
        tmp_cfg = f"/tmp/redi_openfold_speedup_w{num_workers}.yaml"
        with open(tmp_cfg, "w") as f:
            yaml.dump(cfg, f)

        t0 = time.time()
        result = subprocess.run(
            ["redi", "run", OPS_ROOT, "--domain", "life_sciences",
             "-c", tmp_cfg, "--style", "simple"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        wall_s = time.time() - t0
        Path(tmp_cfg).unlink(missing_ok=True)

    if result.returncode != 0:
        print(f"    WARNING: redi exited {result.returncode}")
        print(result.stdout[-2000:])

    # Parse number of proteins actually processed
    n_actual = n_proteins
    m = re.search(r"Saved\s+(\d[\d,]*)/\d", result.stdout)
    if m:
        n_actual = int(m.group(1).replace(",", ""))
    else:
        m = re.search(r"streaming_npz_processor.*?(\d[\d,]*)\s+files", result.stdout)
        if m:
            n_actual = int(m.group(1).replace(",", ""))

    return wall_s, n_actual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-proteins", type=int, default=DEFAULT_N_PROTEINS,
                        help=f"Fixed protein workload (default: {DEFAULT_N_PROTEINS})")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of workers (omit to run full sweep)")
    parser.add_argument("--worker-list", default=None,
                        help="Comma-separated list of worker counts to run (e.g. 2,4,8,16,32)")
    parser.add_argument("--sweep", action="store_true",
                        help="Run full worker sweep: 1,2,4,8,16,32")
    parser.add_argument("--out", default=str(OUT_DIR / "speedup_openfold.json"),
                        help="Output JSON path")
    args = parser.parse_args()

    print(f"\nOpenFold Parallel Speedup Benchmark")
    print(f"  Proteins:  {args.n_proteins:,}")
    print(f"  OPS root:  {OPS_ROOT}")
    print()

    if args.worker_list:
        worker_list = [int(w) for w in args.worker_list.split(",")]
    elif args.sweep or args.workers is None:
        worker_list = WORKER_SWEEP
    else:
        worker_list = [args.workers]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)

    # Load existing results (only reuse if same n_proteins)
    existing = {}
    if out_path.exists():
        with open(out_path) as f:
            prev = json.load(f)
        if prev.get("n_proteins") == args.n_proteins:
            existing = {r["workers"]: r for r in prev.get("runs", [])}

    serial_s = existing.get(1, {}).get("total_s")

    for w in worker_list:
        print(f"  workers={w:2d} ...", end="  ", flush=True)
        wall_s, n_actual = run_redi(args.n_proteins, w)

        if w == 1:
            serial_s = wall_s
        speedup    = round(serial_s / wall_s, 3) if serial_s else None
        efficiency = round(speedup / w, 3)        if speedup  else None
        throughput = round(n_actual / wall_s, 2)  if wall_s > 0 else 0

        run = {
            "workers":                  w,
            "n_proteins":               n_actual,
            "total_s":                  round(wall_s, 1),
            "total_ms":                 int(wall_s * 1000),
            "throughput_proteins_per_s": throughput,
            "speedup":                  speedup,
            "parallel_efficiency":      efficiency,
        }
        existing[w] = run
        print(f"{wall_s:.1f}s  speedup={speedup}  ({throughput:.2f} proteins/s)")

        # Incremental save after each worker so results survive walltime kills
        _runs = sorted(existing.values(), key=lambda r: r["workers"])
        with open(out_path, "w") as f:
            json.dump({"case": "openfold_ops", "domain": "life_sciences",
                       "n_proteins": args.n_proteins, "runs": _runs}, f, indent=2)

    runs = sorted(existing.values(), key=lambda r: r["workers"])
    result = {
        "case":       "openfold_ops",
        "domain":     "life_sciences",
        "n_proteins": args.n_proteins,
        "runs":       runs,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults → {out_path}")

    W = 58
    print("\n" + "=" * W)
    print(f"OPENFOLD SPEEDUP  ({args.n_proteins:,} proteins)")
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
