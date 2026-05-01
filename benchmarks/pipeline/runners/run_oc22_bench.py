#!/usr/bin/env python3
"""
OC22 / HydraGNN pipeline stage timing benchmark.

Runs REDI on the OC22 LMDB train split (or a capped subset via --max-samples),
parses per-step timings from stdout, and writes a JSON summary for the SC26
paper plot.

Usage:
    source ~/venv/redi/bin/activate
    python benchmarks/pipeline/runners/run_oc22_bench.py [--max-samples N] [--out FILE]

    --max-samples   Cap number of structures (default: all ~8.2M)
    --data-dir      OC22 LMDB root (default: OLCF Lustre path)
    --out           Output JSON path
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
LMDB_ROOT = ("/lustre/orion/gen053/world-shared/HydraGNN/data/oc22"
             "/s2ef_total_train_val_test_lmdbs/data/oc22/s2ef-total")
CFG_TPL   = str(Path(__file__).parent.parent / "configs" / "bench_oc22.yaml")
OUT_JSON  = str(Path(__file__).parent.parent / "results" / "bench_oc22.json")

# ── Step → stage mapping ───────────────────────────────────────────────────────
STAGE_MAP = {
    "detect_format":        "ingest",
    "load_oc22_lmdb":       "ingest",
    "preprocess_oc22":      "preprocess",
    "normalize_oc22":       "transform",
    "assemble_oc22":        "structure",
    "train_val_test_split": "structure",
    "save_npz":             "output",
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


def count_structures_from_output(stdout: str) -> int:
    """Extract n_structures from REDI verbose output."""
    for line in stdout.splitlines():
        clean = re.sub(r"\x1b\[[0-9;]*m", "", line)
        m = re.search(r"assembled\s+([\d,]+)\s+structures", clean)
        if m:
            return int(m.group(1).replace(",", ""))
        m = re.search(r"n_structures['\"]?\s*[:=]\s*([\d,]+)", clean)
        if m:
            return int(m.group(1).replace(",", ""))
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap number of structures (default: all ~8.2M)")
    parser.add_argument("--data-dir", default=LMDB_ROOT,
                        help="OC22 LMDB root directory (default: OLCF Lustre path)")
    parser.add_argument("--out", default=OUT_JSON, help="Output JSON path")
    args = parser.parse_args()

    split_dir = str(Path(args.data_dir) / "train")
    n_shards = len([p for p in Path(split_dir).iterdir()
                    if p.suffix == ".lmdb" and not p.name.endswith("-lock")])
    if n_shards == 0:
        raise SystemExit(f"No .lmdb shards found in {split_dir}")

    sample_label = f"{args.max_samples:,}" if args.max_samples else "all"

    print(f"\nOC22 / HydraGNN Stage Timing Benchmark")
    print(f"  LMDB root: {args.data_dir}")
    print(f"  Split:     train  ({n_shards} shards)")
    print(f"  Samples:   {sample_label}")
    print(f"  Config:    {CFG_TPL}")
    print()

    with open(CFG_TPL) as f:
        cfg_tpl = yaml.safe_load(f)

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = dict(cfg_tpl)
        if args.max_samples is not None:
            cfg["max_samples"] = args.max_samples

        tmp_cfg = f"/tmp/redi_oc22_bench.yaml"
        with open(tmp_cfg, "w") as f:
            yaml.dump(cfg, f)

        print(f"  Running: redi run {split_dir} --domain materials -c {tmp_cfg}", flush=True)
        t0 = time.time()
        result = subprocess.run(
            ["redi", "run", split_dir, "--domain", "materials", "-c", tmp_cfg,
             "-o", tmpdir, "-v"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        total_ms = int((time.time() - t0) * 1000)
        Path(tmp_cfg).unlink(missing_ok=True)

    if result.returncode != 0:
        print(f"  WARNING: redi exited {result.returncode}")
        print(result.stdout[-3000:])

    step_timings = parse_timings(result.stdout)
    stage_totals = aggregate_stages(step_timings)
    n_valid = count_structures_from_output(result.stdout)
    throughput = n_valid / (total_ms / 1000) if total_ms > 0 else 0

    result_dict = {
        "case": "oc22_hydragnn",
        "domain": "materials",
        "split": "train",
        "n_shards": n_shards,
        "n_valid": n_valid,
        "max_samples": args.max_samples,
        "stages_ms": stage_totals,
        "total_ms": total_ms,
        "total_s": round(total_ms / 1000, 1),
        "throughput_structs_per_s": round(throughput, 2),
        "throughput_label": "structs/s",
        "step_timings_ms": step_timings,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result_dict, f, indent=2)
    print(f"\nResults → {args.out}")

    STAGES = ["ingest", "preprocess", "transform", "structure", "output"]
    W = 65
    print("\n" + "=" * W)
    print(f"OC22 TIMING SUMMARY  ({n_valid:,} structures, split=train)")
    print("=" * W)
    print(f"  {'Stage':<16}  {'s':>8}  {'%':>6}")
    print(f"  {'-'*16}  {'-'*8}  {'-'*6}")
    for stage in STAGES:
        ms = stage_totals[stage]
        pct = 100 * ms / total_ms if total_ms else 0
        print(f"  {stage:<16}  {ms/1000:>8.1f}  {pct:>5.1f}%")
    print(f"  {'TOTAL':<16}  {total_ms/1000:>8.1f}  100.0%")
    print(f"\n  Throughput: {throughput:.2f} structs/s")
    print("=" * W)


if __name__ == "__main__":
    main()
