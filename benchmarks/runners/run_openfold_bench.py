#!/usr/bin/env python3
"""
OpenFold / OpenProteinSet pipeline stage timing benchmark.

Runs REDI on the OPS root directory (5 000 proteins by default),
parses per-step timings, and writes a JSON summary for the SC26 paper plot.

Usage:
    source ~/venv/redi/bin/activate
    python benchmarks/runners/run_openfold_bench.py [--n-proteins N] [--out FILE]
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
OPS_ROOT = "/lustre/orion/gen053/world-shared/openfold/data"
CFG_TPL  = str(Path(__file__).parent.parent / "configs" / "bench_openfold.yaml")
OUT_JSON = str(Path(__file__).parent.parent / "results" / "bench_openfold.json")

# ── Step → stage mapping ───────────────────────────────────────────────────────
# The OpenProteinSet pipeline runs through:
#   LoadOpenProteinSet / ParseMMCIFStructures  → ingest
#   ParseMSAAlignments                         → preprocess
#   encode_sequences / sequence_features       → transform
#   prepare_transformer / train_val_test_split → structure
#   SavePerProteinNPZ / write_checksums / save_metadata → output
STAGE_MAP = {
    "detect_format":          "ingest",
    "load_open_protein_set":  "ingest",
    "LoadOpenProteinSet":     "ingest",
    "parse_mmcif":            "ingest",
    "ParseMMCIFStructures":   "ingest",
    "parse_msa":              "preprocess",
    "ParseMSAAlignments":     "preprocess",
    "anonymize_pii":          "preprocess",
    "filter_sequences":       "preprocess",
    "encode_sequences":       "transform",
    "sequence_features":      "transform",
    "prepare_transformer":    "structure",
    "PrepareForTransformer":  "structure",
    "train_val_test_split":   "structure",
    "save_per_protein_npz":   "output",
    "SavePerProteinNPZ":      "output",
    "save_hdf5":              "output",
    "write_checksums":        "output",
    "save_metadata":          "output",
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-proteins", type=int, default=5000,
                        help="Number of proteins to process (default: 5000)")
    parser.add_argument("--num-workers", type=int, default=16,
                        help="Parallel workers (default: 16)")
    parser.add_argument("--out", default=OUT_JSON, help="Output JSON path")
    args = parser.parse_args()

    print(f"\nOpenFold / OpenProteinSet Stage Timing Benchmark")
    print(f"  OPS root:  {OPS_ROOT}")
    print(f"  Proteins:  {args.n_proteins:,}")
    print(f"  Workers:   {args.num_workers}")
    print(f"  Config:    {CFG_TPL}")
    print()

    with open(CFG_TPL) as f:
        cfg = yaml.safe_load(f)
    cfg["max_proteins"] = args.n_proteins
    cfg["num_workers"] = args.num_workers

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg["output_dir"] = tmpdir
        tmp_cfg = "/tmp/redi_openfold_bench.yaml"
        with open(tmp_cfg, "w") as f:
            yaml.dump(cfg, f)

        print(f"  Running: redi run {OPS_ROOT} -c {tmp_cfg}", flush=True)
        t0 = time.time()
        result = subprocess.run(
            ["redi", "run", OPS_ROOT, "--domain", "life_sciences", "-c", tmp_cfg],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        total_ms = int((time.time() - t0) * 1000)
        Path(tmp_cfg).unlink(missing_ok=True)

    if result.returncode != 0:
        print(f"  WARNING: redi exited {result.returncode}")
        print(result.stdout[-3000:])

    step_t = parse_timings(result.stdout)
    stage_t = aggregate_stages(step_t)

    # Estimate how many proteins were actually processed from stdout
    n_actual = args.n_proteins
    m = re.search(r"(\d+)\s+proteins?", result.stdout, re.IGNORECASE)
    if m:
        n_actual = int(m.group(1))

    throughput = n_actual / (total_ms / 1000) if total_ms > 0 else 0

    result_dict = {
        "case": "openfold_ops",
        "domain": "life_sciences",
        "n_proteins": n_actual,
        "n_workers": args.num_workers,
        "stages_ms": stage_t,
        "total_ms": total_ms,
        "total_s": round(total_ms / 1000, 1),
        "throughput_proteins_per_s": round(throughput, 3),
        "throughput_label": "proteins/s",
        "step_timings_ms": step_t,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result_dict, f, indent=2)
    print(f"\nResults → {args.out}")

    W = 65
    print("\n" + "=" * W)
    print(f"OPENFOLD TIMING SUMMARY  ({n_actual:,} proteins, {args.num_workers} workers)")
    print("=" * W)
    print(f"  {'Stage':<16}  {'s':>8}  {'%':>6}")
    print(f"  {'-'*16}  {'-'*8}  {'-'*6}")
    for stage in ["ingest", "preprocess", "transform", "structure", "output"]:
        ms = stage_t[stage]
        pct = 100 * ms / total_ms if total_ms else 0
        print(f"  {stage:<16}  {ms/1000:>8.1f}  {pct:>5.1f}%")
    print(f"  {'TOTAL':<16}  {total_ms/1000:>8.1f}  100.0%")
    print(f"\n  Throughput: {throughput:.3f} proteins/s")
    print("=" * W)


if __name__ == "__main__":
    main()
