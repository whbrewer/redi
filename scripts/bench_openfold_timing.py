#!/usr/bin/env python3
"""
Life sciences timing benchmark: run REDI on 6KWC FASTA and parse per-step timings.

Usage:
    source /opt/venvs/drai/bin/activate
    python3 scripts/bench_openfold_timing.py [--runs N]
"""

import argparse
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

FASTA_PATH = "/opt/drai/openfold/examples/monomer/fasta_dir/6kwc.fasta"
OUT_JSON   = "/home/w1b/drai/apps/redi/validate_results/bench_openfold_timing.json"

STAGE_MAP = {
    "detect_format":       "ingest",
    "load_fasta":          "ingest",
    "load_clinical_csv":   "ingest",
    "anonymize_pii":       "preprocess",
    "filter_sequences":    "preprocess",
    "encode_sequences":    "transform",
    "sequence_features":   "transform",
    "prepare_transformer": "structure",
    "train_val_test_split":"output",
    "save_hdf5":           "output",
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
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"  [run {run_idx+1}] redi run {FASTA_PATH} ...", flush=True)
        t0 = time.time()
        result = subprocess.run(
            ["redi", "run", FASTA_PATH, "--domain", "life_sciences", "-o", tmpdir],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        total_ms = int((time.time() - t0) * 1000)
    step_t = parse_timings(result.stdout)
    stage_t = aggregate_stages(step_t)
    return step_t, stage_t, total_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)

    print(f"\nLife Sciences Timing Benchmark  (6KWC, 191 AA, {args.runs} runs)")
    print()

    all_stages, all_totals = [], []
    for i in range(args.runs):
        _, stage_t, total_ms = run_once(i)
        all_stages.append(stage_t)
        all_totals.append(total_ms)
        print(f"  Run {i+1}: {total_ms/1000:.3f}s  "
              + "  ".join(f"{k}={v}ms" for k, v in stage_t.items() if v > 0))

    avg_stages = {k: int(sum(r[k] for r in all_stages) / len(all_stages))
                  for k in all_stages[0]}
    avg_total = int(sum(all_totals) / len(all_totals))
    n_seq = 1
    throughput = n_seq / (avg_total / 1000)

    result = {
        "case": "openfold_6kwc",
        "domain": "life_sciences",
        "n_sequences": n_seq,
        "seq_length": 191,
        "runs": args.runs,
        "stages_ms": avg_stages,
        "total_ms": avg_total,
        "total_s": round(avg_total / 1000, 3),
        "throughput_seqs_per_s": round(throughput, 3),
        "throughput_label": "seq/s",
    }
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved → {OUT_JSON}")

    W = 55
    print("\n" + "=" * W)
    print("LIFE SCIENCES TIMING  (6KWC, 1 sequence, 191 AA)")
    print("=" * W)
    print(f"  {'Stage':<14}  {'ms':>6}  {'%':>6}")
    print(f"  {'-'*14}  {'-'*6}  {'-'*6}")
    for stage in ["ingest", "preprocess", "transform", "structure", "output"]:
        ms = avg_stages[stage]
        pct = 100 * ms / avg_total if avg_total else 0
        print(f"  {stage:<14}  {ms:>6,}  {pct:>5.1f}%")
    print(f"  {'TOTAL':<14}  {avg_total:>6,}  100.0%")
    print(f"\n  Throughput: {throughput:.3f} seq/s")
    print("=" * W)


if __name__ == "__main__":
    main()
