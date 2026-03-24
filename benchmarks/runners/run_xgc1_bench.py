#!/usr/bin/env python3
"""
XGC1 ITER fusion pipeline stage timing benchmark.

Runs REDI on N ADIOS2 BP5 snapshot directories from the n560fr_ITER_PFPO_W_Ne
simulation, parses per-step timings, and writes a JSON summary for the SC26 plot.

Each .bp directory is a single time-step snapshot (~5.5 GiB, 560-node ITER mesh).

Usage:
    source ~/venv/redi/bin/activate
    python benchmarks/runners/run_xgc1_bench.py [--n-files N] [--out FILE]
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
BP_DIR  = ("/lustre/orion/world-shared/fus184/fusiond-seed-xgc1-data"
           "/n560fr_ITER_PFPO_W_Ne")
CFG_TPL = str(Path(__file__).parent.parent / "configs" / "bench_xgc1.yaml")
OUT_JSON = str(Path(__file__).parent.parent / "results" / "bench_xgc1.json")

# ── Step → stage mapping ───────────────────────────────────────────────────────
STAGE_MAP = {
    "detect_format":        "ingest",
    "load_adios":           "ingest",
    "load_fusion_hdf5":     "ingest",
    "align_signals":        "preprocess",
    "handle_shots":         "preprocess",
    "engineer_plasma":      "preprocess",
    "EngineerPlasmaFeatures": "preprocess",
    "normalize_signals":    "transform",
    "NormalizeSignals":     "transform",
    "label_disruptions":    "transform",
    "LabelDisruptions":     "transform",
    "create_sequences":     "structure",
    "CreateSequences":      "structure",
    "fusion_to_tensor":     "structure",
    "FusionToTensor":       "structure",
    "train_val_test_split": "structure",
    "save_npz":             "output",
    "save_hdf5":            "output",
    "write_checksums":      "output",
    "save_metadata":        "output",
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


def run_one_file(bp_path: str, cfg_tpl: dict, run_idx: int) -> tuple:
    """Run REDI on a single .bp directory; return (step_timings, total_ms, size_gib)."""
    bp = Path(bp_path)
    try:
        size_bytes = sum(f.stat().st_size for f in bp.iterdir() if f.is_file())
    except Exception:
        size_bytes = 0
    size_gib = size_bytes / (1024 ** 3)

    cfg = dict(cfg_tpl)
    cfg["input_path"] = str(bp_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg["output_dir"] = tmpdir
        tmp_cfg = f"/tmp/redi_xgc1_bench_{run_idx}.yaml"
        with open(tmp_cfg, "w") as f:
            yaml.dump(cfg, f)

        print(f"  [file {run_idx+1}] {bp.name}  ({size_gib:.1f} GiB)", flush=True)
        t0 = time.time()
        result = subprocess.run(
            ["redi", "run", str(bp_path), "--domain", "fusion", "-c", tmp_cfg],
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
                        help="Number of .bp snapshot dirs to process (default: 5)")
    parser.add_argument("--out", default=OUT_JSON, help="Output JSON path")
    args = parser.parse_args()

    bp_dirs = sorted(Path(BP_DIR).glob("xgc.3d.*.bp"))
    if not bp_dirs:
        raise SystemExit(f"No xgc.3d.*.bp directories found in {BP_DIR}")
    bp_dirs = bp_dirs[: args.n_files]

    print(f"\nXGC1 ITER Stage Timing Benchmark")
    print(f"  BP dir:    {BP_DIR}")
    print(f"  Files:     {len(bp_dirs)} × xgc.3d snapshots")
    for bp in bp_dirs:
        print(f"    {bp.name}")
    print()

    with open(CFG_TPL) as f:
        cfg_tpl = yaml.safe_load(f)

    all_step_t: dict = {}
    total_ms = 0
    total_gib = 0.0

    for i, bp_path in enumerate(bp_dirs):
        step_t, file_ms, size_gib = run_one_file(str(bp_path), cfg_tpl, i)
        total_ms += file_ms
        total_gib += size_gib
        for k, v in step_t.items():
            all_step_t[k] = all_step_t.get(k, 0) + v
        stage_t = aggregate_stages(step_t)
        print(f"    {file_ms/1000:.1f}s  " +
              "  ".join(f"{k}={v/1000:.1f}s" for k, v in stage_t.items() if v > 0))

    stage_totals = aggregate_stages(all_step_t)

    # Throughput: data bytes read per second
    throughput_gib_s = total_gib / (total_ms / 1000) if total_ms > 0 else 0

    result = {
        "case": "xgc1_iter",
        "domain": "fusion",
        "n_files": len(bp_dirs),
        "files": [bp.name for bp in bp_dirs],
        "total_input_gib": round(total_gib, 2),
        "stages_ms": stage_totals,
        "total_ms": total_ms,
        "total_s": round(total_ms / 1000, 1),
        "throughput_gib_per_s": round(throughput_gib_s, 3),
        "throughput_label": "GiB/s",
        "step_timings_ms": all_step_t,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults → {args.out}")

    W = 65
    print("\n" + "=" * W)
    print(f"XGC1 TIMING SUMMARY  ({len(bp_dirs)} ITER snapshots, {total_gib:.1f} GiB)")
    print("=" * W)
    print(f"  {'Stage':<16}  {'s':>8}  {'%':>6}")
    print(f"  {'-'*16}  {'-'*8}  {'-'*6}")
    for stage in ["ingest", "preprocess", "transform", "structure", "output"]:
        ms = stage_totals[stage]
        pct = 100 * ms / total_ms if total_ms else 0
        print(f"  {stage:<16}  {ms/1000:>8.1f}  {pct:>5.1f}%")
    print(f"  {'TOTAL':<16}  {total_ms/1000:>8.1f}  100.0%")
    print(f"\n  Throughput: {throughput_gib_s:.3f} GiB/s")
    print("=" * W)


if __name__ == "__main__":
    main()
