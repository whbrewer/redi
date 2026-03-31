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
    "detect_format":            "ingest",
    "load_open_protein_set":    "ingest",
    "LoadOpenProteinSet":       "ingest",
    "parse_mmcif":              "ingest",
    "ParseMMCIFStructures":     "ingest",
    "parse_msa":                "preprocess",
    "ParseMSAAlignments":       "preprocess",
    "anonymize_pii":            "preprocess",
    "filter_sequences":         "preprocess",
    "streaming_npz_processor":  "preprocess",   # mmCIF→MSA→encode→NPZ per protein
    "StreamingNPZProcessor":    "preprocess",
    "encode_sequences":         "transform",
    "sequence_features":        "transform",
    "prepare_transformer":      "structure",
    "PrepareForTransformer":    "structure",
    "train_val_test_split":     "structure",
    "save_per_protein_npz":     "output",
    "SavePerProteinNPZ":        "output",
    "save_hdf5":                "output",
    "write_checksums":          "output",
    "save_metadata":            "output",
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


def aggregate_stages(step_timings: dict, substep_ms: dict = None) -> dict:
    stages = {"ingest": 0, "preprocess": 0, "transform": 0, "structure": 0, "output": 0}
    for step, ms in step_timings.items():
        stage = STAGE_MAP.get(step)
        if stage:
            stages[stage] += ms

    # streaming_npz_processor bundles preprocess/transform/structure/output work.
    # Redistribute its wall-clock time using substep ratios so all 5 stages appear.
    #   cif_parse + msa_parse → preprocess
    #   encode               → transform
    #   assemble             → structure
    #   save                 → output
    snpz_ms = step_timings.get("streaming_npz_processor", 0)
    if snpz_ms and substep_ms:
        keys = ["cif_parse", "msa_parse", "encode", "assemble", "save"]
        substep_stage = {
            "cif_parse": "preprocess",
            "msa_parse": "preprocess",
            "encode":    "transform",
            "assemble":  "structure",
            "save":      "output",
        }
        total_sub = sum(substep_ms.get(k, 0) for k in keys)
        if total_sub > 0:
            # Remove snpz from preprocess (where it was lumped), then redistribute
            stages["preprocess"] -= snpz_ms
            for k in keys:
                frac = substep_ms.get(k, 0) / total_sub
                stages[substep_stage[k]] += int(snpz_ms * frac)

    return stages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-proteins", type=int, default=None,
                        help="Number of proteins to process (default: all)")
    parser.add_argument("--num-workers", type=int, default=16,
                        help="Parallel workers (default: 16)")
    parser.add_argument("--out", default=OUT_JSON, help="Output JSON path")
    args = parser.parse_args()

    print(f"\nOpenFold / OpenProteinSet Stage Timing Benchmark")
    print(f"  OPS root:  {OPS_ROOT}")
    n_label = f"{args.n_proteins:,}" if args.n_proteins else "all"
    print(f"  Proteins:  {n_label}")
    print(f"  Workers:   {args.num_workers}")
    print(f"  Config:    {CFG_TPL}")
    print()

    with open(CFG_TPL) as f:
        cfg = yaml.safe_load(f)
    if args.n_proteins is not None:
        cfg["max_proteins"] = args.n_proteins
    else:
        cfg.pop("max_proteins", None)  # remove cap — run full dataset
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

    # Parse substep breakdown printed by StreamingNPZProcessor (needed before stage aggregation)
    substep_ms: dict = {}
    avg_residues = avg_msa_depth = None
    substep_re = re.compile(r"(cif_parse|msa_parse|encode|assemble|save)\s*:\s*([\d.]+)\s*ms/protein")
    res_re  = re.compile(r"avg residues\s*:\s*([\d.]+)")
    msa_re  = re.compile(r"avg MSA depth\s*:\s*([\d.]+)")
    for line in result.stdout.splitlines():
        clean = re.sub(r"\x1b\[[0-9;]*m", "", line)
        sm = substep_re.search(clean)
        if sm:
            substep_ms[sm.group(1)] = float(sm.group(2))
        rm = res_re.search(clean)
        if rm:
            avg_residues = float(rm.group(1))
        mm = msa_re.search(clean)
        if mm:
            avg_msa_depth = float(mm.group(1))

    stage_t = aggregate_stages(step_t, substep_ms)

    # Estimate how many proteins were actually processed.
    # Parse from the streaming_npz_processor step detail: "N files"
    n_actual = args.n_proteins or 0
    m = re.search(r"streaming_npz_processor.*?(\d[\d,]*)\s+files", result.stdout)
    if m:
        n_actual = int(m.group(1).replace(",", ""))
    else:
        # Fallback: look for "Saved N/..." in streaming progress output
        m = re.search(r"Saved\s+(\d[\d,]*)/\d", result.stdout)
        if m:
            n_actual = int(m.group(1).replace(",", ""))

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
    if substep_ms:
        result_dict["substep_timings_ms_per_protein"] = substep_ms
    if avg_residues is not None:
        result_dict["avg_residues"] = avg_residues
    if avg_msa_depth is not None:
        result_dict["avg_msa_depth"] = avg_msa_depth

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
    if substep_ms:
        print(f"\n  Preprocess substep breakdown (avg per protein):")
        if avg_residues is not None:
            print(f"    avg residues : {avg_residues:.0f}")
        if avg_msa_depth is not None:
            print(f"    avg MSA depth: {avg_msa_depth:.0f}")
        total_sub = sum(substep_ms.values())
        for k, v in substep_ms.items():
            pct_sub = 100 * v / total_sub if total_sub else 0
            print(f"    {k:<12s}: {v:7.1f} ms/protein  ({pct_sub:.1f}%)")
    print("=" * W)


if __name__ == "__main__":
    main()
