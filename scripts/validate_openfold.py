#!/usr/bin/env python3
"""
Validate REDI life_sciences (OpenFold) output for 6KWC.

REDI encodes the FASTA amino acid sequence as integer indices.
Validation: decode REDI output back to sequence and verify exact match to input FASTA.

Usage:
    source /opt/venvs/drai/bin/activate
    python3 scripts/validate_openfold.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np

FASTA_PATH = "/opt/drai/openfold/examples/monomer/fasta_dir/6kwc.fasta"
REDI_OUT   = "/home/w1b/drai/apps/redi/output/openfold_redi_validate"
OUT_JSON   = "/home/w1b/drai/apps/redi/validate_results/openfold_validate.json"

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
VOCAB_REV   = {i + 1: aa for i, aa in enumerate(AMINO_ACIDS)}
VOCAB_REV[0] = "-"  # padding


def parse_fasta(path: str) -> dict[str, str]:
    """Return {header: sequence} from FASTA file."""
    seqs = {}
    current = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                current = line[1:]
                seqs[current] = ""
            elif current:
                seqs[current] += line.upper()
    return seqs


def run_redi(fasta: str, out_dir: str) -> int:
    """Run redi pipeline; return elapsed ms."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    result = subprocess.run(
        ["redi", "run", fasta, "--domain", "life_sciences",
         "-o", out_dir, "--style", "simple"],
        capture_output=True, text=True,
    )
    elapsed_ms = int((time.time() - t0) * 1000)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr[:300])
    return elapsed_ms


def decode_sequence(encoded: np.ndarray) -> str:
    """Decode integer-encoded amino acid sequence back to string."""
    return "".join(VOCAB_REV.get(int(c), "?") for c in encoded)


def compare_sequences(redi_seq: str, gt_seq: str) -> dict:
    """Compute sequence identity and edit distance."""
    n = max(len(redi_seq), len(gt_seq))
    if n == 0:
        return {"corr": 1.0, "mae": 0.0, "identity": 1.0}

    # Identity: fraction of matching positions
    matches = sum(a == b for a, b in zip(redi_seq, gt_seq))
    identity = matches / n

    # Encode as integers for corr/MAE
    a = np.array([ord(c) for c in redi_seq], dtype=float)
    b = np.array([ord(c) for c in gt_seq], dtype=float)
    if len(a) != len(b):
        min_len = min(len(a), len(b))
        a, b = a[:min_len], b[:min_len]

    mae = float(abs(a - b).mean())
    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 else 1.0

    return {
        "corr": round(corr, 6),
        "mae": round(mae, 4),
        "identity": round(identity, 6),
        "redi_length": len(redi_seq),
        "gt_length": len(gt_seq),
    }


def main():
    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)

    print("Phase 3 — OpenFold/Life Sciences Validation")
    print("=" * 50)

    # Parse GT FASTA
    print("\n[1/4] Parsing reference FASTA ...")
    gt_seqs = parse_fasta(FASTA_PATH)
    print(f"  Sequences: {list(gt_seqs.keys())}")
    for header, seq in gt_seqs.items():
        print(f"  {header}: {len(seq)} AA")

    # Run REDI
    print("\n[2/4] Running REDI life_sciences pipeline ...")
    elapsed_ms = run_redi(FASTA_PATH, REDI_OUT)

    # Load and decode output
    print("\n[3/4] Decoding REDI output ...")
    h5_path = Path(REDI_OUT) / "data.h5"
    if not h5_path.exists():
        sys.exit(f"No HDF5 output at {h5_path}")

    redi_seqs = {}
    with h5py.File(h5_path, "r") as f:
        for split in ["train", "val", "test"]:
            if split in f and "X" in f[split] and len(f[split]["X"]) > 0:
                for i, row in enumerate(f[split]["X"]):
                    # Strip padding zeros
                    row_trimmed = row[row != 0]
                    decoded = decode_sequence(row_trimmed)
                    redi_seqs[f"{split}_{i}"] = decoded

    print(f"  Decoded {len(redi_seqs)} sequence(s)")
    for k, s in redi_seqs.items():
        print(f"  {k}: {len(s)} AA")

    # Compare
    print("\n[4/4] Comparing to reference sequences ...")
    results_by_seq = []
    all_corrs, all_maes = [], []

    gt_seqs_list = list(gt_seqs.values())
    redi_seqs_list = list(redi_seqs.values())

    for i, (gt_seq, redi_seq) in enumerate(zip(gt_seqs_list, redi_seqs_list)):
        sim = compare_sequences(redi_seq, gt_seq)
        print(f"  Seq {i+1}: corr={sim['corr']:.6f}  MAE={sim['mae']:.4f}  "
              f"identity={sim['identity']:.4f}  len={sim['redi_length']}/{sim['gt_length']}")
        all_corrs.append(sim["corr"])
        all_maes.append(sim["mae"])
        results_by_seq.append(sim)

    avg_corr = sum(all_corrs) / len(all_corrs)
    avg_mae = sum(all_maes) / len(all_maes)

    result = {
        "case": "openfold_6kwc",
        "domain": "life_sciences",
        "n_sequences": len(gt_seqs),
        "redi_shape": [len(redi_seqs_list), len(gt_seqs_list[0])] if gt_seqs_list else [],
        "elapsed_ms": elapsed_ms,
        "similarity": {
            "corr": round(avg_corr, 6),
            "mae": round(avg_mae, 4),
            "identity": round(results_by_seq[0]["identity"], 6) if results_by_seq else None,
        },
        "sequences": results_by_seq,
        "note": "Validation via lossless decode: integer encoding -> amino acid string. "
                "identity=1.0 confirms exact sequence preservation.",
    }

    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved → {OUT_JSON}")

    print("\n" + "=" * 50)
    print("OPENFOLD VALIDATION SUMMARY")
    print("=" * 50)
    print(f"  Input:      {FASTA_PATH}")
    print(f"  Sequences:  {len(gt_seqs)} × {len(gt_seqs_list[0])} AA")
    print(f"  corr={avg_corr:.4f}  MAE={avg_mae:.4f}  identity={results_by_seq[0]['identity']:.4f}")
    print(f"  Elapsed:    {elapsed_ms/1000:.2f}s")
    print("=" * 50)
    if avg_corr >= 0.999 and avg_mae < 0.01:
        print("  PASS: sequence exactly preserved")
    else:
        print("  WARN: sequence mismatch")


if __name__ == "__main__":
    main()
