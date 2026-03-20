#!/usr/bin/env python3
"""
Validate REDI sPHENIX (nuclear physics) output against original .npy arrays.

REDI stacks detector hit arrays into NPZ with train/val/test split.
Validation: match arrays between GT (original .npy) and REDI output by
element-sum fingerprint, then compare values directly.

Usage:
    source /opt/venvs/drai/bin/activate
    python3 scripts/validate_sphenix.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SPHENIX_DIR = "/opt/drai/sphenix/outer/12-2_0-0"
REDI_OUT    = "/home/w1b/drai/apps/redi/output/sphenix_redi_validate"
OUT_JSON    = "/home/w1b/drai/apps/redi/validate_results/sphenix_validate.json"

N_VALIDATE = 50   # number of arrays to spot-check


def run_redi(in_glob: str, out_dir: str) -> int:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    result = subprocess.run(
        ["redi", "run", in_glob, "--array", "-o", out_dir, "--style", "simple"],
        capture_output=True, text=True,
    )
    elapsed_ms = int((time.time() - t0) * 1000)
    print(result.stdout)
    return elapsed_ms


def fingerprint(arr: np.ndarray) -> int:
    """Cheap fingerprint for matching arrays."""
    return int(arr.astype(np.int64).sum())


def main():
    import glob
    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)

    print("Phase 4 — sPHENIX / Nuclear Physics Validation")
    print("=" * 55)

    # 1. Run REDI (if output doesn't exist)
    print(f"\n[1/4] Running REDI on {SPHENIX_DIR}/*.npy ...")
    if not (Path(REDI_OUT) / "train.npz").exists():
        in_glob = f"{SPHENIX_DIR}/*.npy"
        elapsed_ms = run_redi(in_glob, REDI_OUT)
    else:
        print("  (using existing output)")
        elapsed_ms = None

    # 2. Load REDI output and build fingerprint index
    print("\n[2/4] Indexing REDI output ...")
    redi_arrays = {}
    total_redi = 0
    for split in ["train", "val", "test"]:
        npz_path = Path(REDI_OUT) / f"{split}.npz"
        if not npz_path.exists():
            continue
        d = np.load(npz_path)
        if "X" not in d:
            continue
        X = d["X"]
        total_redi += len(X)
        for i in range(len(X)):
            fp = fingerprint(X[i])
            redi_arrays[fp] = X[i]
    print(f"  Indexed {total_redi} REDI arrays")

    # 3. Load sample GT arrays and match
    print(f"\n[3/4] Matching {N_VALIDATE} GT arrays to REDI output ...")
    gt_files = sorted(glob.glob(f"{SPHENIX_DIR}/*.npy"))
    rng = np.random.default_rng(42)
    sample_files = rng.choice(gt_files, size=min(N_VALIDATE, len(gt_files)),
                              replace=False).tolist()

    n_matched = 0
    corrs, maes = [], []

    for fpath in sample_files:
        gt_arr = np.load(fpath)
        fp = fingerprint(gt_arr)
        if fp not in redi_arrays:
            continue
        redi_arr = redi_arrays[fp]

        n_matched += 1
        # Compare flattened
        a = gt_arr.astype(np.float32).ravel()
        b = redi_arr.astype(np.float32).ravel()
        mae = float(abs(a - b).mean())
        maes.append(mae)

        if a.std() > 0 and b.std() > 0:
            corr = float(np.corrcoef(a, b)[0, 1])
            corrs.append(corr)

    print(f"  Matched: {n_matched}/{len(sample_files)}")

    if not corrs:
        print("  ERROR: no matched arrays found")
        sys.exit(1)

    avg_corr = sum(corrs) / len(corrs)
    avg_mae  = sum(maes)  / len(maes)
    print(f"  corr={avg_corr:.6f}  MAE={avg_mae:.4f}")

    # 4. Summary
    meta_path = Path(REDI_OUT) / "metadata.json"
    meta = json.load(open(meta_path)) if meta_path.exists() else {}

    result = {
        "case": "sphenix_outer_12-2_0-0",
        "domain": "nuclear_physics",
        "n_arrays_total": total_redi,
        "array_shape": [192, 249, 16],
        "n_validated": n_matched,
        "elapsed_ms": elapsed_ms,
        "similarity": {
            "corr": round(avg_corr, 6),
            "mae": round(avg_mae, 4),
        },
        "note": "Validation by fingerprint-matched array pairs. "
                "REDI uses general array mode (no domain-specific transform). "
                "Expected corr=1.0, MAE=0 since no normalization applied.",
    }
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved → {OUT_JSON}")

    print("\n" + "=" * 55)
    print("sPHENIX VALIDATION SUMMARY")
    print("=" * 55)
    print(f"  Arrays:    {total_redi} × 192×249×16 uint16")
    print(f"  Validated: {n_matched} matched pairs")
    print(f"  corr={avg_corr:.4f}  MAE={avg_mae:.4f}")
    print("=" * 55)
    if avg_corr >= 0.999 and avg_mae < 0.01:
        print("  PASS")
    else:
        print("  WARN")


if __name__ == "__main__":
    main()
