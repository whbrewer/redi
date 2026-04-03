#!/usr/bin/env python3
"""
Validate REDI OC22 output against LMDB ground truth.

Compares REDI-extracted energy-per-atom and forces from the NPZ output
against the corresponding OC22 s2ef-total LMDB values.

Results (N=10,000 structures, train shard 0000):
  energy_per_atom:  Pearson r = 1.000  MAE = 1.3e-07 eV/atom  (float32 rounding)
  forces:           Pearson r = 1.000  (per-structure, after inverting z-score)

Usage:
    source ~/venv/redi/bin/activate
    python validation/runners/validate_oc22.py [--n N]
"""

import argparse
import json
from pathlib import Path

import lmdb
import numpy as np
import pickle
from scipy.stats import pearsonr

LMDB_SHARD = (
    "/lustre/orion/gen053/world-shared/HydraGNN/data/oc22"
    "/s2ef_total_train_val_test_lmdbs/data/oc22/s2ef-total"
    "/train/data.0000.lmdb"
)
NPZ_PATH = (
    "/lustre/orion/scratch/whbrewer/stf053/redi_oc22_bench_output"
    "/oc22_redi_output.npz"
)
OUT_JSON = str(Path(__file__).parent.parent / "results" / "validate_oc22.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10_000,
                        help="Number of structures to compare (default: 10000)")
    parser.add_argument("--lmdb", default=LMDB_SHARD, help="Path to LMDB shard")
    parser.add_argument("--npz",  default=NPZ_PATH,   help="Path to REDI NPZ output")
    parser.add_argument("--out",  default=OUT_JSON,   help="Output JSON path")
    args = parser.parse_args()

    N = args.n
    print(f"\nOC22 fidelity validation  (N={N:,} structures)")
    print(f"  LMDB: {args.lmdb}")
    print(f"  NPZ:  {args.npz}\n")

    # ── Load NPZ targets ──────────────────────────────────────────────────────
    npz        = np.load(args.npz, allow_pickle=True)
    npz_energy = npz["energy_per_atom"][:N].astype(np.float64)
    npz_si     = npz["split_index"]       # cumulative atom offsets per structure

    # ── Read LMDB ground truth ────────────────────────────────────────────────
    # OC22 LMDBs were written with an older PyG version; access via __dict__.
    env = lmdb.open(args.lmdb, subdir=False, readonly=True, lock=False,
                    meminit=False, map_size=int(1e12))
    txn = env.begin()
    cur = txn.cursor()
    cur.first()

    gt_energy    = []
    force_r_list = []   # per-structure Pearson r after de-normalisation

    for i in range(N):
        d     = pickle.loads(cur.value()).__dict__
        y     = float(d["y"])
        na    = int(d["natoms"])
        f_raw = np.array(d["force"], dtype=np.float64)

        gt_energy.append(y / na)

        # Recover REDI force slice and undo per-structure z-score
        start = int(npz_si[i])
        end   = int(npz_si[i + 1]) if i + 1 < len(npz_si) else None
        f_redi = npz["forces"][start:end].astype(np.float64)
        f_reconstructed = f_redi * (f_raw.std() + 1e-8) + f_raw.mean()
        r_f, _ = pearsonr(f_raw.ravel(), f_reconstructed.ravel())
        force_r_list.append(r_f)

        if not cur.next():
            break

    txn.abort()
    env.close()

    # ── Energy comparison ─────────────────────────────────────────────────────
    gt_e   = np.array(gt_energy)
    r_e, _ = pearsonr(gt_e, npz_energy)
    mae_e  = float(np.mean(np.abs(gt_e - npz_energy)))
    max_e  = float(np.max(np.abs(gt_e - npz_energy)))

    # ── Force comparison (per-structure r after de-normalisation) ─────────────
    force_r  = np.array(force_r_list)
    mean_r_f = float(force_r.mean())
    pct_exact = float(np.mean(force_r > 0.9999) * 100)

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"  energy_per_atom:")
    print(f"    Pearson r = {r_e:.10f}")
    print(f"    MAE       = {mae_e:.4e} eV/atom")
    print(f"    max err   = {max_e:.4e} eV/atom")
    print(f"  forces (per-structure r after inverting z-score):")
    print(f"    mean r    = {mean_r_f:.10f}")
    print(f"    % with r > 0.9999: {pct_exact:.1f}%")

    result = {
        "n_structures": N,
        "lmdb_shard": args.lmdb,
        "npz_path": args.npz,
        "energy_per_atom": {
            "pearson_r": round(r_e, 10),
            "mae_ev_per_atom": round(mae_e, 10),
            "max_err_ev_per_atom": round(max_e, 10),
        },
        "forces": {
            "mean_pearson_r_after_denorm": round(mean_r_f, 10),
            "pct_structures_r_gt_0p9999": round(pct_exact, 2),
        },
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results → {args.out}")


if __name__ == "__main__":
    main()
