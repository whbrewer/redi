#!/usr/bin/env python3
"""
OpenFold before/after aggregate assessment for SC26 paper.

Samples matched (raw CIF, processed NPZ) protein pairs, runs redi assess on
each, and produces a raw-vs-ready comparison table with mean ± std per metric.
Also computes resolution distribution from chain_data_cache.json.

Usage:
    source ~/venv/redi/bin/activate
    python scripts/openfold_assess_aggregate.py --sample 500
    python scripts/openfold_assess_aggregate.py --sample 500 --output results/openfold_assess.json
"""

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CIF_DIR   = Path("/lustre/orion/gen053/world-shared/openfold/data/pdb_mmcif/mmcif_files")
NPZ_DIR   = Path("/lustre/orion/gen053/world-shared/openfold/redi_output/train")
CACHE     = Path("/lustre/orion/gen053/world-shared/openfold/out/chain_data_cache.json")

# Metrics to collect from raw CIF assess
RAW_SCALAR_METRICS = [
    "atom_completeness", "msa_depth", "msa_files",
    "resolution_angstrom", "r_factor", "residue_count",
    "avg_atoms_per_residue",
]

# Metrics to collect from processed NPZ assess
READY_SCALAR_METRICS = [
    "backbone_completeness", "atom_completeness",
    "msa_depth", "msa_duplicate_ratio", "msa_gap_ratio",
    "msa_unique_sequences", "resolution_angstrom",
    "sequence_length", "aa_entropy_bits", "aa_entropy_pct_uniform",
    "aa_imbalance_ratio",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mean_std(values):
    if not values:
        return None, None
    n = len(values)
    mu = sum(values) / n
    if n < 2:
        return round(mu, 4), None
    var = sum((x - mu) ** 2 for x in values) / (n - 1)
    return round(mu, 4), round(math.sqrt(var), 4)


def fmt(mu, sd):
    if mu is None:
        return "N/A"
    if sd is None:
        return str(mu)
    return f"{mu} ± {sd}"


def collect_scalar(result, keys):
    s = result.get("domain_specific", {})
    out = {}
    for k in keys:
        v = s.get(k)
        if v is not None:
            out[k] = float(v)
    return out


# ---------------------------------------------------------------------------
# Resolution distribution from chain_data_cache.json
# ---------------------------------------------------------------------------

def resolution_distribution(cache_path: Path):
    print(f"Loading {cache_path} ...", flush=True)
    with open(cache_path) as f:
        cache = json.load(f)

    resolutions = [v["resolution"] for v in cache.values()
                   if isinstance(v.get("resolution"), (int, float))]
    if not resolutions:
        return {}

    n = len(resolutions)
    sorted_res = sorted(resolutions)
    mu, sd = mean_std(resolutions)
    median = sorted_res[n // 2]
    pct_lt_2 = sum(1 for r in resolutions if r < 2.0) / n
    pct_lt_2_5 = sum(1 for r in resolutions if r < 2.5) / n
    pct_2_5_3_5 = sum(1 for r in resolutions if 2.5 <= r < 3.5) / n
    pct_gt_3_5 = sum(1 for r in resolutions if r >= 3.5) / n

    print(f"  {n:,} chains with resolution data")
    return {
        "n_chains": n,
        "mean_angstrom": mu,
        "std_angstrom": sd,
        "median_angstrom": round(median, 3),
        "pct_lt_2.0A": round(pct_lt_2 * 100, 1),
        "pct_lt_2.5A": round(pct_lt_2_5 * 100, 1),
        "pct_2.5_to_3.5A": round(pct_2_5_3_5 * 100, 1),
        "pct_gt_3.5A": round(pct_gt_3_5 * 100, 1),
    }


# ---------------------------------------------------------------------------
# Main assessment loop
# ---------------------------------------------------------------------------

def find_matched_pairs(cif_dir: Path, npz_dir: Path):
    """Return list of (cif_path, npz_path) for proteins present in both."""
    cif_stems = {p.stem.lower(): p for p in cif_dir.glob("*.cif")}
    # NPZ stem format: <pdbid>_<chain>.npz  → pdb_id is first 4 chars
    pairs = []
    for npz in npz_dir.glob("*.npz"):
        pdb_id = npz.stem.split("_")[0].lower()
        if pdb_id in cif_stems:
            pairs.append((cif_stems[pdb_id], npz))
    return pairs


def run_aggregate(sample: int, seed: int, verbose: bool):
    # Import here so script can be run from repo root
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from redi.assess import run_assess

    print(f"\nFinding matched CIF / NPZ pairs ...", flush=True)
    pairs = find_matched_pairs(CIF_DIR, NPZ_DIR)
    print(f"  {len(pairs):,} matched pairs found", flush=True)

    random.seed(seed)
    sample_pairs = random.sample(pairs, min(sample, len(pairs)))
    print(f"  Sampling {len(sample_pairs)} pairs (seed={seed})\n", flush=True)

    raw_scalars  = defaultdict(list)
    ready_scalars = defaultdict(list)
    method_counts = Counter()
    errors = {"raw": 0, "ready": 0}

    for i, (cif_path, npz_path) in enumerate(sample_pairs):
        if verbose or (i % 50 == 0):
            print(f"  [{i+1}/{len(sample_pairs)}] {npz_path.stem}", flush=True)

        # --- Raw CIF ---
        try:
            result = run_assess(str(cif_path), domain="life_sciences")
            s = result.get("domain_specific", {})
            for k in RAW_SCALAR_METRICS:
                v = s.get(k)
                if v is not None:
                    raw_scalars[k].append(float(v))
            method = s.get("experimental_method")
            if method:
                method_counts[method] += 1
        except Exception as e:
            errors["raw"] += 1
            if verbose:
                print(f"    RAW error: {e}", flush=True)

        # --- Ready NPZ ---
        try:
            result = run_assess(str(npz_path), domain="life_sciences")
            s = result.get("domain_specific", {})
            for k in READY_SCALAR_METRICS:
                v = s.get(k)
                if v is not None:
                    ready_scalars[k].append(float(v))
        except Exception as e:
            errors["ready"] += 1
            if verbose:
                print(f"    READY error: {e}", flush=True)

    return raw_scalars, ready_scalars, method_counts, errors


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

DISPLAY_METRICS = [
    # (label, raw_key, ready_key, show_delta)
    # backbone: raw has no explicit mask; ready has per-residue mask
    ("backbone completeness",  None,                   "backbone_completeness",   False),
    # avg atoms/residue (raw structural quality proxy) vs sparse 37-slot fill rate (ready)
    ("avg atoms/residue [raw]","avg_atoms_per_residue", None,                     False),
    ("atom mask fill rate [ready]", None,              "atom_completeness",       False),
    # MSA: raw = max .a3m depth, ready = merged tensor depth
    ("msa depth",              "msa_depth",            "msa_depth",               True),
    ("msa duplicate ratio",    None,                   "msa_duplicate_ratio",     False),
    ("msa gap ratio",          None,                   "msa_gap_ratio",           False),
    # Structure quality: resolution preserved, R-factor only in raw
    ("resolution (Å)",         "resolution_angstrom",  "resolution_angstrom",     True),
    ("R-factor",               "r_factor",             None,                      False),
    # Class imbalance: only surfaced post-REDI
    ("AA entropy (bits)",      None,                   "aa_entropy_bits",         False),
    ("AA entropy (% uniform)", None,                   "aa_entropy_pct_uniform",  False),
    ("AA imbalance ratio",     None,                   "aa_imbalance_ratio",      False),
    # Per-chain length: raw=full structure (multi-chain CIF), ready=single chain NPZ
    ("residues [raw, all chains]", "residue_count",    None,                      False),
    ("residues [ready, per chain]",None,               "sequence_length",         False),
]


def print_table(raw_scalars, ready_scalars, n_raw, n_ready):
    print("\n" + "=" * 76)
    print(f"OpenFold: Raw vs Ready Assessment  (raw n={n_raw}, ready n={n_ready}, mean ± std)")
    print("=" * 76)
    print(f"  {'Metric':<34} {'Raw':>16} {'Ready':>16} {'Δ':>6}")
    print("-" * 76)

    for label, raw_key, ready_key, show_delta in DISPLAY_METRICS:
        raw_mu, raw_sd = mean_std(raw_scalars.get(raw_key, []))
        rdy_mu, rdy_sd = mean_std(ready_scalars.get(ready_key, []))
        raw_str = fmt(raw_mu, raw_sd) if raw_key and raw_mu is not None else ("N/A" if raw_key else "—")
        rdy_str = fmt(rdy_mu, rdy_sd) if ready_key and rdy_mu is not None else ("N/A" if ready_key else "—")
        if show_delta and raw_mu is not None and rdy_mu is not None:
            delta = f"{rdy_mu - raw_mu:+.4f}"
        else:
            delta = ""
        print(f"  {label:<34} {raw_str:>16} {rdy_str:>16} {delta:>6}")

    print("=" * 76)


def print_method_distribution(method_counts):
    total = sum(method_counts.values())
    if not total:
        return
    print(f"\nExperimental Method Bias (raw CIF, n={total:,}):")
    print("-" * 40)
    for method, count in method_counts.most_common(10):
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {method:<30} {pct:5.1f}%  {bar}")


def print_resolution_dist(res_dist):
    if not res_dist:
        return
    print(f"\nResolution Distribution (all {res_dist['n_chains']:,} chains, chain_data_cache.json):")
    print("-" * 40)
    print(f"  mean:          {res_dist['mean_angstrom']} ± {res_dist['std_angstrom']} Å")
    print(f"  median:        {res_dist['median_angstrom']} Å")
    print(f"  < 2.0 Å:       {res_dist['pct_lt_2.0A']}%")
    print(f"  < 2.5 Å:       {res_dist['pct_lt_2.5A']}%")
    print(f"  2.5 – 3.5 Å:  {res_dist['pct_2.5_to_3.5A']}%")
    print(f"  > 3.5 Å:       {res_dist['pct_gt_3.5A']}%")


def build_output_dict(raw_scalars, ready_scalars, method_counts, errors, res_dist, n_sampled):
    rows = []
    for label, raw_key, ready_key, show_delta in DISPLAY_METRICS:
        raw_mu, raw_sd = mean_std(raw_scalars.get(raw_key, []))
        rdy_mu, rdy_sd = mean_std(ready_scalars.get(ready_key, []))
        rows.append({
            "metric": label,
            "raw_mean": raw_mu,
            "raw_std": raw_sd,
            "raw_n": len(raw_scalars.get(raw_key, [])),
            "ready_mean": rdy_mu,
            "ready_std": rdy_sd,
            "ready_n": len(ready_scalars.get(ready_key, [])),
            "delta": round(rdy_mu - raw_mu, 4) if raw_mu is not None and rdy_mu is not None else None,
        })
    return {
        "n_sampled": n_sampled,
        "errors": errors,
        "metrics": rows,
        "experimental_method_distribution": {
            m: {"count": c, "pct": round(c / sum(method_counts.values()) * 100, 1)}
            for m, c in method_counts.most_common()
        },
        "resolution_distribution": res_dist,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=int, default=200,
                    help="Number of matched protein pairs to assess (default: 200)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for sampling (default: 42)")
    ap.add_argument("--output", "-o", help="Save JSON results to this path")
    ap.add_argument("--skip-resolution", action="store_true",
                    help="Skip the chain_data_cache resolution distribution step")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Print each protein as it's assessed")
    args = ap.parse_args()

    raw_scalars, ready_scalars, method_counts, errors = run_aggregate(
        args.sample, args.seed, args.verbose
    )

    res_dist = {}
    if not args.skip_resolution and CACHE.exists():
        res_dist = resolution_distribution(CACHE)

    n_raw = len(raw_scalars.get("resolution_angstrom", raw_scalars.get("r_factor", [])))
    n_ready = len(ready_scalars.get("backbone_completeness", []))
    print_table(raw_scalars, ready_scalars, n_raw, n_ready)
    print_method_distribution(method_counts)
    print_resolution_dist(res_dist)

    if errors["raw"] or errors["ready"]:
        print(f"\nErrors: raw={errors['raw']}, ready={errors['ready']}")

    if args.output:
        out = build_output_dict(
            raw_scalars, ready_scalars, method_counts, errors, res_dist, args.sample
        )
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
