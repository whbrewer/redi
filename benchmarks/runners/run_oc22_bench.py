#!/usr/bin/env python3
"""
OC22 / HydraGNN pipeline stage timing benchmark.

The OC22 data is stored as sharded LMDB files containing torch_geometric Data
objects (atomic positions, numbers, energies, forces).  REDI's materials domain
currently handles crystal CIF/POSCAR files; this standalone runner implements the
equivalent five-stage pipeline for the LMDB format and times each stage.

Stages:
  ingest     – open LMDB shards, iterate and deserialize Data objects
  preprocess – validate structures (drop NaN energies/forces, check natoms)
  transform  – normalize: centre-of-mass positions, per-atom energies, z-score forces
  structure  – assemble padded numpy tensors (atomic_numbers, pos, energy, forces)
  output     – save consolidated NPZ + metadata JSON

Usage:
    source ~/venv/redi/bin/activate
    python benchmarks/runners/run_oc22_bench.py [--split SPLIT] [--max-samples N] [--out FILE]

    --split      train | val_id | val_ood | test_id | test_ood  (default: train)
    --max-samples  Cap number of structures (default: all)
    --out          Output JSON path
"""

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

# ── Data locations ─────────────────────────────────────────────────────────────
LMDB_ROOT = ("/lustre/orion/gen053/world-shared/HydraGNN/data/oc22"
             "/s2ef_total_train_val_test_lmdbs/data/oc22/s2ef-total")
OUT_JSON  = str(Path(__file__).parent.parent / "results" / "bench_oc22.json")

STAGES = ["ingest", "preprocess", "transform", "structure", "output"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_lmdb_shards(split_dir: str) -> list:
    """Return sorted list of .lmdb shard paths (exclude -lock files)."""
    return sorted(
        p for p in Path(split_dir).iterdir()
        if p.suffix == ".lmdb" and not p.name.endswith("-lock")
    )


def lmdb_iter(shard_paths: list, max_samples: int | None):
    """Yield raw pickled bytes from LMDB shards up to max_samples."""
    import lmdb
    count = 0
    for shard in shard_paths:
        env = lmdb.open(
            str(shard), subdir=False, readonly=True, lock=False,
            meminit=False, map_size=int(1e12),
        )
        txn = env.begin()
        cursor = txn.cursor()
        cursor.first()
        while True:
            raw = cursor.value()
            yield raw
            count += 1
            if max_samples is not None and count >= max_samples:
                txn.abort()
                env.close()
                return
            if not cursor.next():
                break
        txn.abort()
        env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline stages
# ─────────────────────────────────────────────────────────────────────────────

def stage_ingest(shards: list, max_samples: int | None) -> list:
    """Load and deserialise Data objects from LMDB shards."""
    records = []
    for raw in lmdb_iter(shards, max_samples):
        obj = pickle.loads(raw)
        records.append(obj)
    return records


def _get(obj, key, default=None):
    """Get attribute from a PyG Data object regardless of PyG version."""
    try:
        d = vars(obj)
        return d.get(key, default)
    except TypeError:
        return getattr(obj, key, default)


def stage_preprocess(records: list) -> list:
    """Drop structures with NaN energy or forces, or zero atoms."""
    valid = []
    n_dropped = 0
    for obj in records:
        try:
            energy = _get(obj, "y")
            natoms = _get(obj, "natoms")
            force  = _get(obj, "force")
            if energy is None or natoms is None or force is None:
                n_dropped += 1
                continue
            energy = float(energy)
            natoms = int(natoms)
            if natoms == 0:
                n_dropped += 1
                continue
            if not np.isfinite(energy):
                n_dropped += 1
                continue
            forces = np.array(force)
            if not np.all(np.isfinite(forces)):
                n_dropped += 1
                continue
            valid.append(obj)
        except Exception:
            n_dropped += 1
    if n_dropped:
        print(f"    preprocess: dropped {n_dropped} invalid structures", flush=True)
    return valid


def stage_transform(records: list) -> list:
    """
    Normalise each structure:
    - Centre atomic positions (subtract centre of mass, equal weights).
    - Compute per-atom energy: E_atom = y / natoms.
    - Z-score forces across each structure.
    Returns list of dicts with numpy arrays.
    """
    out = []
    for obj in records:
        pos            = np.array(_get(obj, "pos"), dtype=np.float32)
        atomic_numbers = np.array(_get(obj, "atomic_numbers"), dtype=np.int32)
        forces         = np.array(_get(obj, "force"), dtype=np.float32)
        energy         = float(_get(obj, "y"))
        natoms         = int(_get(obj, "natoms"))
        tags_raw       = _get(obj, "tags")
        tags = np.array(tags_raw, dtype=np.int32) if tags_raw is not None else np.zeros(natoms, np.int32)

        # Centre of mass (equal weights)
        com = pos.mean(axis=0)
        pos_centred = pos - com

        # Per-atom energy
        e_per_atom = energy / natoms

        # Z-score forces (per-structure, avoid div-by-zero)
        f_std = forces.std()
        forces_norm = (forces - forces.mean()) / (f_std + 1e-8)

        out.append({
            "atomic_numbers": atomic_numbers,
            "pos": pos_centred,
            "energy_per_atom": np.float32(e_per_atom),
            "forces": forces_norm,
            "tags": tags,
            "natoms": natoms,
            "sid": getattr(obj, "sid", None),
        })
    return out


def stage_structure(records: list) -> tuple:
    """
    Assemble a consolidated set of arrays for the full batch:
    - flat_atomic_numbers:  [total_atoms]
    - flat_pos:             [total_atoms, 3]
    - flat_forces:          [total_atoms, 3]
    - energies_per_atom:    [N_structures]
    - natoms:               [N_structures]
    - structure_ids:        [N_structures] (sid strings)
    - split_index:          cumulative atom counts for un-batching
    """
    flat_z   = []
    flat_pos = []
    flat_f   = []
    energies = []
    natoms   = []
    sids     = []

    if not records:
        empty = np.empty(0, dtype=np.int32)
        return {
            "atomic_numbers": empty,
            "positions":      np.empty((0, 3), dtype=np.float32),
            "forces":         np.empty((0, 3), dtype=np.float32),
            "energy_per_atom": np.empty(0, dtype=np.float32),
            "natoms":         empty,
            "split_index":    empty,
        }, []

    for r in records:
        flat_z.append(r["atomic_numbers"])
        flat_pos.append(r["pos"])
        flat_f.append(r["forces"])
        energies.append(r["energy_per_atom"])
        natoms.append(r["natoms"])
        sids.append(str(r.get("sid", "")))

    tensors = {
        "atomic_numbers": np.concatenate(flat_z),
        "positions":      np.concatenate(flat_pos, axis=0),
        "forces":         np.concatenate(flat_f, axis=0),
        "energy_per_atom": np.array(energies, dtype=np.float32),
        "natoms":         np.array(natoms, dtype=np.int32),
        "split_index":    np.cumsum([0] + natoms[:-1], dtype=np.int32),
    }
    return tensors, sids


def stage_output(tensors: dict, sids: list, out_dir: str, n_total: int) -> dict:
    """
    Save tensors to NPZ + write metadata JSON.
    Returns dict with output file sizes.
    """
    import tempfile
    out_path = Path(out_dir) / "oc22_redi_output.npz"
    meta_path = Path(out_dir) / "metadata.json"

    np.savez_compressed(str(out_path), **tensors)

    meta = {
        "n_structures": n_total,
        "n_atoms_total": int(tensors["atomic_numbers"].shape[0]),
        "output_format": "npz",
        "arrays": {k: list(v.shape) for k, v in tensors.items()},
        "source": "OC22 s2ef-total LMDB",
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    size_mb = out_path.stat().st_size / (1024 ** 2) if out_path.exists() else 0
    return {"npz_path": str(out_path), "size_mb": round(size_mb, 1)}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train",
                        choices=["train", "val_id", "val_ood", "test_id", "test_ood"])
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap number of structures (default: process all)")
    parser.add_argument("--out", default=OUT_JSON, help="Output JSON path")
    parser.add_argument("--save-dir", default="/tmp/redi_oc22_bench_output",
                        help="Directory for NPZ output (default: /tmp/...)")
    args = parser.parse_args()

    split_dir = str(Path(LMDB_ROOT) / args.split)
    shards = get_lmdb_shards(split_dir)
    if not shards:
        raise SystemExit(f"No .lmdb shards found in {split_dir}")

    n_shards = len(shards)
    sample_label = f"{args.max_samples:,}" if args.max_samples else "all"

    print(f"\nOC22 / HydraGNN Stage Timing Benchmark")
    print(f"  LMDB root: {LMDB_ROOT}")
    print(f"  Split:     {args.split}  ({n_shards} shards)")
    print(f"  Samples:   {sample_label}")
    print()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    stage_ms: dict = {s: 0 for s in STAGES}

    # ── Ingest ────────────────────────────────────────────────────────────────
    print("  [ingest]  loading from LMDB …", flush=True)
    t0 = time.time()
    records = stage_ingest(shards, args.max_samples)
    stage_ms["ingest"] = int((time.time() - t0) * 1000)
    n_loaded = len(records)
    print(f"    loaded {n_loaded:,} structures  ({stage_ms['ingest']/1000:.1f}s)", flush=True)

    # ── Preprocess ───────────────────────────────────────────────────────────
    print("  [preprocess]  validating structures …", flush=True)
    t0 = time.time()
    records = stage_preprocess(records)
    stage_ms["preprocess"] = int((time.time() - t0) * 1000)
    n_valid = len(records)
    print(f"    {n_valid:,} valid  ({stage_ms['preprocess']/1000:.1f}s)", flush=True)

    # ── Transform ────────────────────────────────────────────────────────────
    print("  [transform]  normalising positions / energies / forces …", flush=True)
    t0 = time.time()
    records = stage_transform(records)
    stage_ms["transform"] = int((time.time() - t0) * 1000)
    print(f"    done  ({stage_ms['transform']/1000:.1f}s)", flush=True)

    # ── Structure ────────────────────────────────────────────────────────────
    print("  [structure]  assembling batch tensors …", flush=True)
    t0 = time.time()
    tensors, sids = stage_structure(records)
    stage_ms["structure"] = int((time.time() - t0) * 1000)
    n_atoms_total = int(tensors["atomic_numbers"].shape[0])
    print(f"    {n_atoms_total:,} atoms total  ({stage_ms['structure']/1000:.1f}s)", flush=True)

    # ── Output ───────────────────────────────────────────────────────────────
    print("  [output]  saving NPZ …", flush=True)
    t0 = time.time()
    out_info = stage_output(tensors, sids, args.save_dir, n_valid)
    stage_ms["output"] = int((time.time() - t0) * 1000)
    print(f"    {out_info['npz_path']}  ({out_info['size_mb']:.0f} MB)  "
          f"({stage_ms['output']/1000:.1f}s)", flush=True)

    total_ms = sum(stage_ms.values())
    throughput = n_valid / (total_ms / 1000) if total_ms > 0 else 0

    result = {
        "case": "oc22_hydragnn",
        "domain": "materials",
        "split": args.split,
        "n_shards": n_shards,
        "n_loaded": n_loaded,
        "n_valid": n_valid,
        "n_atoms_total": n_atoms_total,
        "stages_ms": stage_ms,
        "total_ms": total_ms,
        "total_s": round(total_ms / 1000, 1),
        "throughput_structs_per_s": round(throughput, 2),
        "throughput_label": "structs/s",
        "output": out_info,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults → {args.out}")

    W = 65
    print("\n" + "=" * W)
    print(f"OC22 TIMING SUMMARY  ({n_valid:,} structures, split={args.split})")
    print("=" * W)
    print(f"  {'Stage':<16}  {'s':>8}  {'%':>6}")
    print(f"  {'-'*16}  {'-'*8}  {'-'*6}")
    for stage in STAGES:
        ms = stage_ms[stage]
        pct = 100 * ms / total_ms if total_ms else 0
        print(f"  {stage:<16}  {ms/1000:>8.1f}  {pct:>5.1f}%")
    print(f"  {'TOTAL':<16}  {total_ms/1000:>8.1f}  100.0%")
    print(f"\n  Throughput: {throughput:.2f} structs/s")
    print("=" * W)


if __name__ == "__main__":
    main()
