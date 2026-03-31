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
  transform  – z-score normalize node features; compute cutoff-radius graph edges
  structure  – assemble per-structure PyG Data objects (nodes, edges, targets)
  output     – write ADIOS2 .bp file via HydraGNN AdiosWriter (train/val/test splits)

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


RADIUS_CUTOFF   = 6.0   # Angstroms — standard for OC22/HydraGNN
MAX_NEIGHBOURS  = 100   # max edges per node


def _radius_graph_numpy(pos: np.ndarray, r: float, max_neighbours: int):
    """Build radius graph using scipy cKDTree. Returns edge_index [2, E] and
    edge_attr (distances) [E, 1] as float32 numpy arrays."""
    from scipy.spatial import cKDTree
    tree = cKDTree(pos)
    pairs = tree.query_pairs(r=r, output_type="ndarray")   # [M, 2] undirected
    if len(pairs) == 0:
        return np.empty((2, 0), dtype=np.int64), np.empty((0, 1), dtype=np.float32)

    # Make directed (both directions) and prune to max_neighbours per node
    src = np.concatenate([pairs[:, 0], pairs[:, 1]])
    dst = np.concatenate([pairs[:, 1], pairs[:, 0]])
    diff = pos[dst] - pos[src]
    dist = np.linalg.norm(diff, axis=1).astype(np.float32)

    if max_neighbours < len(src):
        # Keep closest max_neighbours per node
        order = np.argsort(dist)
        keep  = np.zeros(len(src), dtype=bool)
        count = np.zeros(len(pos), dtype=np.int32)
        for idx in order:
            n = src[idx]
            if count[n] < max_neighbours:
                keep[idx] = True
                count[n] += 1
        src, dst, dist = src[keep], dst[keep], dist[keep]

    edge_index = np.stack([src, dst], axis=0).astype(np.int64)
    edge_attr  = dist[:, None]   # [E, 1]
    return edge_index, edge_attr


def stage_transform(records: list) -> list:
    """
    Per structure:
    - Centre atomic positions (subtract centre of mass).
    - Compute per-atom energy.
    - Z-score normalise forces.
    - Build cutoff-radius graph edges (radius=6 Å, max_neighbours=100).
    Returns list of dicts ready for PyG Data assembly.
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

        # Centre of mass
        pos_centred = pos - pos.mean(axis=0)

        # Per-atom energy
        e_per_atom = np.float32(energy / natoms)

        # Z-score forces (per-structure)
        f_std = forces.std()
        forces_norm = (forces - forces.mean()) / (f_std + 1e-8)

        # Cutoff-radius graph edges
        edge_index, edge_attr = _radius_graph_numpy(pos_centred, RADIUS_CUTOFF, MAX_NEIGHBOURS)

        out.append({
            "atomic_numbers": atomic_numbers,
            "pos":            pos_centred,
            "energy_per_atom": e_per_atom,
            "forces":         forces_norm,
            "tags":           tags,
            "natoms":         natoms,
            "edge_index":     edge_index,
            "edge_attr":      edge_attr,
            "sid":            getattr(obj, "sid", None),
        })
    return out


def stage_structure(records: list) -> list:
    """
    Assemble per-structure PyG Data objects.
    Node features x = [Z/100, tag]  (shape: [N, 2])
    Graph target  y = [energy_per_atom]
    Node target stored as additional x columns (forces_x, forces_y, forces_z)
    so HydraGNN can read them as node-level outputs.
    """
    import torch
    from torch_geometric.data import Data

    data_objects = []
    for r in records:
        N   = r["natoms"]
        Z   = r["atomic_numbers"].astype(np.float32) / 100.0   # normalise Z
        tag = r["tags"].astype(np.float32)
        x   = np.stack([Z, tag], axis=1)                       # [N, 2]

        data = Data(
            x          = torch.as_tensor(x,              dtype=torch.float32),
            pos        = torch.as_tensor(r["pos"],        dtype=torch.float32),
            y          = torch.as_tensor([r["energy_per_atom"]], dtype=torch.float32),
            edge_index = torch.as_tensor(r["edge_index"], dtype=torch.long),
            edge_attr  = torch.as_tensor(r["edge_attr"],  dtype=torch.float32),
            force      = torch.as_tensor(r["forces"],     dtype=torch.float32),
            natoms     = torch.tensor([N],                dtype=torch.long),
        )
        data_objects.append(data)
    return data_objects


def stage_output(data_objects: list, out_dir: str, split_ratios=(0.8, 0.1, 0.1)) -> dict:
    """
    Write train/val/test splits to an ADIOS2 .bp file using HydraGNN's AdiosWriter.
    """
    import sys
    sys.path.insert(0, "/lustre/orion/gen053/world-shared/HydraGNN/code")
    from mpi4py import MPI
    from hydragnn.utils.datasets.adiosdataset import AdiosWriter

    out_path = Path(out_dir) / "oc22_redi_output.bp"

    n       = len(data_objects)
    n_train = int(n * split_ratios[0])
    n_val   = int(n * split_ratios[1])
    trainset = data_objects[:n_train]
    valset   = data_objects[n_train:n_train + n_val]
    testset  = data_objects[n_train + n_val:]

    comm = MPI.COMM_SELF
    adwriter = AdiosWriter(str(out_path), comm)
    adwriter.add("trainset", trainset)
    adwriter.add("valset",   valset)
    adwriter.add("testset",  testset)
    adwriter.save()

    size_mb = sum(
        f.stat().st_size for f in out_path.iterdir() if out_path.is_dir()
    ) / (1024 ** 2) if out_path.exists() else 0

    return {
        "bp_path":    str(out_path),
        "n_train":    len(trainset),
        "n_val":      len(valset),
        "n_test":     len(testset),
        "size_mb":    round(size_mb, 1),
        "output_format": "adios2",
    }


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
    print("  [structure]  assembling PyG Data objects …", flush=True)
    t0 = time.time()
    data_objects = stage_structure(records)
    stage_ms["structure"] = int((time.time() - t0) * 1000)
    n_atoms_total = sum(int(d.natoms[0]) for d in data_objects)
    print(f"    {len(data_objects):,} Data objects, {n_atoms_total:,} atoms total  "
          f"({stage_ms['structure']/1000:.1f}s)", flush=True)

    # ── Output ───────────────────────────────────────────────────────────────
    print("  [output]  writing ADIOS2 .bp via HydraGNN AdiosWriter …", flush=True)
    t0 = time.time()
    out_info = stage_output(data_objects, args.save_dir)
    stage_ms["output"] = int((time.time() - t0) * 1000)
    print(f"    {out_info['bp_path']}  ({out_info['size_mb']:.0f} MB)  "
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
    print(f"  Output:     {out_info['bp_path']}  (ADIOS2, {out_info['n_train']:,} train / {out_info['n_val']:,} val / {out_info['n_test']:,} test)")
    print("=" * W)


if __name__ == "__main__":
    main()
