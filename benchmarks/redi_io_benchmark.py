"""
REDI I/O Format Benchmark
==========================
Compares read AND write throughput across file formats for multiple domains,
simulating DataLoader access patterns at training time.

Domains and formats tested:
  life_sciences (OpenFold): native (mmCIF+A3M), per-sample NPZ, LMDB, HDF5
  materials    (OC22/HydraGNN): native sharded LMDB, REDI NPZ, REDI HDF5
  fusion       (XGC1):      ADIOS2 (native BP), HDF5, Zarr
  climate      (ClimaX):    NPZ, HDF5, Zarr

Access patterns:
  random     — pick samples uniformly at random (OpenFold, XGC1)
  sequential — read contiguous timestep windows (ClimaX)

Usage:
    python redi_io_benchmark.py --domain life_sciences \
        --npz-dir /path/to/npz --output-dir /tmp/bench \
        --alignment-dir /path/to/alignments \
        --mmcif-dir /path/to/mmcif_files

    python redi_io_benchmark.py --domain climate \
        --npz-file /path/to/data.npz --output-dir /tmp/bench

    python redi_io_benchmark.py --domain fusion \
        --h5-file /path/to/data.h5 --output-dir /tmp/bench \
        --bp-dir /path/to/xgc1_bp_snapshots

    python redi_io_benchmark.py --domain materials \
        --lmdb-dir /path/to/oc22/train \
        --npz-file /path/to/oc22_redi_output.npz \
        --output-dir /tmp/bench
"""

import argparse
import io
import multiprocessing as mp
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------

def check_dependencies(domain):
    needed = {
        "life_sciences": ["lmdb", "h5py"],
        "materials":     ["lmdb", "h5py"],
        "climate":       ["h5py", "zarr"],
        "fusion":        ["h5py", "zarr"],
    }
    missing = []
    for pkg in needed.get(domain, []):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise ImportError(
            f"Missing packages: {', '.join(missing)}\n"
            f"Install with: pip install {' '.join(missing)}"
        )
    print(f"Dependencies OK for domain={domain}")


# ---------------------------------------------------------------------------
# Packing — life_sciences
# ---------------------------------------------------------------------------

def _npz_to_bytes(path):
    """Compress a NPZ file's arrays into an in-memory NPZ buffer."""
    buf = io.BytesIO()
    d = np.load(path, allow_pickle=True)
    np.savez_compressed(buf, **{k: d[k] for k in d.keys()})
    return buf.getvalue()

def _bytes_to_arrays(data):
    return np.load(io.BytesIO(data), allow_pickle=True)

def pack_lmdb(npz_files, out_path, n_samples):
    import lmdb
    print(f"Packing {n_samples} proteins → LMDB (compressed): {out_path}")
    sample_size = np.mean([len(_npz_to_bytes(p)) for p in npz_files[:20]])
    map_size = int(n_samples * sample_size * 3)
    print(f"  avg compressed size: {sample_size/1024:.0f} KB, map_size: {map_size/1024**3:.1f} GB")
    env = lmdb.open(str(out_path), map_size=map_size, subdir=False)
    keys = []
    with env.begin(write=True) as txn:
        for i, path in enumerate(npz_files[:n_samples]):
            key = Path(path).stem.encode()
            txn.put(key, _npz_to_bytes(path))
            keys.append(key)
            if (i + 1) % 500 == 0:
                print(f"  packed {i+1}/{n_samples}")
    env.close()
    with open(str(out_path) + ".keys.pkl", "wb") as f:
        pickle.dump(keys, f)
    print(f"  LMDB size: {os.path.getsize(out_path)/1024**2:.0f} MB")
    return keys

def pack_hdf5_proteins(npz_files, out_path, n_samples):
    import h5py
    print(f"Packing {n_samples} proteins → HDF5: {out_path}")
    with h5py.File(out_path, "w") as f:
        for i, path in enumerate(npz_files[:n_samples]):
            grp = f.create_group(Path(path).stem)
            d = np.load(path, allow_pickle=True)
            for k in d.keys():
                arr = d[k]
                if arr.ndim == 0:
                    grp.create_dataset(k, data=arr)
                else:
                    grp.create_dataset(k, data=arr, compression="gzip", compression_opts=1)
            if (i + 1) % 500 == 0:
                print(f"  packed {i+1}/{n_samples}")
    print(f"  HDF5 size: {os.path.getsize(out_path)/1024**2:.0f} MB")


# ---------------------------------------------------------------------------
# Packing — climate / fusion (large array files)
# ---------------------------------------------------------------------------

def convert_npz_to_hdf5(npz_path, out_path, key="X"):
    import h5py
    # Remove stale file/lock from previous runs
    for suffix in ["", ".lock"]:
        p = Path(str(out_path) + suffix)
        if p.exists():
            p.unlink()
    print(f"Converting NPZ → HDF5: {out_path}")
    d = np.load(npz_path, allow_pickle=True)
    X = d[key]
    chunk_t = min(100, X.shape[0])
    with h5py.File(out_path, "w") as f:
        f.create_dataset(key, data=X, chunks=(chunk_t,) + X.shape[1:],
                         compression="gzip", compression_opts=1)
        for k in d.keys():
            if k != key:
                f.create_dataset(k, data=d[k])
    print(f"  HDF5 size: {os.path.getsize(out_path)/1024**2:.0f} MB")

def _zarr_write(grp, name, arr, chunk=None):
    """Write an array to a Zarr group (zarr v3)."""
    from zarr.codecs import BloscCodec
    kwargs = {"data": arr, "overwrite": True}
    if chunk:
        kwargs["chunks"] = chunk
        kwargs["compressors"] = [BloscCodec(cname="lz4", clevel=1)]
    grp.create_array(name, **kwargs)


def convert_npz_to_zarr(npz_path, out_path, key="X"):
    import zarr
    print(f"Converting NPZ → Zarr: {out_path}")
    d = np.load(npz_path, allow_pickle=True)
    X = d[key]
    chunk_t = min(100, X.shape[0])
    root = zarr.open_group(str(out_path), mode="w")
    _zarr_write(root, key, X, chunk=(chunk_t,) + X.shape[1:])
    for k in d.keys():
        if k != key:
            _zarr_write(root, k, np.asarray(d[k]))
    zarr.consolidate_metadata(str(out_path))
    print("  Zarr written (consolidated metadata)")

def convert_h5_to_zarr(h5_path, out_path):
    """Convert XGC1-style HDF5 (multiple variables, possibly nested) → Zarr."""
    import h5py, zarr
    print(f"Converting HDF5 → Zarr: {out_path}")
    root = zarr.open_group(str(out_path), mode="w")
    with h5py.File(h5_path, "r") as f:
        def _copy(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return
            arr = obj[()] if obj.ndim == 0 else obj[:]
            parts = name.split("/")
            grp = root
            for part in parts[:-1]:
                grp = grp.require_group(part)
            ds_name = parts[-1]
            chunk = (min(4, arr.shape[0]),) + arr.shape[1:] if arr.ndim >= 2 else None
            _zarr_write(grp, ds_name, arr, chunk)
        f.visititems(_copy)
    zarr.consolidate_metadata(str(out_path))
    print("  Zarr written")


# ---------------------------------------------------------------------------
# Worker functions (run in subprocess for multi-worker test)
# ---------------------------------------------------------------------------

def _worker_npz_proteins(args):
    paths, n_reads, seed = args
    rng = random.Random(seed)
    sample = rng.choices(paths, k=n_reads)
    t0 = time.perf_counter()
    for path in sample:
        d = np.load(path, allow_pickle=True)
        _ = d["aatype"]
    return n_reads, time.perf_counter() - t0

def _worker_lmdb(args):
    import lmdb
    lmdb_path, keys, n_reads, seed = args
    rng = random.Random(seed)
    sample = rng.choices(keys, k=n_reads)
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, subdir=False)
    t0 = time.perf_counter()
    with env.begin() as txn:
        for key in sample:
            d = _bytes_to_arrays(txn.get(key))
            _ = d["aatype"]
    env.close()
    return n_reads, time.perf_counter() - t0

def _worker_hdf5_proteins(args):
    import h5py
    hdf5_path, names, n_reads, seed = args
    rng = random.Random(seed)
    sample = rng.choices(names, k=n_reads)
    t0 = time.perf_counter()
    with h5py.File(hdf5_path, "r") as f:
        for name in sample:
            _ = f[name]["aatype"][:]
    return n_reads, time.perf_counter() - t0

def _worker_native_openfold(args):
    """Read native OpenFold format: mmCIF + 3 A3M alignment files per protein."""
    alignment_dir, mmcif_dir, names, n_reads, seed = args
    rng = random.Random(seed)
    sample = rng.choices(names, k=n_reads)
    t0 = time.perf_counter()
    for name in sample:
        # Read mmCIF
        pdb_id = name.split("_")[0].lower()
        mmcif_path = Path(mmcif_dir) / f"{pdb_id}.cif"
        if mmcif_path.exists():
            _ = mmcif_path.read_bytes()
        # Read 3 A3M alignment files
        aln_path = Path(alignment_dir) / name / "a3m"
        for a3m in ["uniref90_hits.a3m", "mgnify_hits.a3m", "bfd_uniclust_hits.a3m"]:
            p = aln_path / a3m
            if p.exists():
                _ = p.read_bytes()
    return n_reads, time.perf_counter() - t0

def _worker_npz_array(args):
    npz_path, n_timesteps, window, n_reads, seed = args
    rng = random.Random(seed)
    d = np.load(npz_path, allow_pickle=True)
    X = d["X"]
    t0 = time.perf_counter()
    for _ in range(n_reads):
        i = rng.randint(0, n_timesteps - window - 1)
        _ = X[i:i+window]
    return n_reads, time.perf_counter() - t0

def _worker_hdf5_array(args):
    import h5py
    hdf5_path, array_key, n_timesteps, window, n_reads, seed = args
    rng = random.Random(seed)
    t0 = time.perf_counter()
    with h5py.File(hdf5_path, "r") as f:
        X = f[array_key]
        for _ in range(n_reads):
            i = rng.randint(0, n_timesteps - window - 1)
            _ = X[i:i+window][:]
    return n_reads, time.perf_counter() - t0

def _worker_zarr_array(args):
    import zarr
    zarr_path, array_key, n_timesteps, window, n_reads, seed = args
    rng = random.Random(seed)
    root = zarr.open_consolidated(str(zarr_path), mode="r")
    X = root[array_key]
    t0 = time.perf_counter()
    for _ in range(n_reads):
        i = rng.randint(0, n_timesteps - window - 1)
        _ = X[i:i+window]
    return n_reads, time.perf_counter() - t0

def _worker_adios2(args):
    """Read native ADIOS2 BP format — simulates XGC1 training data access."""
    import adios2
    bp_paths, array_key, n_reads, seed = args
    # array_key may be a nested HDF5 path like "train/dpot"; use leaf name for ADIOS2
    adios_key = array_key.split("/")[-1]
    rng = random.Random(seed)
    sample = rng.choices(bp_paths, k=n_reads)
    t0 = time.perf_counter()
    for bp_path in sample:
        r = adios2.FileReader(bp_path)
        _ = r.read(adios_key)
        r.close()
    return n_reads, time.perf_counter() - t0


def _worker_lmdb_oc22_sharded(args):
    """Read from native OC22 sharded LMDB (HydraGNN format).

    Each shard is an independent LMDB file. Workers pick a random shard per
    read, simulating the access pattern of fairchem/HydraGNN DataLoaders.
    """
    import lmdb, pickle
    lmdb_paths, n_reads, seed = args
    rng = random.Random(seed)
    t0 = time.perf_counter()
    # open all shards (each is small enough)
    envs = [lmdb.open(p, readonly=True, lock=False, subdir=False)
            for p in lmdb_paths]
    for _ in range(n_reads):
        env = rng.choice(envs)
        with env.begin() as txn:
            n = txn.stat()["entries"]
            idx = rng.randint(0, n - 1)
            val = txn.get(str(idx).encode())
            if val is None:
                # some shards use integer keys; try sequential fallback
                cur = txn.cursor()
                cur.first()
                for _ in range(idx % min(n, 100)):
                    cur.next()
                val = cur.value()
            _ = pickle.loads(val)
    for env in envs:
        env.close()
    return n_reads, time.perf_counter() - t0


def _worker_npz_graphs(args):
    """Read random graphs from flat REDI NPZ (split_index-based access)."""
    npz_path, n_graphs, n_reads, seed = args
    rng = random.Random(seed)
    d = np.load(npz_path, allow_pickle=True)
    natoms  = d["natoms"]
    offsets = np.concatenate([[0], np.cumsum(natoms)])
    positions = d["positions"]
    forces    = d["forces"]
    t0 = time.perf_counter()
    for _ in range(n_reads):
        i = rng.randint(0, n_graphs - 1)
        s, e = offsets[i], offsets[i + 1]
        _ = positions[s:e]
        _ = forces[s:e]
    return n_reads, time.perf_counter() - t0


def _worker_hdf5_graphs(args):
    """Read random graphs from REDI HDF5 (flat arrays + offsets)."""
    import h5py
    h5_path, n_graphs, n_reads, seed = args
    rng = random.Random(seed)
    t0 = time.perf_counter()
    with h5py.File(h5_path, "r") as f:
        offsets   = f["offsets"][:]
        positions = f["positions"]
        forces    = f["forces"]
        for _ in range(n_reads):
            i = rng.randint(0, n_graphs - 1)
            s, e = int(offsets[i]), int(offsets[i + 1])
            _ = positions[s:e]
            _ = forces[s:e]
    return n_reads, time.perf_counter() - t0


def pack_hdf5_graphs(npz_path, out_path, n_graphs):
    """Pack first n_graphs from flat REDI NPZ into chunked HDF5."""
    import h5py
    print(f"Packing {n_graphs:,} graphs → HDF5: {out_path}")
    d = np.load(npz_path, allow_pickle=True)
    natoms  = d["natoms"][:n_graphs]
    offsets = np.concatenate([[0], np.cumsum(natoms)]).astype(np.int64)
    total_atoms = int(offsets[-1])
    positions = d["positions"][:total_atoms]
    forces    = d["forces"][:total_atoms]
    atomic_numbers = d["atomic_numbers"][:total_atoms]
    energy    = d["energy_per_atom"][:n_graphs]
    chunk = min(4096, total_atoms)
    for suffix in ["", ".lock"]:
        p = Path(str(out_path) + suffix)
        if p.exists(): p.unlink()
    with h5py.File(out_path, "w") as f:
        f.create_dataset("positions", data=positions,
                         chunks=(chunk, 3), compression="gzip", compression_opts=1)
        f.create_dataset("forces", data=forces,
                         chunks=(chunk, 3), compression="gzip", compression_opts=1)
        f.create_dataset("atomic_numbers", data=atomic_numbers,
                         chunks=(chunk,), compression="gzip", compression_opts=1)
        f.create_dataset("energy_per_atom", data=energy,
                         chunks=(min(4096, n_graphs),), compression="gzip", compression_opts=1)
        f.create_dataset("natoms",  data=natoms)
        f.create_dataset("offsets", data=offsets)
    print(f"  HDF5 size: {os.path.getsize(out_path)/1024**3:.2f} GB")
    return n_graphs


def benchmark_materials(args, out, timeout=120):
    """Benchmark OC22/HydraGNN: native sharded LMDB vs REDI HDF5."""
    import lmdb
    wc  = [1, 2, 4, 8, 16, 32]
    all_results = {}
    hdf5_path = out / "oc22_graphs.h5"

    # --- Native sharded LMDB ---
    if args.lmdb_dir:
        lmdb_paths = sorted(Path(args.lmdb_dir).glob("*.lmdb"))
        lmdb_paths = [str(p) for p in lmdb_paths if not p.name.endswith("-lock")]
        if lmdb_paths:
            print(f"\n--- Native sharded LMDB ({len(lmdb_paths)} shards) ---")
            all_results["lmdb_sharded"] = run_multiworker(
                _worker_lmdb_oc22_sharded, (lmdb_paths,), args.n_reads, wc, timeout=timeout)
        else:
            print(f"  No LMDB shards found in {args.lmdb_dir}")

    # --- REDI HDF5 (NPZ is not benchmarked here: flat graph NPZ requires loading
    #     the entire file per worker, making it unsuitable for this access pattern) ---
    if args.npz_file:
        d = np.load(args.npz_file, allow_pickle=True)
        n_graphs = len(d["natoms"])
        n_bench  = min(args.n_samples, n_graphs)
        print(f"\nGraphs available: {n_graphs:,}, benchmarking {n_bench:,}")

        if not args.skip_pack:
            pack_hdf5_graphs(args.npz_file, hdf5_path, n_bench)
        print("\n--- REDI HDF5 (flat + offsets, chunked gzip) ---")
        all_results["hdf5"] = run_multiworker(
            _worker_hdf5_graphs, (str(hdf5_path), n_bench), args.n_reads, wc, timeout=timeout)

    return all_results


# ---------------------------------------------------------------------------
# Write benchmarks (single-threaded — packing is sequential in REDI)
# ---------------------------------------------------------------------------

def benchmark_write_npz(npz_files, n_samples, out_path):
    """Measure write throughput: copy N NPZ files to output dir."""
    import shutil
    out_path.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    for p in npz_files[:n_samples]:
        shutil.copy2(p, out_path / Path(p).name)
    elapsed = time.perf_counter() - t0
    tput = n_samples / elapsed
    print(f"  write NPZ ({n_samples} files): {tput:.1f} files/sec  ({elapsed:.1f}s)")
    return tput

def benchmark_write_lmdb(npz_files, n_samples, lmdb_path):
    """Measure LMDB write throughput (compressed)."""
    import lmdb
    sample_size = np.mean([len(_npz_to_bytes(p)) for p in npz_files[:10]])
    map_size = int(n_samples * sample_size * 10)  # 10x headroom
    t0 = time.perf_counter()
    env = lmdb.open(str(lmdb_path) + "_write_bench", map_size=map_size, subdir=False)
    with env.begin(write=True) as txn:
        for p in npz_files[:n_samples]:
            txn.put(Path(p).stem.encode(), _npz_to_bytes(p))
    env.close()
    elapsed = time.perf_counter() - t0
    tput = n_samples / elapsed
    print(f"  write LMDB ({n_samples} proteins): {tput:.1f} proteins/sec  ({elapsed:.1f}s)")
    return tput

def benchmark_write_hdf5(npz_files, n_samples, hdf5_path):
    """Measure HDF5 write throughput."""
    import h5py
    t0 = time.perf_counter()
    with h5py.File(str(hdf5_path) + "_write_bench.h5", "w") as f:
        for p in npz_files[:n_samples]:
            grp = f.create_group(Path(p).stem)
            d = np.load(p, allow_pickle=True)
            for k in d.keys():
                arr = d[k]
                if arr.ndim == 0:
                    grp.create_dataset(k, data=arr)
                else:
                    grp.create_dataset(k, data=arr, compression="gzip", compression_opts=1)
    elapsed = time.perf_counter() - t0
    tput = n_samples / elapsed
    print(f"  write HDF5 ({n_samples} proteins): {tput:.1f} proteins/sec  ({elapsed:.1f}s)")
    return tput


# ---------------------------------------------------------------------------
# Multi-worker runner
# ---------------------------------------------------------------------------

def run_multiworker(worker_fn, base_args, n_reads_total, worker_counts, timeout=120):
    results = {}
    for nw in worker_counts:
        reads_per_worker = max(1, n_reads_total // nw)
        args_list = [base_args + (reads_per_worker, 42 + w) for w in range(nw)]
        t0 = time.perf_counter()
        with mp.Pool(nw) as pool:
            async_result = pool.map_async(worker_fn, args_list)
            try:
                outs = async_result.get(timeout=timeout)
            except mp.TimeoutError:
                pool.terminate()
                print(f"    workers={nw:2d}: TIMEOUT after {timeout}s — skipping")
                results[nw] = float("nan")
                continue
        wall = time.perf_counter() - t0
        tput = sum(o[0] for o in outs) / wall
        results[nw] = tput
        print(f"    workers={nw:2d}: {tput:8.1f} samples/sec  (wall={wall:.2f}s)")
    return results


# ---------------------------------------------------------------------------
# Domain benchmarks
# ---------------------------------------------------------------------------

def benchmark_life_sciences(args, out, timeout=120):
    npz_files = sorted(Path(args.npz_dir).glob("*.npz"))
    print(f"Found {len(npz_files):,} NPZ files")
    n = min(args.n_samples, len(npz_files))
    npz_sample = [str(p) for p in npz_files[:n]]
    names = [Path(p).stem for p in npz_sample]
    lmdb_path = out / "train.lmdb"
    hdf5_path = out / "train.h5"

    if not args.skip_pack:
        keys = pack_lmdb([Path(p) for p in npz_sample], lmdb_path, n)
        pack_hdf5_proteins([Path(p) for p in npz_sample], hdf5_path, n)
    else:
        with open(str(lmdb_path) + ".keys.pkl", "rb") as f:
            keys = pickle.load(f)

    wc = [1, 2, 4, 8, 16, 32]
    read_results = {}
    write_results = {}

    # --- Read benchmarks ---
    if args.alignment_dir and args.mmcif_dir:
        # Only include proteins that have native alignment data
        native_names = [n for n in names
                        if (Path(args.alignment_dir) / n / "a3m" / "uniref90_hits.a3m").exists()][:n]
        print(f"\n--- Native OpenFold (mmCIF + 3×A3M, {len(native_names)} proteins) ---")
        read_results["native"] = run_multiworker(
            _worker_native_openfold,
            (args.alignment_dir, args.mmcif_dir, native_names),
            args.n_reads, wc, timeout=timeout)

    print("\n--- NPZ (per-protein files) ---")
    read_results["npz"] = run_multiworker(_worker_npz_proteins, (npz_sample,), args.n_reads, wc, timeout=timeout)
    print("\n--- LMDB (compressed) ---")
    read_results["lmdb"] = run_multiworker(_worker_lmdb, (str(lmdb_path), keys), args.n_reads, wc, timeout=timeout)
    print("\n--- HDF5 (all proteins, one file) ---")
    read_results["hdf5"] = run_multiworker(_worker_hdf5_proteins, (str(hdf5_path), names), args.n_reads, wc, timeout=timeout)

    # --- Write benchmarks (n_samples=200 for speed) ---
    n_write = min(200, n)
    print(f"\n=== WRITE BENCHMARKS ({n_write} proteins) ===")
    write_results["npz"]  = benchmark_write_npz([Path(p) for p in npz_sample[:n_write]],
                                                 n_write, out / "write_npz")
    write_results["lmdb"] = benchmark_write_lmdb([Path(p) for p in npz_sample[:n_write]],
                                                  n_write, lmdb_path)
    write_results["hdf5"] = benchmark_write_hdf5([Path(p) for p in npz_sample[:n_write]],
                                                  n_write, hdf5_path)

    return {"read": read_results, "write": write_results}


def benchmark_array_domain(args, out, timeout=120):
    import h5py
    wc = [1, 2, 4, 8, 16, 32]
    all_results = {}
    zarr_path = out / "data.zarr"

    if args.h5_file:
        # Fusion: input is HDF5; benchmark ADIOS2 (native) vs HDF5 vs Zarr
        # Datasets may be nested under a group (e.g. train/dpot)
        with h5py.File(args.h5_file, "r") as f:
            found = []
            f.visititems(lambda name, obj:
                found.append(name) if isinstance(obj, h5py.Dataset) and obj.ndim >= 2 and not found else None)
            if not found:
                raise RuntimeError(f"No 2-D datasets found in {args.h5_file}")
            key = found[0]          # e.g. "train/dpot"
            shape = f[key].shape
        n_timesteps = shape[0]
        print(f"HDF5 key='{key}', shape={shape}, window={args.window}")
        if not args.skip_pack:
            convert_h5_to_zarr(args.h5_file, zarr_path)

        # ADIOS2 native: benchmark across multiple BP snapshot files
        if args.bp_dir:
            bp_paths = sorted(Path(args.bp_dir).glob("xgc.3d.*.bp"))[:20]
            if bp_paths:
                print(f"\n--- ADIOS2 native ({len(bp_paths)} BP snapshots) ---")
                all_results["adios2"] = run_multiworker(
                    _worker_adios2,
                    (list(map(str, bp_paths)), key),
                    args.n_reads, wc, timeout=timeout)
            else:
                print(f"  No BP files found in {args.bp_dir} — skipping ADIOS2")

        print("\n--- HDF5 (REDI output) ---")
        all_results["hdf5"] = run_multiworker(_worker_hdf5_array,
                                              (str(args.h5_file), key, n_timesteps, args.window),
                                              args.n_reads, wc, timeout=timeout)
        print("\n--- Zarr (converted, lz4) ---")
        all_results["zarr"] = run_multiworker(_worker_zarr_array,
                                              (str(zarr_path), key, n_timesteps, args.window),
                                              args.n_reads, wc, timeout=timeout)
    else:
        # Climate: input is large NPZ; benchmark NPZ vs HDF5 vs Zarr
        d = np.load(args.npz_file, allow_pickle=True)
        n_timesteps = d["X"].shape[0]
        print(f"NPZ shape: {d['X'].shape}, window: {args.window}")
        hdf5_path = out / "data.h5"
        if not args.skip_pack:
            convert_npz_to_hdf5(args.npz_file, hdf5_path)
            convert_npz_to_zarr(args.npz_file, zarr_path)
        print("\n--- NPZ (single large file) ---")
        all_results["npz"] = run_multiworker(_worker_npz_array,
                                             (args.npz_file, n_timesteps, args.window),
                                             args.n_reads, wc, timeout=timeout)
        print("\n--- HDF5 (chunked) ---")
        all_results["hdf5"] = run_multiworker(_worker_hdf5_array,
                                              (str(hdf5_path), "X", n_timesteps, args.window),
                                              args.n_reads, wc, timeout=timeout)
        print("\n--- Zarr (chunked, lz4) ---")
        all_results["zarr"] = run_multiworker(_worker_zarr_array,
                                              (str(zarr_path), "X", n_timesteps, args.window),
                                              args.n_reads, wc, timeout=timeout)
    return all_results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results, domain):
    # Handle both flat dict (climate/fusion) and {read, write} dict (life_sciences)
    read_results  = results.get("read",  results) if isinstance(results, dict) and "read" in results else results
    write_results = results.get("write", {})

    wc = sorted(next(iter(read_results.values())).keys())
    print(f"\n{'='*65}")
    print(f"SUMMARY — {domain.upper()}  (READ samples/sec)")
    header = f"  {'Format':<8}" + "".join(f"  {f'w={n}':>10}" for n in wc)
    print(header)
    print("-" * 65)
    for fmt, r in read_results.items():
        def _fmt(v):
            return f"{'TIMEOUT':>10}" if (isinstance(v, float) and np.isnan(v)) else f"{v:>10.1f}"
        row = f"  {fmt:<8}" + "".join(f"  {_fmt(r[n])}" for n in wc)
        print(row)
    valid = {f: r for f, r in read_results.items()
             if not (isinstance(r[max(wc)], float) and np.isnan(r[max(wc)]))}
    best = max(valid, key=lambda f: valid[f][max(wc)]) if valid else "N/A"
    print(f"\nBest read at {max(wc)} workers: {best.upper()}")

    if write_results:
        print(f"\n{'WRITE throughput (proteins/sec)':}")
        for fmt, tput in write_results.items():
            print(f"  {fmt:<8}: {tput:.1f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True,
                        choices=["life_sciences", "materials", "climate", "fusion"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--n-reads",   type=int, default=1000)
    parser.add_argument("--window",    type=int, default=10,
                        help="Timestep window for climate/fusion reads")
    parser.add_argument("--skip-pack", action="store_true")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Per-worker-count timeout in seconds (default: 120)")
    parser.add_argument("--npz-dir",       help="Dir of per-protein NPZ (life_sciences)")
    parser.add_argument("--npz-file",      help="Single large NPZ file (climate)")
    parser.add_argument("--h5-file",       help="HDF5 file (fusion)")
    parser.add_argument("--alignment-dir", help="OpenFold alignment dir (native benchmark)")
    parser.add_argument("--mmcif-dir",     help="OpenFold mmCIF dir (native benchmark)")
    parser.add_argument("--bp-dir",        help="Dir of ADIOS2 BP snapshots (fusion native benchmark)")
    parser.add_argument("--lmdb-dir",      help="Dir of sharded OC22 LMDB files (materials native benchmark)")
    args = parser.parse_args()

    check_dependencies(args.domain)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n== REDI I/O Benchmark: {args.domain} ==")
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    if args.domain == "life_sciences":
        if not args.npz_dir:
            parser.error("--npz-dir required for life_sciences")
        results = benchmark_life_sciences(args, out, timeout=args.timeout)
    elif args.domain == "materials":
        if not args.lmdb_dir and not args.npz_file:
            parser.error("--lmdb-dir and/or --npz-file required for materials")
        results = benchmark_materials(args, out, timeout=args.timeout)
    elif args.domain == "climate":
        if not args.npz_file:
            parser.error("--npz-file required for climate")
        results = benchmark_array_domain(args, out, timeout=args.timeout)
    else:  # fusion
        if not args.h5_file and not args.npz_file:
            parser.error("--h5-file or --npz-file required for fusion")
        results = benchmark_array_domain(args, out, timeout=args.timeout)

    print_summary(results, args.domain)

    import json
    out_path = out / f"benchmark_{args.domain}.json"
    with open(out_path, "w") as f:
        json.dump({"domain": args.domain,
                   "results": {fmt: {str(k): v for k, v in r.items()}
                                for fmt, r in results.items()}}, f, indent=2)
    print(f"\nResults → {out_path}")
    print(f"End: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
