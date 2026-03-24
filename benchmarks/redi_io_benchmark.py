"""
REDI I/O Format Benchmark
==========================
Compares read throughput across file formats for multiple domains,
simulating DataLoader access patterns at training time.

Domains and formats tested:
  life_sciences (OpenFold): per-sample NPZ, LMDB (compressed), HDF5
  fusion       (XGC1):      NPZ, HDF5, Zarr
  climate      (ClimaX):    NPZ, HDF5, Zarr

Access patterns:
  random     — pick samples uniformly at random (OpenFold, XGC1)
  sequential — read contiguous timestep windows (ClimaX)

Usage:
    python redi_io_benchmark.py --domain life_sciences \
        --npz-dir /path/to/npz --output-dir /tmp/bench

    python redi_io_benchmark.py --domain climate \
        --npz-file /path/to/data.npz --output-dir /tmp/bench

    python redi_io_benchmark.py --domain fusion \
        --npz-file /path/to/data.npz --output-dir /tmp/bench
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

def convert_npz_to_zarr(npz_path, out_path, key="X"):
    import zarr
    print(f"Converting NPZ → Zarr: {out_path}")
    d = np.load(npz_path, allow_pickle=True)
    X = d[key]
    chunk_t = min(100, X.shape[0])
    root = zarr.open_group(str(out_path), mode="w")
    root.array(key, X, chunks=(chunk_t,) + X.shape[1:],
               compressor=zarr.Blosc(cname="lz4", clevel=1))
    for k in d.keys():
        if k != key:
            root.array(k, d[k])
    zarr.consolidate_metadata(str(out_path))
    print("  Zarr written (consolidated metadata)")

def convert_h5_to_zarr(h5_path, out_path):
    """Convert XGC1-style HDF5 (multiple variables) → Zarr."""
    import h5py, zarr
    print(f"Converting HDF5 → Zarr: {out_path}")
    root = zarr.open_group(str(out_path), mode="w")
    with h5py.File(h5_path, "r") as f:
        for key in f.keys():
            arr = f[key][:]
            if arr.ndim >= 2:
                chunk = (min(4, arr.shape[0]),) + arr.shape[1:]
                root.array(key, arr, chunks=chunk,
                           compressor=zarr.Blosc(cname="lz4", clevel=1))
            else:
                root.array(key, arr)
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


# ---------------------------------------------------------------------------
# Multi-worker runner
# ---------------------------------------------------------------------------

def run_multiworker(worker_fn, base_args, n_reads_total, worker_counts):
    results = {}
    for nw in worker_counts:
        reads_per_worker = max(1, n_reads_total // nw)
        args_list = [base_args + (reads_per_worker, 42 + w) for w in range(nw)]
        t0 = time.perf_counter()
        with mp.Pool(nw) as pool:
            outs = pool.map(worker_fn, args_list)
        wall = time.perf_counter() - t0
        tput = sum(o[0] for o in outs) / wall
        results[nw] = tput
        print(f"    workers={nw:2d}: {tput:8.1f} samples/sec  (wall={wall:.2f}s)")
    return results


# ---------------------------------------------------------------------------
# Domain benchmarks
# ---------------------------------------------------------------------------

def benchmark_life_sciences(args, out):
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

    wc = [1, 2, 4, 8]
    all_results = {}
    print("\n--- NPZ (per-protein files) ---")
    all_results["npz"] = run_multiworker(_worker_npz_proteins, (npz_sample,), args.n_reads, wc)
    print("\n--- LMDB (compressed) ---")
    all_results["lmdb"] = run_multiworker(_worker_lmdb, (str(lmdb_path), keys), args.n_reads, wc)
    print("\n--- HDF5 (all proteins, one file) ---")
    all_results["hdf5"] = run_multiworker(_worker_hdf5_proteins, (str(hdf5_path), names), args.n_reads, wc)
    return all_results


def benchmark_array_domain(args, out):
    import h5py
    wc = [1, 2, 4, 8]
    all_results = {}
    zarr_path = out / "data.zarr"

    if args.h5_file:
        # Fusion: input is HDF5; benchmark HDF5 vs Zarr
        with h5py.File(args.h5_file, "r") as f:
            # Use first 2D variable as the benchmark array
            key = next(k for k in f.keys() if f[k].ndim >= 2)
            shape = f[key].shape
        n_timesteps = shape[0]
        print(f"HDF5 key='{key}', shape={shape}, window={args.window}")
        if not args.skip_pack:
            convert_h5_to_zarr(args.h5_file, zarr_path)
        print("\n--- HDF5 (original) ---")
        all_results["hdf5"] = run_multiworker(_worker_hdf5_array,
                                              (str(args.h5_file), key, n_timesteps, args.window),
                                              args.n_reads, wc)
        print("\n--- Zarr (converted, lz4) ---")
        all_results["zarr"] = run_multiworker(_worker_zarr_array,
                                              (str(zarr_path), key, n_timesteps, args.window),
                                              args.n_reads, wc)
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
                                             args.n_reads, wc)
        print("\n--- HDF5 (chunked) ---")
        all_results["hdf5"] = run_multiworker(_worker_hdf5_array,
                                              (str(hdf5_path), "X", n_timesteps, args.window),
                                              args.n_reads, wc)
        print("\n--- Zarr (chunked, lz4) ---")
        all_results["zarr"] = run_multiworker(_worker_zarr_array,
                                              (str(zarr_path), "X", n_timesteps, args.window),
                                              args.n_reads, wc)
    return all_results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(all_results, domain):
    wc = sorted(next(iter(all_results.values())).keys())
    print(f"\n{'='*65}")
    print(f"SUMMARY — {domain.upper()}  (samples/sec)")
    header = f"  {'Format':<8}" + "".join(f"  {f'w={n}':>10}" for n in wc)
    print(header)
    print("-" * 65)
    for fmt, results in all_results.items():
        row = f"  {fmt:<8}" + "".join(f"  {results[n]:>10.1f}" for n in wc)
        print(row)
    best = max(all_results, key=lambda f: all_results[f][max(wc)])
    print(f"\nBest at {max(wc)} workers: {best.upper()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True,
                        choices=["life_sciences", "climate", "fusion"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--n-reads",   type=int, default=1000)
    parser.add_argument("--window",    type=int, default=10,
                        help="Timestep window for climate/fusion reads")
    parser.add_argument("--skip-pack", action="store_true")
    parser.add_argument("--npz-dir",  help="Dir of per-protein NPZ (life_sciences)")
    parser.add_argument("--npz-file", help="Single large NPZ file (climate)")
    parser.add_argument("--h5-file",  help="HDF5 file (fusion)")
    args = parser.parse_args()

    check_dependencies(args.domain)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n== REDI I/O Benchmark: {args.domain} ==")
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    if args.domain == "life_sciences":
        if not args.npz_dir:
            parser.error("--npz-dir required for life_sciences")
        results = benchmark_life_sciences(args, out)
    elif args.domain == "climate":
        if not args.npz_file:
            parser.error("--npz-file required for climate")
        results = benchmark_array_domain(args, out)
    else:  # fusion
        if not args.h5_file and not args.npz_file:
            parser.error("--h5-file or --npz-file required for fusion")
        results = benchmark_array_domain(args, out)

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
