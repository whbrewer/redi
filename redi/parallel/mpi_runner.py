"""
MPI-based REDI parallel runner.

Each MPI rank processes its share of input files independently through the
REDI pipeline. After processing, ranks communicate:
  - comm.allreduce()  — aggregates global statistics (sample counts) across all ranks
  - comm.gather()     — collects per-file metadata/paths from all ranks to rank 0
  - Rank 0 merges per-file NPZ outputs into N output shards and writes a unified
    metadata.json and datacard.json

Usage:
    # Point at a directory (rank 0 discovers files):
    srun python -m redi.parallel.mpi_runner --input /data/nc/ --output /path/out --domain climate --merge
    srun python -m redi.parallel.mpi_runner --input /data/nc/ --output /path/out --n-files 16 --shards 4 --merge

    # Use a pre-built file list (compatible with gnu backend):
    srun python -m redi.parallel.mpi_runner --filelist files.txt --output /path/out --merge

    # Pass a YAML config to each redi run subprocess (e.g. for domain-specific settings):
    srun python -m redi.parallel.mpi_runner --input /data/xgc1/ --output /path/out --domain xgc1 -c bench_xgc1.yaml --merge
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# PMI/SLURM env vars that cause child processes linked against libmpi to try
# to initialize MPI.  Strip them so `redi run` subprocesses stay serial.
_PMI_VARS = {"PMI_FD", "PMI_RANK", "PMI_SIZE", "PMI_PORT",
             "SLURM_MPI_TYPE", "SLURM_STEP_LAUNCHER_PORT"}
_PMI_PREFIXES = ("PMI2_", "PMIX_", "OMPI_MCA_orte", "OMPI_MCA_plm")


def _child_env() -> dict:
    """Return os.environ minus any PMI/MPI init vars."""
    return {k: v for k, v in os.environ.items()
            if k not in _PMI_VARS and not any(k.startswith(p) for p in _PMI_PREFIXES)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MPI-based REDI parallel runner")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--filelist",
                     help="Text file with one input path per line (used by gnu backend)")
    src.add_argument("--input",
                     help="Input directory or glob pattern (e.g. /data/nc/ or '/data/*.nc')")
    p.add_argument("--output", required=True,
                   help="Base output directory")
    p.add_argument("--n-files", type=int, default=None, metavar="N",
                   help="Limit to first N files when using --input (default: all)")
    p.add_argument("--glob", default=None,
                   help="Glob pattern when --input is a directory (default: auto by domain)")
    p.add_argument("--domain",
                   help="Domain override passed to redi run")
    p.add_argument("--format", dest="output_format", default="npz",
                   help="Output format: npz, hdf5, zarr (default: npz)")
    p.add_argument("--stages",
                   help="Comma-separated pipeline stages to run")
    p.add_argument("--merge", action="store_true",
                   help="Gather per-rank results on rank 0 and write unified outputs")
    p.add_argument("--shards", type=int, default=1, metavar="N",
                   help="Number of output shard files to write on rank 0 (default: 1)")
    p.add_argument("--style", default="simple",
                   help="redi output style: simple, rich, json (default: simple)")
    p.add_argument("--provenance", action="store_true",
                   help="Enable FlowCept provenance tracking per file; merge buffers on rank 0")
    p.add_argument("-c", "--config",
                   help="YAML config file passed to each redi run subprocess")
    return p.parse_args()


# Default glob patterns per domain (used when --input is a directory)
_DOMAIN_GLOBS = {
    "climate":      "*.nc",
    "fusion":       "*.bp",
    "materials":    "*.lmdb",
    "life_sciences": "*",   # protein directories
    "hpc":          "*.csv",
    "geospatial":   "*.geojson",
    # xgc1: case subdirectories identified by presence of fort.input.used
    "xgc1":         "__xgc1_cases__",
}


def _discover_xgc1_cases(input_path: Path) -> list:
    """Return subdirectories that look like XGC1 cases (contain fort.input.used)."""
    cases = sorted(
        str(p) for p in input_path.iterdir()
        if p.is_dir() and (p / "fort.input.used").exists()
    )
    return cases


def discover_files(args: argparse.Namespace) -> list:
    """
    Rank 0: return the list of input file paths from either --filelist or --input.
    """
    import glob as _glob

    if args.filelist:
        paths = [line.strip() for line in Path(args.filelist).read_text().splitlines()
                 if line.strip()]
        return paths

    # --input: directory or glob pattern
    input_path = Path(args.input)
    if input_path.is_dir():
        pattern = args.glob or _DOMAIN_GLOBS.get(args.domain or "", "*")
        if pattern == "__xgc1_cases__":
            paths = _discover_xgc1_cases(input_path)
        else:
            paths = sorted(str(p) for p in input_path.glob(pattern))
    else:
        paths = sorted(_glob.glob(str(args.input)))

    if args.n_files:
        paths = paths[: args.n_files]

    return paths


def process_file(filepath: str, args: argparse.Namespace, rank: int) -> dict:
    """
    Run `redi run` on a single file as a subprocess.

    Returns a result dict with input path, output directory, paths to any
    NPZ files written, and metadata extracted from metadata.json (if present).
    """
    output_dir = Path(args.output) / "per_rank" / f"rank{rank:04d}" / Path(filepath).stem
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["redi", "run", filepath, "-o", str(output_dir), "--style", args.style]
    if args.domain:
        cmd += ["-d", args.domain]
    if args.stages:
        cmd += ["-s", args.stages]
    if args.output_format:
        cmd += ["-f", args.output_format]
    if getattr(args, "provenance", False):
        cmd += ["--provenance"]
    if getattr(args, "config", None):
        cmd += ["-c", args.config]

    result = {
        "input": filepath,
        "output_dir": str(output_dir),
        "rank": rank,
        "npz_files": [],
        "errors": [],
        "metadata": {},
        "flowcept_buffer": None,
    }

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=_child_env())
        if proc.returncode != 0:
            result["errors"].append(proc.stderr.strip() or f"redi exited {proc.returncode}")
            return result

        # Collect any NPZ files written by the pipeline
        result["npz_files"] = [str(p) for p in sorted(output_dir.glob("*.npz"))]

        # Read metadata.json produced by SaveMetadata pipeline step
        meta_path = output_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                result["metadata"] = json.load(f)

        # Capture FlowCept buffer path if provenance was enabled
        buffer_path = output_dir / "flowcept_buffer.jsonl"
        if buffer_path.exists():
            result["flowcept_buffer"] = str(buffer_path)

    except Exception as e:
        result["errors"].append(str(e))

    return result


def _allreduce_stats(comm, my_results: list) -> dict:
    """
    Compute global aggregate statistics via MPI allreduce (symmetric on all ranks).

    Aggregates:
      - total_samples: sum of train+val+test sample counts across all files/ranks
    """
    from mpi4py import MPI

    local_samples = sum(
        r["metadata"].get("split", {}).get("train_size", 0) +
        r["metadata"].get("split", {}).get("val_size", 0) +
        r["metadata"].get("split", {}).get("test_size", 0)
        for r in my_results
        if not r["errors"]
    )
    global_samples = comm.allreduce(local_samples, op=MPI.SUM)

    local_ok = sum(1 for r in my_results if not r["errors"])
    global_ok = comm.allreduce(local_ok, op=MPI.SUM)

    local_failed = sum(1 for r in my_results if r["errors"])
    global_failed = comm.allreduce(local_failed, op=MPI.SUM)

    return {
        "total_samples": global_samples,
        "successful": global_ok,
        "failed": global_failed,
    }


def merge_npz_shards(all_results: list, output_dir: Path, n_shards: int) -> list:
    """
    Load per-file NPZ outputs gathered from all ranks, concatenate, and write
    N evenly-sized shard files to output_dir.

    Returns list of shard file Paths that were written.
    """
    import numpy as np

    npz_paths = []
    for r in all_results:
        if r["errors"]:
            continue
        for p in r.get("npz_files", []):
            if Path(p).exists():
                npz_paths.append(Path(p))

    if not npz_paths:
        print("[rank 0] No NPZ files found — skipping merge", flush=True)
        return []

    print(f"[rank 0] Merging {len(npz_paths)} NPZ file(s) into {n_shards} shard(s) ...",
          flush=True)

    # Load and concatenate all arrays
    arrays: dict = {}
    for npz_path in npz_paths:
        data = np.load(npz_path, allow_pickle=True)
        for key in data.files:
            if key not in arrays:
                arrays[key] = []
            arrays[key].append(data[key])

    merged = {k: np.concatenate(v, axis=0) for k, v in arrays.items()}
    total = len(next(iter(merged.values())))

    # Shard and write
    shard_size = max(1, (total + n_shards - 1) // n_shards)
    shard_paths = []
    for i in range(n_shards):
        start = i * shard_size
        if start >= total:
            break
        end = min(start + shard_size, total)
        shard = {k: v[start:end] for k, v in merged.items()}
        shard_path = output_dir / f"shard_{i:04d}.npz"
        np.savez_compressed(shard_path, **shard)
        shard_paths.append(shard_path)
        print(f"[rank 0]   {shard_path.name}  ({end - start} samples)", flush=True)

    return shard_paths


def merge_provenance(all_results: list, output_dir: Path) -> dict:
    """
    Merge per-file flowcept_buffer.jsonl files into one and generate
    a unified provenance card + timing summary (rank 0 only).

    Returns a dict with keys: buffer_path, timing, card_path, error.
    """
    from redi.provenance_card import (
        aggregate_parallel_timings,
        generate_provenance_artifacts,
        merge_flowcept_buffers,
    )

    buffer_paths = [
        r["flowcept_buffer"]
        for r in all_results
        if r.get("flowcept_buffer")
    ]
    if not buffer_paths:
        return {"buffer_path": None, "timing": None, "card_path": None,
                "error": "no flowcept buffers found"}

    merged_path = output_dir / "flowcept_buffer.jsonl"
    n_lines = merge_flowcept_buffers(buffer_paths, merged_path)
    print(f"[rank 0] merged {len(buffer_paths)} FlowCept buffer(s) → {n_lines} records", flush=True)

    timing = aggregate_parallel_timings(merged_path)
    print(f"[rank 0] timing: wall={timing['wall_time_s']:.1f}s  "
          f"compute={timing['total_compute_s']:.1f}s  tasks={timing['n_tasks']}", flush=True)
    for stage, s in timing["per_stage"].items():
        print(f"[rank 0]   {stage:20s}  n={s['count']}  "
              f"mean={s['mean_s']:.2f}s  max={s['max_s']:.2f}s", flush=True)

    try:
        artifacts = generate_provenance_artifacts(output_dir=output_dir)
        card_path = artifacts.get("card_file")
        print(f"[rank 0] PROVENANCE_CARD.md → {card_path}", flush=True)
    except Exception as exc:
        card_path = None
        print(f"[rank 0] WARNING: could not generate provenance card: {exc}", flush=True)

    return {"buffer_path": str(merged_path), "timing": timing, "card_path": card_path,
            "error": None}


def write_metadata(all_results: list, global_stats: dict,
                   shard_paths: list, args: argparse.Namespace,
                   n_ranks: int):
    """Write a single metadata.json to the output directory (rank 0 only)."""
    output_dir = Path(args.output)

    successful = [r for r in all_results if not r["errors"]]
    domains = list({r["metadata"].get("domain") for r in successful
                    if r["metadata"].get("domain")})

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "backend": "mpi",
        "mpi_ranks": n_ranks,
        "domain": domains[0] if len(domains) == 1 else (domains or None),
        "output_format": args.output_format,
        "total_files": len(all_results),
        "successful": global_stats.get("successful", len(successful)),
        "failed": global_stats.get("failed", len(all_results) - len(successful)),
        "total_samples": global_stats.get("total_samples", 0),
        "shards": len(shard_paths),
        "shard_files": [str(p) for p in shard_paths],
        "global_stats": global_stats,
    }

    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, default=str))
    print(f"[rank 0] metadata.json  → {meta_path}", flush=True)
    return meta_path


def write_datacard(all_results: list, global_stats: dict,
                   shard_paths: list, args: argparse.Namespace):
    """Write a unified datacard.json to the output directory (rank 0 only)."""
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    successful = [r for r in all_results if not r["errors"]]
    failed = [r for r in all_results if r["errors"]]

    domains = list({r["metadata"].get("domain") for r in successful
                    if r["metadata"].get("domain")})

    datacard = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "backend": "mpi",
        "output_format": args.output_format,
        "domain": domains[0] if len(domains) == 1 else (domains or None),
        "total_files": len(all_results),
        "successful": len(successful),
        "failed": len(failed),
        "global_stats": global_stats,
        "shards": len(shard_paths),
        "shard_files": [str(p) for p in shard_paths],
        "failed_files": [
            {"input": r["input"], "errors": r["errors"]}
            for r in failed
        ],
        "per_file": [
            {
                "input": r["input"],
                "output_dir": r["output_dir"],
                "rank": r["rank"],
                "npz_files": r["npz_files"],
                "metadata": r["metadata"],
            }
            for r in successful
        ],
    }

    datacard_path = output_dir / "datacard.json"
    datacard_path.write_text(json.dumps(datacard, indent=2, default=str))
    print(f"[rank 0] datacard.json  → {datacard_path}", flush=True)

    if failed:
        print(f"[rank 0] {len(failed)} file(s) failed:", flush=True)
        for r in failed:
            print(f"  {r['input']}: {r['errors']}", flush=True)


def main():
    try:
        from mpi4py import MPI
    except ImportError:
        print(
            "Error: mpi4py is required for the MPI backend.\n"
            "Install with: MPICC=cc pip install --no-binary mpi4py mpi4py",
            file=sys.stderr,
        )
        sys.exit(1)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    args = parse_args()

    # ── Rank 0: discover files and broadcast ────────────────────────────
    if rank == 0:
        all_files = discover_files(args)
        if not all_files:
            print(f"Error: no files found for input={args.input!r}", file=sys.stderr, flush=True)
            comm.Abort(1)
        print(f"[rank 0] {len(all_files)} file(s) discovered → {size} rank(s), "
              f"shards={args.shards}, merge={args.merge}", flush=True)
    else:
        all_files = None

    all_files = comm.bcast(all_files, root=0)

    # ── Round-robin scatter: each rank takes every size-th file ─────────
    my_files = [f for i, f in enumerate(all_files) if i % size == rank]
    print(f"[rank {rank}] processing {len(my_files)} file(s)", flush=True)

    # ── Process files — ranks are independent here ──────────────────────
    my_results = [process_file(f, args, rank) for f in my_files]

    local_errors = sum(1 for r in my_results if r["errors"])
    print(f"[rank {rank}] done — {len(my_results) - local_errors} ok, "
          f"{local_errors} failed", flush=True)

    # ── Allreduce: global statistics (symmetric on all ranks) ────────────
    global_stats = _allreduce_stats(comm, my_results)

    if not args.merge:
        if rank == 0:
            print(f"[rank 0] global_stats = {global_stats}", flush=True)
        return

    # ── Gather all results to rank 0 ────────────────────────────────────
    all_results_nested = comm.gather(my_results, root=0)

    if rank == 0:
        # Flatten and restore original file order
        all_results = [r for rank_results in all_results_nested for r in rank_results]
        order = {f: i for i, f in enumerate(all_files)}
        all_results.sort(key=lambda r: order.get(r["input"], 9999))

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Merge per-file NPZ outputs → N shard files
        shard_paths = []
        if args.output_format == "npz":
            shard_paths = merge_npz_shards(all_results, output_dir, args.shards)

        # Write unified metadata.json and datacard.json
        write_metadata(all_results, global_stats, shard_paths, args, size)
        write_datacard(all_results, global_stats, shard_paths, args)

        # Merge FlowCept provenance buffers if provenance was enabled
        if getattr(args, "provenance", False):
            merge_provenance(all_results, output_dir)

        print(f"\n[rank 0] == All done ==", flush=True)
        print(f"[rank 0]   output: {output_dir}", flush=True)
        if shard_paths:
            print(f"[rank 0]   shards: {len(shard_paths)} × {output_dir.name}/shard_NNNN.npz",
                  flush=True)


if __name__ == "__main__":
    main()
