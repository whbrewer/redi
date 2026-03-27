"""
MPI-based REDI parallel runner.

Each MPI rank processes its share of input files independently through the
REDI pipeline. After processing, ranks communicate:
  - comm.gather()   — collects per-file metadata/stats from all ranks to rank 0
  - comm.allreduce() — aggregates global statistics (sample counts, feature ranges)
  - Rank 0 writes a unified DataCard and (optionally) a merged output file

Usage:
    srun python -m redi.parallel.mpi_runner --filelist files.txt --output /path/out
    mpirun -n 8 python -m redi.parallel.mpi_runner --filelist files.txt --output /path/out
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MPI-based REDI parallel runner")
    p.add_argument("--filelist", required=True,
                   help="Text file with one input path per line")
    p.add_argument("--output", required=True,
                   help="Base output directory")
    p.add_argument("--domain",
                   help="Domain override passed to redi run")
    p.add_argument("--format", dest="output_format", default="npz",
                   help="Output format: npz, hdf5, zarr (default: npz)")
    p.add_argument("--stages",
                   help="Comma-separated pipeline stages to run")
    p.add_argument("--merge", action="store_true",
                   help="Gather per-file results and write a unified DataCard on rank 0")
    p.add_argument("--style", default="simple",
                   help="redi output style: simple, rich, json (default: simple)")
    return p.parse_args()


def process_file(filepath: str, args: argparse.Namespace, rank: int) -> dict:
    """
    Run `redi run` on a single file as a subprocess.

    Returns a result dict with input path, output directory, and metadata
    extracted from the pipeline's metadata.json (if present).
    """
    output_dir = Path(args.output) / Path(filepath).stem
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["redi", "run", filepath, "-o", str(output_dir), "--style", args.style]
    if args.domain:
        cmd += ["-d", args.domain]
    if args.stages:
        cmd += ["-s", args.stages]
    if args.output_format:
        cmd += ["-f", args.output_format]

    result = {
        "input": filepath,
        "output_dir": str(output_dir),
        "rank": rank,
        "errors": [],
        "metadata": {},
    }

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            result["errors"].append(proc.stderr.strip() or f"redi exited {proc.returncode}")
            return result

        # Read metadata.json produced by SaveMetadata pipeline step
        meta_path = output_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                result["metadata"] = json.load(f)

    except Exception as e:
        result["errors"].append(str(e))

    return result


def _allreduce_stats(comm, all_results: list) -> dict:
    """
    Compute global aggregate statistics via MPI allreduce.

    Currently aggregates:
      - total_samples: sum across all files
      - global feature min/max (when per-file stats are present)

    Returns a dict of global stats on all ranks (allreduce result is symmetric).
    """
    try:
        from mpi4py import MPI
    except ImportError:
        return {}

    # Aggregate total sample count
    local_samples = sum(
        r["metadata"].get("split", {}).get("train_size", 0) +
        r["metadata"].get("split", {}).get("val_size", 0) +
        r["metadata"].get("split", {}).get("test_size", 0)
        for r in all_results
        if not r["errors"]
    )
    global_samples = comm.allreduce(local_samples, op=MPI.SUM)

    return {
        "total_samples": global_samples,
    }


def write_datacard(all_results: list, global_stats: dict, args: argparse.Namespace):
    """Write a unified DataCard JSON on rank 0."""
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
        "domain": domains[0] if len(domains) == 1 else domains,
        "total_files": len(all_results),
        "successful": len(successful),
        "failed": len(failed),
        "global_stats": global_stats,
        "failed_files": [
            {"input": r["input"], "errors": r["errors"]}
            for r in failed
        ],
        "per_file": [
            {
                "input": r["input"],
                "output_dir": r["output_dir"],
                "rank": r["rank"],
                "metadata": r["metadata"],
            }
            for r in successful
        ],
    }

    datacard_path = output_dir / "datacard.json"
    datacard_path.write_text(json.dumps(datacard, indent=2, default=str))
    print(f"[rank 0] DataCard written to {datacard_path}", flush=True)

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
            "Install with: pip install mpi4py",
            file=sys.stderr,
        )
        sys.exit(1)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    args = parse_args()

    # Rank 0 reads the file list and broadcasts to all ranks
    if rank == 0:
        filelist_path = Path(args.filelist)
        if not filelist_path.exists():
            print(f"Error: filelist not found: {filelist_path}", file=sys.stderr, flush=True)
            comm.Abort(1)
        all_files = [line.strip() for line in filelist_path.read_text().splitlines()
                     if line.strip()]
        print(f"[rank 0] {len(all_files)} files across {size} rank(s)", flush=True)
    else:
        all_files = None

    all_files = comm.bcast(all_files, root=0)

    # Each rank takes every size-th file starting from its rank (round-robin)
    my_files = [f for i, f in enumerate(all_files) if i % size == rank]
    print(f"[rank {rank}] processing {len(my_files)} file(s)", flush=True)

    # Process files — each rank is independent here
    my_results = [process_file(f, args, rank) for f in my_files]

    local_errors = sum(1 for r in my_results if r["errors"])
    print(f"[rank {rank}] done — {len(my_results) - local_errors} ok, {local_errors} failed",
          flush=True)

    # Allreduce for global statistics (symmetric result on all ranks)
    global_stats = _allreduce_stats(comm, my_results)

    if not args.merge:
        return

    # Gather all results to rank 0 for DataCard
    all_results_nested = comm.gather(my_results, root=0)

    if rank == 0:
        all_results = [r for rank_results in all_results_nested for r in rank_results]
        # Sort by original file order for deterministic DataCard output
        order = {f: i for i, f in enumerate(all_files)}
        all_results.sort(key=lambda r: order.get(r["input"], 9999))
        write_datacard(all_results, global_stats, args)


if __name__ == "__main__":
    main()
