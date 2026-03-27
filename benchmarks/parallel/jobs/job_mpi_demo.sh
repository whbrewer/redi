#!/bin/bash
# REDI MPI Demo — 2 Andes nodes, 4 ranks/node = 8 MPI ranks total
#
# Processes 16 ClimaX/CMCC NC files across 8 ranks (2 files per rank).
# After processing, rank 0:
#   - allreduce: aggregates global sample counts
#   - gather:    collects per-file metadata and NPZ paths
#   - merges all NPZ outputs into --shards output shard files
#   - writes a single metadata.json and datacard.json
#
# Outputs written to: benchmarks/parallel/results/mpi_demo/
#
# Submit: sbatch benchmarks/parallel/jobs/job_mpi_demo.sh
#
#SBATCH -A gen053
#SBATCH -J redi-mpi-demo
#SBATCH -o redi-mpi-demo-%j.out
#SBATCH -N 2
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 8
#SBATCH -t 1:00:00
#SBATCH -p batch

set -e

source ~/venv/redi/bin/activate

SUBMIT_DIR="$SLURM_SUBMIT_DIR"
OUT_DIR="$SUBMIT_DIR/benchmarks/parallel/results/mpi_demo"
FILELIST="$OUT_DIR/filelist.txt"

echo "================================================================"
echo "REDI MPI Demo"
echo "Node(s):  $(squeue -j $SLURM_JOB_ID -h -o '%N')"
echo "Ranks:    $SLURM_NTASKS  ($SLURM_NNODES nodes x $SLURM_NTASKS_PER_NODE/node)"
echo "Date:     $(date)"
echo "Output:   $OUT_DIR"
echo "================================================================"

# Build file list: first 16 ClimaX/CMCC annual temperature NC files
mkdir -p "$OUT_DIR"
python - <<'EOF'
from pathlib import Path
nc_dir = Path("/lustre/orion/stf053/world-shared/vga/projects/ClimaX/data/CMCC/raw/temperature")
files = sorted(nc_dir.glob("*.nc"))[:16]
if not files:
    raise SystemExit(f"No .nc files found in {nc_dir}")
import os
out = os.environ["FILELIST"]
with open(out, "w") as f:
    for p in files:
        f.write(str(p) + "\n")
print(f"File list: {len(files)} NC files → {out}")
EOF

echo ""
echo "--- Launching mpi_runner across $SLURM_NTASKS ranks ---"

srun python -m redi.parallel.mpi_runner \
    --filelist  "$FILELIST"    \
    --output    "$OUT_DIR"     \
    --domain    climate        \
    --format    npz            \
    --shards    4              \
    --merge

echo ""
echo "================================================================"
echo "Done in $(($SECONDS / 60)) minutes"
echo "Results:"
ls -lh "$OUT_DIR"/*.json "$OUT_DIR"/shard_*.npz 2>/dev/null || true
echo "================================================================"
