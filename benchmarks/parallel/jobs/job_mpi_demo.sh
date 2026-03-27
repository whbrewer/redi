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
# Outputs written to: /lustre/orion/scratch/whbrewer/stf053/redi-mpi-demo-$SLURM_JOB_ID/
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

NC_DIR="/lustre/orion/stf053/world-shared/vga/projects/ClimaX/data/CMCC/raw/temperature"
OUT_DIR="/lustre/orion/scratch/whbrewer/stf053/redi-mpi-demo-$SLURM_JOB_ID"

echo "================================================================"
echo "REDI MPI Demo"
echo "Node(s):  $(squeue -j $SLURM_JOB_ID -h -o '%N')"
echo "Ranks:    $SLURM_NTASKS  ($SLURM_NNODES nodes x $SLURM_NTASKS_PER_NODE/node)"
echo "Date:     $(date)"
echo "Input:    $NC_DIR"
echo "Output:   $OUT_DIR"
echo "================================================================"

mkdir -p "$OUT_DIR"

echo ""
echo "--- Launching mpi_runner across $SLURM_NTASKS ranks ---"

srun python -m redi.parallel.mpi_runner \
    --input     "$NC_DIR"      \
    --output    "$OUT_DIR"     \
    --n-files   16             \
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
