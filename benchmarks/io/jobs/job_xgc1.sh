#!/bin/bash
# ── REDI Benchmark: XGC1 ITER fusion snapshots ────────────────────────────────
#SBATCH -A stf053
#SBATCH -J redi_bench_xgc1
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -t 02:00:00
#SBATCH -o /autofs/nccs-svm1_home1/whbrewer/drai/redi/benchmarks/jobs/logs/bench_xgc1_%j.out
#SBATCH -e /autofs/nccs-svm1_home1/whbrewer/drai/redi/benchmarks/jobs/logs/bench_xgc1_%j.err

# 5 × xgc.3d snapshots (~5.5 GiB each, ~27.5 GiB total)
# ADIOS2 BP5 format — adios2 Python wheel installed in venv

set -euo pipefail

REDI_DIR="/autofs/nccs-svm1_home1/whbrewer/drai/redi"
VENV="$HOME/venv/redi/bin/activate"

echo "=== REDI XGC1 Benchmark  $(date) ==="
echo "  Node:    $(hostname)"
echo "  Job ID:  $SLURM_JOB_ID"

source "$VENV"
cd "$REDI_DIR"

mkdir -p benchmarks/jobs/logs benchmarks/results

python benchmarks/runners/run_xgc1_bench.py \
    --n-files 5 \
    --out benchmarks/results/bench_xgc1.json

echo "=== Done  $(date) ==="
