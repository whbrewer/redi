#!/bin/bash
# ── REDI Benchmark: XGC1 ITER — xgc1 plugin ───────────────────────────────────
#SBATCH -A stf053
#SBATCH -J redi_bench_xgc1
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -t 02:00:00
#SBATCH -o /autofs/nccs-svm1_home1/whbrewer/drai/redi/benchmarks/pipeline/logs/bench_xgc1_%j.out
#SBATCH -e /autofs/nccs-svm1_home1/whbrewer/drai/redi/benchmarks/pipeline/logs/bench_xgc1_%j.out

# 5 × xgc.3d timestep snapshots from n560fr_ITER_PFPO_W_Ne (~5.5 GiB each, ~27.5 GiB total)
# Uses the redi-xgc1 plugin (--domain xgc1) instead of the built-in fusion domain

set -euo pipefail

REDI_DIR="/autofs/nccs-svm1_home1/whbrewer/drai/redi"
VENV="$HOME/venv/redi/bin/activate"

echo "=== REDI XGC1 Plugin Benchmark  $(date) ==="
echo "  Node:    $(hostname)"
echo "  Job ID:  $SLURM_JOB_ID"

source "$VENV"
cd "$REDI_DIR"

mkdir -p benchmarks/pipeline/logs benchmarks/pipeline/results

python benchmarks/pipeline/runners/run_xgc1_bench.py \
    --n-timesteps 5 \
    --out benchmarks/pipeline/results/bench_xgc1.json

echo "=== Done  $(date) ==="
