#!/bin/bash
# ── REDI Benchmark: OC22 / HydraGNN LMDB ─────────────────────────────────────
#SBATCH -A stf053
#SBATCH -J redi_bench_oc22
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -t 08:00:00
#SBATCH -o /autofs/nccs-svm1_home1/whbrewer/drai/redi/benchmarks/jobs/logs/bench_oc22_%j.out
#SBATCH -e /autofs/nccs-svm1_home1/whbrewer/drai/redi/benchmarks/jobs/logs/bench_oc22_%j.err

# Full OC22 train split: 41 shards × ~200 k structures = ~8.2 M structures (69 GiB).
# Expected wall-clock: 3–8 h depending on Lustre throughput.
#
# To run a quick test first:
#   sbatch --time=00:30:00 benchmarks/jobs/job_oc22.sh  (with --max-samples 500000)

set -euo pipefail

REDI_DIR="/autofs/nccs-svm1_home1/whbrewer/drai/redi"
VENV="$HOME/venv/redi/bin/activate"
SAVE_DIR="/lustre/orion/stf053/scratch/$USER/redi_oc22_bench_output"

echo "=== REDI OC22 Benchmark  $(date) ==="
echo "  Node:    $(hostname)"
echo "  Job ID:  $SLURM_JOB_ID"

source "$VENV"
cd "$REDI_DIR"

mkdir -p benchmarks/jobs/logs benchmarks/results "$SAVE_DIR"

python benchmarks/runners/run_oc22_bench.py \
    --split train \
    --save-dir "$SAVE_DIR" \
    --out benchmarks/results/bench_oc22.json

echo "=== Done  $(date) ==="
