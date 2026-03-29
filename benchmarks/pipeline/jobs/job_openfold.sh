#!/bin/bash
# ── REDI Benchmark: OpenFold / OpenProteinSet ─────────────────────────────────
#SBATCH -A stf053
#SBATCH -J redi_bench_openfold
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -t 02:00:00
#SBATCH -o /autofs/nccs-svm1_home1/whbrewer/drai/redi/benchmarks/pipeline/logs/bench_openfold_%j.out
#SBATCH -e /autofs/nccs-svm1_home1/whbrewer/drai/redi/benchmarks/pipeline/logs/bench_openfold_%j.out

# 5 000 proteins from the OpenProteinSet on Lustre/Orion
# Expected wall-clock: ~15–30 min with 32 workers

set -euo pipefail

REDI_DIR="/autofs/nccs-svm1_home1/whbrewer/drai/redi"
VENV="$HOME/venv/redi/bin/activate"

echo "=== REDI OpenFold Benchmark  $(date) ==="
echo "  Node:    $(hostname)"
echo "  Job ID:  $SLURM_JOB_ID"

source "$VENV"
cd "$REDI_DIR"

mkdir -p benchmarks/pipeline/logs benchmarks/pipeline/results

python benchmarks/pipeline/runners/run_openfold_bench.py \
    --n-proteins 5000 \
    --num-workers 32 \
    --out benchmarks/pipeline/results/bench_openfold.json

echo "=== Done  $(date) ==="
