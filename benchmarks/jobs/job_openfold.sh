#!/bin/bash
# ── REDI Benchmark: OpenFold / OpenProteinSet ─────────────────────────────────
#SBATCH -A stf053
#SBATCH -J redi_bench_openfold
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -t 04:00:00
#SBATCH -o /autofs/nccs-svm1_home1/whbrewer/drai/redi/benchmarks/jobs/logs/bench_openfold_%j.out
#SBATCH -e /autofs/nccs-svm1_home1/whbrewer/drai/redi/benchmarks/jobs/logs/bench_openfold_%j.err

# 5 000 proteins from the OpenProteinSet on Lustre/Orion
# Expected wall-clock: ~30–60 min with 16 workers

set -euo pipefail

REDI_DIR="/autofs/nccs-svm1_home1/whbrewer/drai/redi"
VENV="$HOME/venv/redi/bin/activate"

echo "=== REDI OpenFold Benchmark  $(date) ==="
echo "  Node:    $(hostname)"
echo "  Job ID:  $SLURM_JOB_ID"

source "$VENV"
cd "$REDI_DIR"

mkdir -p benchmarks/jobs/logs benchmarks/results

python benchmarks/runners/run_openfold_bench.py \
    --n-proteins 5000 \
    --num-workers 16 \
    --out benchmarks/results/bench_openfold.json

echo "=== Done  $(date) ==="
