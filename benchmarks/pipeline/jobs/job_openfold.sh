#!/bin/bash
# ── REDI Benchmark: OpenFold / OpenProteinSet ─────────────────────────────────
#SBATCH -A stf053
#SBATCH -J redi_bench_openfold
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -t 05:00:00
#SBATCH -o /autofs/nccs-svm1_home1/whbrewer/drai/redi/benchmarks/pipeline/logs/bench_openfold_%j.out
#SBATCH -e /autofs/nccs-svm1_home1/whbrewer/drai/redi/benchmarks/pipeline/logs/bench_openfold_%j.out

# Full OpenProteinSet (~131k proteins) on Lustre/Orion
# Expected wall-clock: ~3–4 h with 32 workers

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
    --num-workers 32 \
    --out benchmarks/pipeline/results/bench_openfold.json

echo "=== Done  $(date) ==="
