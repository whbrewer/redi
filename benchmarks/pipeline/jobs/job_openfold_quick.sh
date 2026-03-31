#!/bin/bash
# ── REDI Benchmark: OpenFold quick substep timing probe ───────────────────────
#SBATCH -A stf053
#SBATCH -J redi_bench_openfold_quick
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -t 00:30:00
#SBATCH -o /autofs/nccs-svm1_home1/whbrewer/drai/redi/benchmarks/pipeline/logs/bench_openfold_quick_%j.out
#SBATCH -e /autofs/nccs-svm1_home1/whbrewer/drai/redi/benchmarks/pipeline/logs/bench_openfold_quick_%j.out

# 5 000 proteins, 32 workers — for per-substep timing breakdown
# (cif_parse / msa_parse / encode / save)

set -euo pipefail

REDI_DIR="/autofs/nccs-svm1_home1/whbrewer/drai/redi"
VENV="$HOME/venv/redi/bin/activate"

echo "=== REDI OpenFold Quick Timing Probe  $(date) ==="
echo "  Node:    $(hostname)"
echo "  Job ID:  $SLURM_JOB_ID"

source "$VENV"
cd "$REDI_DIR"

mkdir -p benchmarks/pipeline/logs benchmarks/pipeline/results

python benchmarks/pipeline/runners/run_openfold_bench.py \
    --n-proteins 5000 \
    --num-workers 32 \
    --out benchmarks/pipeline/results/bench_openfold_quick.json

echo "=== Done  $(date) ==="
