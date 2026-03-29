#!/bin/bash
# ── REDI Benchmark: ClimaX / TaiESM1 ──────────────────────────────────────────
#SBATCH -A stf053
#SBATCH -J redi_bench_climax
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -t 04:00:00
#SBATCH -o /autofs/nccs-svm1_home1/whbrewer/drai/redi/benchmarks/jobs/logs/bench_climax_%j.out
#SBATCH -e /autofs/nccs-svm1_home1/whbrewer/drai/redi/benchmarks/jobs/logs/bench_climax_%j.err

# 5 × 10-year TaiESM1 2m_temperature NC chunks (~30–40 GiB each)
# Expected wall-clock: ~2–3 h (dominated by regridding)

set -euo pipefail

REDI_DIR="/autofs/nccs-svm1_home1/whbrewer/drai/redi"
VENV="$HOME/venv/redi/bin/activate"

echo "=== REDI ClimaX Benchmark  $(date) ==="
echo "  Node:    $(hostname)"
echo "  Job ID:  $SLURM_JOB_ID"

source "$VENV"
cd "$REDI_DIR"

mkdir -p benchmarks/jobs/logs benchmarks/results

python benchmarks/runners/run_climax_bench.py \
    --n-files 5 \
    --out benchmarks/results/bench_climax.json

echo "=== Done  $(date) ==="
