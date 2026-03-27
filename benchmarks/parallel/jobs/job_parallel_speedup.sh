#!/bin/bash
# REDI Parallel Speedup Benchmark — single Andes node, up to 32 workers
#
# Runs the speedup sweep for each case (ClimaX, XGC1, OpenFold).
# Results written to benchmarks/parallel/results/.
#
# Submit: sbatch benchmarks/parallel/jobs/job_parallel_speedup.sh
#
#SBATCH -A gen053
#SBATCH -J redi-parallel-speedup
#SBATCH -o redi-parallel-speedup-%j.out
#SBATCH -e redi-parallel-speedup-%j.err
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 32
#SBATCH -t 8:00:00
#SBATCH -p batch

set -e

REDI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source ~/venv/redi/bin/activate

echo "================================================================"
echo "REDI Parallel Speedup Benchmark"
echo "Node:   $(hostname)"
echo "Cores:  $(nproc)"
echo "Date:   $(date)"
echo "================================================================"

# ── ClimaX / TaiESM1 ────────────────────────────────────────────────
# 17 NC files; sweep 1,2,4,8,16 workers
echo ""
echo "--- ClimaX ---"
python "$REDI_ROOT/parallel/runners/run_climax_speedup.py" --sweep

# ── XGC1 / ITER ─────────────────────────────────────────────────────
# 32 .bp snapshots (~5.5 GiB each); sweep 1,2,4,8,16,32 workers
echo ""
echo "--- XGC1 ---"
python "$REDI_ROOT/parallel/runners/run_xgc1_speedup.py" --n-files 32 --sweep

# ── OpenFold / OpenProteinSet ────────────────────────────────────────
# 128 protein directories; sweep 1,2,4,8,16,32 workers
echo ""
echo "--- OpenFold ---"
python "$REDI_ROOT/parallel/runners/run_openfold_speedup.py" --n-proteins 128 --sweep

echo ""
echo "================================================================"
echo "All done in $(($SECONDS / 60)) minutes"
echo "Results in: $REDI_ROOT/parallel/results/"
echo "================================================================"
