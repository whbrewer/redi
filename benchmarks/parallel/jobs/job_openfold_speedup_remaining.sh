#!/bin/bash
# REDI OpenFold Speedup Benchmark — workers 2,4,8,16,32 only
#
# Runs after job_openfold_speedup_4096 (workers=1) completes.
# Reads workers=1 baseline from existing speedup_openfold.json.
#
# Submit: sbatch --dependency=afterany:3251553 benchmarks/parallel/jobs/job_openfold_speedup_remaining.sh
#
#SBATCH -A gen053
#SBATCH -J redi-openfold-speedup-remaining
#SBATCH -o /autofs/nccs-svm1_home1/whbrewer/drai/redi/benchmarks/parallel/logs/openfold_speedup_remaining_%j.out
#SBATCH -e /autofs/nccs-svm1_home1/whbrewer/drai/redi/benchmarks/parallel/logs/openfold_speedup_remaining_%j.out
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 32
#SBATCH -t 01:00:00
#SBATCH -p batch

set -euo pipefail

REDI_DIR="/autofs/nccs-svm1_home1/whbrewer/drai/redi"
source ~/venv/redi/bin/activate
cd "$REDI_DIR"

mkdir -p benchmarks/parallel/logs benchmarks/parallel/results

echo "================================================================"
echo "REDI OpenFold Speedup Benchmark — workers 2,4,8,16,32"
echo "Node:   $(hostname)"
echo "Cores:  $(nproc)"
echo "Date:   $(date)"
echo "================================================================"

python benchmarks/parallel/runners/run_openfold_speedup.py \
    --n-proteins 4096 \
    --worker-list 2,4,8,16,32

echo ""
echo "=== Done $(date) ==="
