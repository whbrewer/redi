#!/bin/bash
# REDI Parallel Speedup Benchmark — OpenFold, 4096 proteins
#
# Sweeps 1,2,4,8,16,32 internal workers on a fixed workload of 4096 proteins.
# Uses redi run OPS_ROOT --workers W (same code path as pipeline benchmark).
# Results written to benchmarks/parallel/results/speedup_openfold.json
#
# Submit: sbatch benchmarks/parallel/jobs/job_openfold_speedup_4096.sh
#
#SBATCH -A gen053
#SBATCH -J redi-openfold-speedup-4096
#SBATCH -o /autofs/nccs-svm1_home1/whbrewer/drai/redi/benchmarks/parallel/logs/openfold_speedup_4096_%j.out
#SBATCH -e /autofs/nccs-svm1_home1/whbrewer/drai/redi/benchmarks/parallel/logs/openfold_speedup_4096_%j.out
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 32
#SBATCH -t 04:00:00
#SBATCH -p batch

set -euo pipefail

REDI_DIR="/autofs/nccs-svm1_home1/whbrewer/drai/redi"
source ~/venv/redi/bin/activate
cd "$REDI_DIR"

mkdir -p benchmarks/parallel/logs benchmarks/parallel/results

echo "================================================================"
echo "REDI OpenFold Parallel Speedup Benchmark (4096 proteins)"
echo "Node:   $(hostname)"
echo "Cores:  $(nproc)"
echo "Date:   $(date)"
echo "================================================================"

python benchmarks/parallel/runners/run_openfold_speedup.py \
    --n-proteins 4096 \
    --sweep

echo ""
echo "=== Done $(date) ==="
