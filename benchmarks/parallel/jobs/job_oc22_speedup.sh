#!/bin/bash
# REDI Parallel Speedup Benchmark — OC22/HydraGNN, 800k structures
#
# Sweeps 1,2,4,8,16,32 num_workers on a fixed workload of 800,000 structures
# (32 LMDB shards × 25k each) via redi run --domain hydragnn.
# Results written to benchmarks/parallel/results/speedup_oc22.json
#
# Submit: sbatch benchmarks/parallel/jobs/job_oc22_speedup.sh
#
#SBATCH -A gen053
#SBATCH -J redi-oc22-speedup
#SBATCH -o /autofs/nccs-svm1_home1/whbrewer/drai/redi/benchmarks/parallel/logs/oc22_speedup_%j.out
#SBATCH -e /autofs/nccs-svm1_home1/whbrewer/drai/redi/benchmarks/parallel/logs/oc22_speedup_%j.out
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
echo "REDI OC22/HydraGNN Parallel Speedup Benchmark (32 shards × 5k structures)"
echo "Node:   $(hostname)"
echo "Cores:  $(nproc)"
echo "Date:   $(date)"
echo "================================================================"

python benchmarks/parallel/runners/run_oc22_speedup.py \
    --n-shards 32 \
    --max-samples-per-shard 5000 \
    --sweep

echo ""
echo "=== Done $(date) ==="
