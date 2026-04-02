#!/bin/bash
# REDI MPI — XGC1 timestep-level parallelism, 8 nodes, 64 ranks
#
# Distributes 64 timestep .bp directories from n560fr_ITER_PFPO_W_Ne across
# 64 MPI ranks (8/node). Each rank processes one timestep independently:
#   redi run <case_dir>/xgc.f3d.NNNNN.bp --domain xgc1 -c <cfg>
#
# 8 tasks/node keeps memory pressure low for the large xgc.f3d fields.
# Outputs written to: /lustre/orion/scratch/whbrewer/stf053/redi-xgc1-mpi-$SLURM_JOB_ID/
#
# Submit: sbatch benchmarks/parallel/jobs/job_xgc1_mpi.sh
#
#SBATCH -A stf053
#SBATCH -J redi-xgc1-mpi
#SBATCH -o redi-xgc1-mpi-%j.out
#SBATCH -N 8
#SBATCH --ntasks-per-node 8
#SBATCH --cpus-per-task 1
#SBATCH -t 2:00:00
#SBATCH -p batch

set -e

source ~/venv/redi/bin/activate

XGC1_CASE="/lustre/orion/world-shared/fus184/jyc/fusiond-seed-xgc1-data-samples/n560fr_ITER_PFPO_W_Ne"
OUT_DIR="/lustre/orion/scratch/whbrewer/stf053/redi-xgc1-mpi-$SLURM_JOB_ID"
CFG="$OUT_DIR/bench_xgc1_mpi.yaml"

mkdir -p "$OUT_DIR"

cat > "$CFG" <<EOF
domain: xgc1
output_format: pt
EOF

echo "================================================================"
echo "REDI MPI — XGC1 timestep-level parallelism"
echo "Node(s):  $SLURM_NODELIST"
echo "Ranks:    $SLURM_NTASKS  ($SLURM_NNODES nodes x $SLURM_NTASKS_PER_NODE/node)"
echo "Date:     $(date)"
echo "Input:    $XGC1_CASE"
echo "Output:   $OUT_DIR"
echo "================================================================"

srun python -m redi.parallel.mpi_runner \
    --input   "$XGC1_CASE" \
    --output  "$OUT_DIR"   \
    --domain  xgc1         \
    --n-files 64           \
    --config  "$CFG"       \
    --merge

rm -f "$CFG"

echo ""
echo "================================================================"
echo "Done in $(($SECONDS / 60)) minutes"
echo "Results:"
ls -lh "$OUT_DIR"/*.json 2>/dev/null || true
echo "================================================================"
