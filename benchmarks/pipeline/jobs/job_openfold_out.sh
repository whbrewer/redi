#!/bin/bash
# ── REDI Output Run: OpenFold / OpenProteinSet ────────────────────────────────
# Runs the full REDI pipeline on the OpenProteinSet (5000 proteins by default)
# and writes DATA_CARD.md, metadata.json, and HDF5 output to shared Lustre path.
#SBATCH -A stf053
#SBATCH -J redi_out_openfold
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -t 05:00:00
#SBATCH -o /autofs/nccs-svm1_home1/whbrewer/drai/redi/benchmarks/pipeline/logs/out_openfold_%j.out
#SBATCH -e /autofs/nccs-svm1_home1/whbrewer/drai/redi/benchmarks/pipeline/logs/out_openfold_%j.out

set -euo pipefail

REDI_DIR="/autofs/nccs-svm1_home1/whbrewer/drai/redi"
VENV="$HOME/venv/redi/bin/activate"
OPS_ROOT="/lustre/orion/gen053/world-shared/openfold/data"
OUT_DIR="/lustre/orion/proj-shared/stf249/redi/outputs/openfold"
CFG="$REDI_DIR/benchmarks/pipeline/configs/bench_openfold.yaml"

echo "=== REDI OpenFold Output Run  $(date) ==="
echo "  Node:    $(hostname)"
echo "  Job ID:  $SLURM_JOB_ID"
echo "  Output:  $OUT_DIR"

source "$VENV"
cd "$REDI_DIR"

mkdir -p benchmarks/pipeline/logs "$OUT_DIR"

redi run "$OPS_ROOT" \
    --domain life_sciences \
    -c "$CFG" \
    -o "$OUT_DIR" \
    -v

echo "=== Done  $(date) ==="
echo "Output files:"
ls -lh "$OUT_DIR" 2>/dev/null || true
