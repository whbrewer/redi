#!/bin/bash
# ── REDI Output Run: XGC1 ITER ────────────────────────────────────────────────
# Runs the full REDI pipeline on the XGC1 ITER case (5 timesteps) and writes
# READINESS_REPORT.md, metadata.json, and PT output to the shared Lustre path.
#SBATCH -A stf053
#SBATCH -J redi_out_xgc1
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -t 02:00:00
#SBATCH -o /autofs/nccs-svm1_home1/whbrewer/drai/redi/benchmarks/pipeline/logs/out_xgc1_%j.out
#SBATCH -e /autofs/nccs-svm1_home1/whbrewer/drai/redi/benchmarks/pipeline/logs/out_xgc1_%j.out

set -euo pipefail

REDI_DIR="/autofs/nccs-svm1_home1/whbrewer/drai/redi"
VENV="$HOME/venv/redi/bin/activate"
XGC1_ROOT="/lustre/orion/world-shared/fus184/jyc/fusiond-seed-xgc1-data-samples"
OUT_DIR="/lustre/orion/proj-shared/stf249/redi/outputs/xgc1"
CFG="$REDI_DIR/benchmarks/pipeline/configs/bench_xgc1_out.yaml"

echo "=== REDI XGC1 Output Run  $(date) ==="
echo "  Node:    $(hostname)"
echo "  Job ID:  $SLURM_JOB_ID"
echo "  Output:  $OUT_DIR"

source "$VENV"
cd "$REDI_DIR"

mkdir -p benchmarks/pipeline/logs "$OUT_DIR"

redi run "$XGC1_ROOT" \
    --domain xgc1 \
    -c "$CFG" \
    -o "$OUT_DIR" \
    -v

echo "=== Done  $(date) ==="
echo "Output files:"
ls -lh "$OUT_DIR" 2>/dev/null || true
