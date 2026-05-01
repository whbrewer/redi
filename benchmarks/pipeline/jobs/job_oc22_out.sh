#!/bin/bash
# ── REDI Output Run: OC22 / HydraGNN ─────────────────────────────────────────
# Runs the full REDI pipeline via `redi run` on the OC22 LMDB train split and
# writes data.npz, DATA_CARD.md, and metadata.json to the shared Lustre path.
#SBATCH -A stf053
#SBATCH -J redi_out_oc22
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -t 24:00:00
#SBATCH -o /autofs/nccs-svm1_home1/whbrewer/drai/redi/benchmarks/pipeline/logs/out_oc22_%j.out
#SBATCH -e /autofs/nccs-svm1_home1/whbrewer/drai/redi/benchmarks/pipeline/logs/out_oc22_%j.out

set -euo pipefail

REDI_DIR="/autofs/nccs-svm1_home1/whbrewer/drai/redi"
VENV="$HOME/venv/redi/bin/activate"
LMDB_SPLIT="/lustre/orion/gen053/world-shared/HydraGNN/data/oc22/s2ef_total_train_val_test_lmdbs/data/oc22/s2ef-total/train"
OUT_DIR="/lustre/orion/proj-shared/stf249/redi/outputs/oc22"
CFG="$REDI_DIR/benchmarks/pipeline/configs/bench_oc22.yaml"

echo "=== REDI OC22 Output Run  $(date) ==="
echo "  Node:    $(hostname)"
echo "  Job ID:  $SLURM_JOB_ID"
echo "  Input:   $LMDB_SPLIT"
echo "  Output:  $OUT_DIR"

source "$VENV"
cd "$REDI_DIR"

mkdir -p benchmarks/pipeline/logs "$OUT_DIR"

redi run "$LMDB_SPLIT" \
    --domain materials \
    -c "$CFG" \
    -o "$OUT_DIR" \
    -v

echo "=== Done  $(date) ==="
echo "Output files:"
ls -lh "$OUT_DIR" 2>/dev/null || true
