#!/bin/bash
# ── REDI Output Run: ClimaX / TaiESM1 ─────────────────────────────────────────
# Runs the full REDI pipeline on all TaiESM1 2m_temperature NetCDF chunks and
# writes DATA_CARD.md, metadata.json, and NPZ output to the shared Lustre path.
#SBATCH -A stf053
#SBATCH -J redi_out_climax
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -t 04:00:00
#SBATCH -o /autofs/nccs-svm1_home1/whbrewer/drai/redi/benchmarks/pipeline/logs/out_climax_%j.out
#SBATCH -e /autofs/nccs-svm1_home1/whbrewer/drai/redi/benchmarks/pipeline/logs/out_climax_%j.out

set -euo pipefail

REDI_DIR="/autofs/nccs-svm1_home1/whbrewer/drai/redi"
VENV="$HOME/venv/redi/bin/activate"
NC_DIR="/lustre/orion/stf053/world-shared/vga/projects/ClimaX/data/TaiESM1/raw/2m_temperature"
OUT_DIR="/lustre/orion/proj-shared/stf249/redi/outputs/climax"
CFG="$REDI_DIR/benchmarks/pipeline/configs/bench_climax.yaml"

echo "=== REDI ClimaX Output Run  $(date) ==="
echo "  Node:    $(hostname)"
echo "  Job ID:  $SLURM_JOB_ID"
echo "  Output:  $OUT_DIR"

source "$VENV"
cd "$REDI_DIR"

mkdir -p benchmarks/pipeline/logs "$OUT_DIR"

# Process each NC file individually to avoid OOM from xr.concat on all 17 files
for nc_file in "$NC_DIR"/*.nc; do
    fname=$(basename "$nc_file" .nc)
    file_out="$OUT_DIR/$fname"
    mkdir -p "$file_out"
    echo "  Processing: $nc_file"
    redi run "$nc_file" \
        --domain climate \
        -c "$CFG" \
        -o "$file_out" \
        -v
done

echo "=== Done  $(date) ==="
echo "Output files:"
ls -lh "$OUT_DIR" 2>/dev/null || true
