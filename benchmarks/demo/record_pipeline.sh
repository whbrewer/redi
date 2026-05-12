#!/bin/bash
# ── REDI Demo: sequential pipeline recording (asciinema-compatible) ────────────
#
# Runs all four cases one after another in a single terminal with clear
# section banners. Records cleanly with asciinema; speed up 8-10x in post.
#
# Usage:
#   asciinema rec -t "REDI SC26 Demo - Pipeline" demo_pipeline.cast \
#       --command "bash benchmarks/demo/record_pipeline.sh"

REDI_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
CFG="$REDI_DIR/benchmarks/pipeline/configs"

NC_FILE="/lustre/orion/stf053/world-shared/vga/projects/ClimaX/data/TaiESM1/raw/2m_temperature/2m_temperature_185001010000-186001010000_raw.nc"
OC22_DIR="/lustre/orion/gen053/world-shared/HydraGNN/data/oc22/s2ef_total_train_val_test_lmdbs/data/oc22/s2ef-total/train"
OPS_ROOT="/lustre/orion/gen053/world-shared/openfold/data"
XGC1_ROOT="/lustre/orion/world-shared/fus184/jyc/fusiond-seed-xgc1-data-samples"

OUT_BASE="/tmp/redi_demo_$$"

source "$HOME/venv/redi/bin/activate"
cd "$REDI_DIR"

banner() {
    clear
    echo "╔══════════════════════════════════════════════════════════════════╗"
    printf  "║  %-64s║\n" "$1"
    printf  "║  %-64s║\n" "$2"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo
    sleep 1.5
}

# ── Case 1: ClimaX ─────────────────────────────────────────────────────────────
banner "REDI  ▸  Case 1 / 4" "ClimaX / TaiESM1  (climate)"
redi run "$NC_FILE" \
    --domain climate \
    -c "$CFG/demo_climax.yaml" \
    -o "$OUT_BASE/climax" \
    -v
sleep 2

# ── Case 2: OC22 ───────────────────────────────────────────────────────────────
banner "REDI  ▸  Case 2 / 4" "OC22 / HydraGNN  (materials)"
redi run "$OC22_DIR" \
    --domain materials \
    -c "$CFG/demo_oc22.yaml" \
    -o "$OUT_BASE/oc22" \
    -v
sleep 2

# ── Case 3: OpenFold ───────────────────────────────────────────────────────────
banner "REDI  ▸  Case 3 / 4" "OpenFold / OpenProteinSet  (life sciences)"
redi run "$OPS_ROOT" \
    --domain life_sciences \
    -c "$CFG/demo_openfold.yaml" \
    -o "$OUT_BASE/openfold" \
    -v
sleep 2

# ── Case 4: XGC1 ───────────────────────────────────────────────────────────────
banner "REDI  ▸  Case 4 / 4" "XGC1 ITER  (fusion)"
redi run "$XGC1_ROOT" \
    --domain xgc1 \
    -c "$CFG/demo_xgc1.yaml" \
    -o "$OUT_BASE/xgc1" \
    -v
sleep 2

# ── Done ───────────────────────────────────────────────────────────────────────
clear
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  REDI  ▸  All 4 cases complete                                  ║"
echo "║                                                                  ║"
echo "║  climate · materials · life sciences · fusion                   ║"
echo "║  Readiness Level 5 — FULLY AI-READY                             ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo
sleep 3
