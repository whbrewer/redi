#!/bin/bash
# ── REDI Demo: sequential pipeline recording with simulated typing ─────────────
#
# Usage:
#   asciinema rec -t "REDI SC26 Demo - Pipeline" /tmp/demo_pipeline.cast \
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

# ── helpers ───────────────────────────────────────────────────────────────────

PROMPT="(redi) whbrewer@andes$ "

# Print text character by character, like someone typing
type_text() {
    local text="$1"
    local delay="${2:-0.04}"
    for ((i=0; i<${#text}; i++)); do
        printf '%s' "${text:$i:1}"
        sleep "$delay"
    done
}

# Show a command being typed, pause, then run it
run_cmd() {
    local cmd="$1"
    echo -n "$PROMPT"
    type_text "$cmd" 0.045
    sleep 0.6
    echo
    eval "$cmd"
}

# Blank line pause
pause() { sleep "${1:-1.5}"; }

# ── intro ─────────────────────────────────────────────────────────────────────
clear
echo
pause 1
run_cmd "redi"
pause 4

clear
echo
run_cmd "redi plugins"
pause 4
clear

# ── Case 1: ClimaX ─────────────────────────────────────────────────────────────
echo
echo "# Case 1/4 — ClimaX / TaiESM1  (climate)"
echo
pause 1
run_cmd "redi run '$NC_FILE' --domain climate -c '$CFG/demo_climax.yaml' -o '$OUT_BASE/climax' -v"
pause 2

# ── Case 2: OC22 ───────────────────────────────────────────────────────────────
clear
echo
echo "# Case 2/4 — OC22 / HydraGNN  (materials)"
echo
pause 1
run_cmd "redi run '$OC22_DIR' --domain materials -c '$CFG/demo_oc22.yaml' -o '$OUT_BASE/oc22' -v"
pause 2

# ── Case 3: OpenFold ───────────────────────────────────────────────────────────
clear
echo
echo "# Case 3/4 — OpenFold / OpenProteinSet  (life sciences)"
echo
pause 1
run_cmd "redi run '$OPS_ROOT' --domain life_sciences -c '$CFG/demo_openfold.yaml' -o '$OUT_BASE/openfold' -v"
pause 2

# ── Case 4: XGC1 ───────────────────────────────────────────────────────────────
clear
echo
echo "# Case 4/4 — XGC1 ITER  (fusion)"
echo
pause 1
run_cmd "redi run '$XGC1_ROOT' --domain xgc1 -c '$CFG/demo_xgc1.yaml' -o '$OUT_BASE/xgc1' -v"
pause 3
