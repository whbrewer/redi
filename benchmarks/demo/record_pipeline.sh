#!/bin/bash
# ── REDI Demo: sequential pipeline recording with simulated typing ─────────────
#
# Usage:
#   asciinema rec -t "REDI SC26 Demo" /tmp/demo_pipeline.cast \
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

type_text() {
    local text="$1"
    local delay="${2:-0.04}"
    for ((i=0; i<${#text}; i++)); do
        printf '%s' "${text:$i:1}"
        sleep "$delay"
    done
}

run_cmd() {
    local cmd="$1"
    echo -n "$PROMPT"
    type_text "$cmd" 0.045
    sleep 0.6
    echo
    eval "$cmd"
}

pause() { sleep "${1:-1.5}"; }

show_metadata() {
    python3 - "$1" <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1]))
skip = {"per_file", "saved_files", "input_files", "normalization_stats", "lmdb_shards"}
for k, v in m.items():
    if k in skip:
        continue
    if isinstance(v, (dict, list)) and len(str(v)) > 80:
        continue
    print(f"  {k}: {v}")
PYEOF
}

show_outputs() {
    local label="$1"
    local dir="$2"
    echo
    echo "── outputs: $label ──"
    run_cmd "ls -lh $dir/"
    pause 2
    echo
    run_cmd "cat $dir/metadata.json"
    pause 3
}

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
pause 3
show_outputs "ClimaX / TaiESM1" "$OUT_BASE/climax"

# ── Case 2: OC22 ───────────────────────────────────────────────────────────────
clear
echo
echo "# Case 2/4 — OC22 / HydraGNN  (materials)"
echo
pause 1
run_cmd "redi run '$OC22_DIR' --domain materials -c '$CFG/demo_oc22.yaml' -o '$OUT_BASE/oc22' -v"
pause 3
show_outputs "OC22 / HydraGNN" "$OUT_BASE/oc22"

# ── Case 3: OpenFold ───────────────────────────────────────────────────────────
clear
echo
echo "# Case 3/4 — OpenFold / OpenProteinSet  (life sciences)"
echo
pause 1
run_cmd "redi run '$OPS_ROOT' --domain life_sciences -c '$CFG/demo_openfold.yaml' -o '$OUT_BASE/openfold' -v"
pause 3
show_outputs "OpenFold / OpenProteinSet" "$OUT_BASE/openfold"

# ── Case 4: XGC1 ───────────────────────────────────────────────────────────────
clear
echo
echo "# Case 4/4 — XGC1 ITER  (fusion)"
echo
pause 1
run_cmd "redi run '$XGC1_ROOT' --domain xgc1 -c '$CFG/demo_xgc1.yaml' -o '$OUT_BASE/xgc1' -v"
pause 3
show_outputs "XGC1 ITER" "$OUT_BASE/xgc1"

pause 2
