#!/bin/bash
# ── REDI Demo: 4-pane tmux pipeline recording ─────────────────────────────────
#
# Opens a tmux session with one pane per case, fires all four redi runs
# simultaneously, then waits for all to finish.
#
# Usage (on an Andes login or interactive node):
#   source ~/venv/redi/bin/activate
#   cd ~/drai/redi
#   bash benchmarks/demo/record_pipeline.sh
#
# To record with asciinema:
#   asciinema rec -t "REDI SC26 Demo - Pipeline" demo_pipeline.cast \
#       --command "bash benchmarks/demo/record_pipeline.sh"

REDI_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="$HOME/venv/redi/bin/activate"
CFG="$REDI_DIR/benchmarks/pipeline/configs"

NC_FILE="/lustre/orion/stf053/world-shared/vga/projects/ClimaX/data/TaiESM1/raw/2m_temperature/2m_temperature_185001010000-186001010000_raw.nc"
OC22_DIR="/lustre/orion/gen053/world-shared/HydraGNN/data/oc22/s2ef_total_train_val_test_lmdbs/data/oc22/s2ef-total/train"
OPS_ROOT="/lustre/orion/gen053/world-shared/openfold/data"
XGC1_ROOT="/lustre/orion/world-shared/fus184/jyc/fusiond-seed-xgc1-data-samples"

OUT_BASE="/tmp/redi_demo_$$"
mkdir -p "$OUT_BASE"/{climax,oc22,openfold,xgc1}

SESSION="redi_demo"
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session  -d -s "$SESSION" -x 220 -y 50

# Layout: 2×2 grid
tmux split-window  -t "$SESSION" -h
tmux split-window  -t "$SESSION:0.0" -v
tmux split-window  -t "$SESSION:0.2" -v

# Label each pane
tmux send-keys -t "$SESSION:0.0" "printf '\033]2;ClimaX / TaiESM1\007'" Enter
tmux send-keys -t "$SESSION:0.1" "printf '\033]2;OC22 / HydraGNN\007'" Enter
tmux send-keys -t "$SESSION:0.2" "printf '\033]2;OpenFold / OpenProteinSet\007'" Enter
tmux send-keys -t "$SESSION:0.3" "printf '\033]2;XGC1 ITER\007'" Enter

# Activate venv in all panes
for pane in 0 1 2 3; do
    tmux send-keys -t "$SESSION:0.$pane" "source $VENV && cd $REDI_DIR" Enter
done

sleep 1

# Fire all four cases
tmux send-keys -t "$SESSION:0.0" \
    "echo '=== ClimaX / TaiESM1 ===' && redi run '$NC_FILE' --domain climate -c '$CFG/demo_climax.yaml' -o '$OUT_BASE/climax' -v" Enter

tmux send-keys -t "$SESSION:0.1" \
    "echo '=== OC22 / HydraGNN ===' && redi run '$OC22_DIR' --domain materials -c '$CFG/demo_oc22.yaml' -o '$OUT_BASE/oc22' -v" Enter

tmux send-keys -t "$SESSION:0.2" \
    "echo '=== OpenFold / OpenProteinSet ===' && redi run '$OPS_ROOT' --domain life_sciences -c '$CFG/demo_openfold.yaml' -o '$OUT_BASE/openfold' -v" Enter

tmux send-keys -t "$SESSION:0.3" \
    "echo '=== XGC1 ITER ===' && redi run '$XGC1_ROOT' --domain xgc1 -c '$CFG/demo_xgc1.yaml' -o '$OUT_BASE/xgc1' -v" Enter

# Attach so the recording captures it
tmux attach-session -t "$SESSION"
