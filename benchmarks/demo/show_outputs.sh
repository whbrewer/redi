#!/bin/bash
# ── REDI Demo: show outputs ────────────────────────────────────────────────────
#
# Walks through the four completed output directories on Lustre, showing
# file listings and key content from each case.  Designed to be recorded
# with asciinema after the pipeline run.
#
# Usage:
#   asciinema rec -t "REDI SC26 Demo - Outputs" demo_outputs.cast \
#       --command "bash benchmarks/demo/show_outputs.sh"

LUSTRE="/lustre/orion/proj-shared/stf249/redi/outputs"

# ── helpers ───────────────────────────────────────────────────────────────────
pause() { sleep "${1:-1.5}"; }

banner() {
    echo
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $*"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo
    pause
}

show_metadata() {
    local mf="$1"
    python3 - "$mf" <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1]))
skip = {"per_file", "saved_files", "input_files", "normalization_stats"}
for k, v in m.items():
    if k in skip:
        continue
    if isinstance(v, (dict, list)) and len(str(v)) > 80:
        continue
    print(f"  {k}: {v}")
PYEOF
}

# ── ClimaX ────────────────────────────────────────────────────────────────────
banner "Case 1/4 — ClimaX / TaiESM1  (climate)"
echo "Output directory:"
ls -lh "$LUSTRE/climax/" | grep -v "^total\|2m_temp"
pause
echo
echo "Top-level metadata.json:"
show_metadata "$LUSTRE/climax/metadata.json"
pause 2

# ── OC22 ──────────────────────────────────────────────────────────────────────
banner "Case 2/4 — OC22 / HydraGNN  (materials)"
echo "Output directory:"
ls -lh "$LUSTRE/oc22/"
pause
echo
echo "metadata.json:"
show_metadata "$LUSTRE/oc22/metadata.json"
pause 2

# ── OpenFold ──────────────────────────────────────────────────────────────────
banner "Case 3/4 — OpenFold / OpenProteinSet  (life sciences)"
echo "Output directory:"
ls -lh "$LUSTRE/openfold/"
pause
echo
echo "metadata.json:"
show_metadata "$LUSTRE/openfold/metadata.json"
pause 2

# ── XGC1 ──────────────────────────────────────────────────────────────────────
banner "Case 4/4 — XGC1 ITER  (fusion)"
echo "Output directory:"
ls -lh "$LUSTRE/xgc1/"
pause
echo
echo "metadata.json:"
show_metadata "$LUSTRE/xgc1/metadata.json"
pause 2

# ── DATA_CARD sample ──────────────────────────────────────────────────────────
banner "DATA_CARD.md — OC22 / HydraGNN"
head -60 "$LUSTRE/oc22/DATA_CARD.md"
pause 3

echo
echo "All four cases complete. Outputs at:"
echo "  $LUSTRE/{climax,oc22,openfold,xgc1}/"
echo
