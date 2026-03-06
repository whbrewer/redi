#!/usr/bin/env bash
# REDI vs RAPS HPC Validation Benchmark
#
# For each case in examples/hpc/:
#   1. redi run -c examples/hpc/<case>.yaml   → output/<case>_raps/raps_jobs.npz
#   2. raps run <config> --workload replay     → validate_results/<case>_raps_gt/snapshot.npz
#   3. redi validate <redi_out> --ground-truth snapshot.npz --style json
#
# Monitor progress from home:
#   ssh <host> -t "tail -f ~/redi_validate_progress.log"
#   ssh <host> -t "tmux attach -t redi_benchmark"

set -uo pipefail

REDI_DIR="/home/w1b/drai/apps/redi"
RAPS_DIR="/home/w1b/exadigit/raps"
DRAI_VENV="/opt/venvs/drai/bin/activate"
RAPS_VENV="/opt/venvs/exadigit/bin/activate"
RESULTS_DIR="$REDI_DIR/validate_results"
LOG="$HOME/redi_validate_progress.log"

mkdir -p "$RESULTS_DIR"

# Truncate/create log
exec > >(tee "$LOG") 2>&1

echo "╔══════════════════════════════════════════════╗"
echo "║   REDI HPC Validation Benchmark              ║"
echo "╚══════════════════════════════════════════════╝"
echo "Started: $(date)"
echo "Log:     $LOG"
echo "Results: $RESULTS_DIR"
echo ""

log()   { echo "[$(date '+%H:%M:%S')] $*"; }
sep()   { echo ""; echo "────────────────────────────────────────────────"; log "$*"; echo "────────────────────────────────────────────────"; }

# Ground-truth generation strategy per case.
# Three modes:
#   yaml:<path>              — raps run <experiments/x.yaml> -t 1
#   cli:<system>:<data>      — raps run --system <sys> --replay <path> -t 1
#                              (needed when dataloader can't handle a list-style path)
#   parts:<path>:<gt_file>   — raps run-parts <experiments/x.yaml> -t 1h
#                              gt_file is the partition snapshot to use for validation
#   ""                       — no RAPS support, skip
declare -A RAPS_CONFIGS=(
    [adastra_mi250]="cli:adastraMI250:/opt/data/adastraMI250/AdastaJobsMI250_15days.parquet"
    [aurora]="yaml:experiments/aurora.yaml"
    [bluewaters]="yaml:experiments/bluewaters.yaml"
    [frontier]="yaml:experiments/frontier.yaml"
    [fugaku]="cli:fugaku:/opt/data/fugaku/21_04.parquet"
    [gcloud_v2]="yaml:experiments/gcloudv2.yaml"
    [lassen]="yaml:experiments/lassen.yaml"
    [marconi100]="yaml:experiments/marconi100.yaml"
    [mit_supercloud]="parts:experiments/mit-replay-24hrs.yaml:part-cpu.npz"
    [philly]="parts:experiments/philly.yaml:2-gpu.npz"
)

CASES=(adastra_mi250 aurora bluewaters frontier fugaku gcloud_v2 lassen marconi100 mit_supercloud philly)

# Helper: get output_dir from a REDI YAML config
get_output_dir() {
    local config_path="$1"
    local fallback="$2"
    python3 -c "
import yaml
try:
    with open('$config_path') as f:
        c = yaml.safe_load(f)
    print(c.get('output_dir', '$fallback'))
except:
    print('$fallback')
" 2>/dev/null
}

# ─────────────────────────────────────────────────────────────────────────────
sep "PHASE 1 of 3 — redi run (extract jobs + produce raps_jobs.npz)"

source "$DRAI_VENV"
cd "$REDI_DIR"

declare -A REDI_OUT_DIRS  # case → absolute path to redi output dir

for CASE in "${CASES[@]}"; do
    CONFIG="$REDI_DIR/examples/hpc/${CASE}.yaml"
    REL_OUT=$(get_output_dir "$CONFIG" "./output/${CASE}_raps")
    ABS_OUT="$REDI_DIR/${REL_OUT#./}"
    REDI_OUT_DIRS[$CASE]="$ABS_OUT"

    log "[$CASE] output → $REL_OUT"

    if [ -f "$ABS_OUT/raps_jobs.npz" ]; then
        log "[$CASE] REDI ✓ cached (raps_jobs.npz exists)"
        continue
    fi

    log "[$CASE] Running: redi run -c examples/hpc/${CASE}.yaml --output $ABS_OUT"
    if redi run -c "$CONFIG" --output "$ABS_OUT" --style simple; then
        log "[$CASE] REDI ✓ OK"
    else
        log "[$CASE] REDI ✗ FAILED"
    fi
done

# ─────────────────────────────────────────────────────────────────────────────
sep "PHASE 2 of 3 — raps run (load raw job logs natively, generate ground truth)"

declare -A RAPS_GT_PATHS  # case → absolute path to snapshot.npz (or empty)

for CASE in "${CASES[@]}"; do
    RAPS_CFG="${RAPS_CONFIGS[$CASE]:-}"
    GT_DIR="$RESULTS_DIR/${CASE}_raps_gt"
    GT_SNAP="$GT_DIR/snapshot.npz"

    RAPS_GT_PATHS[$CASE]=""

    if [ -z "$RAPS_CFG" ]; then
        log "[$CASE] RAPS ✗ skipped — no RAPS support for this system"
        continue
    fi

    # Parse mode prefix: yaml:, cli:, or parts:<yaml>:<gt_file>
    MODE="${RAPS_CFG%%:*}"
    RAPS_ARG="${RAPS_CFG#*:}"

    # For run-parts, snapshot filename is the optional 3rd component (default: part-cpu.npz)
    if [ "$MODE" = "parts" ]; then
        PARTS_YAML="${RAPS_ARG%%:*}"
        PARTS_GT="${RAPS_ARG#*:}"
        # If no second colon, PARTS_GT == PARTS_YAML (no override) → use default
        if [ "$PARTS_GT" = "$PARTS_YAML" ]; then
            PARTS_GT="part-cpu.npz"
        fi
        RAPS_ARG="$PARTS_YAML"
        GT_SNAP="$GT_DIR/$PARTS_GT"
    fi

    if [ -f "$GT_SNAP" ]; then
        log "[$CASE] RAPS ✓ cached"
        RAPS_GT_PATHS[$CASE]="$GT_SNAP"
        continue
    fi

    # For parts mode: raps run-parts uses pathlib.mkdir() without exist_ok,
    # so the output dir must NOT exist yet. Remove any leftover empty dir.
    if [ "$MODE" = "parts" ]; then
        rm -rf "$GT_DIR"
    else
        mkdir -p "$GT_DIR"
    fi

    (
        cd "$RAPS_DIR"
        source "$RAPS_VENV"
        case "$MODE" in
            yaml)
                log "[$CASE] Running: raps run $RAPS_ARG -t 1 --noui --no-cooling --output $GT_DIR"
                raps run "$RAPS_ARG" -t 1 --noui --no-cooling --output "$GT_DIR"
                ;;
            cli)
                SYSTEM="${RAPS_ARG%%:*}"
                DATA_PATH="${RAPS_ARG#*:}"
                log "[$CASE] Running: raps run --system $SYSTEM --replay $DATA_PATH -t 1 --noui --no-cooling --output $GT_DIR"
                raps run --system "$SYSTEM" --replay "$DATA_PATH" -t 1 --noui --no-cooling --output "$GT_DIR"
                ;;
            parts)
                log "[$CASE] Running: raps run-parts $RAPS_ARG -t 1h --noui --no-cooling --output $GT_DIR"
                raps run-parts "$RAPS_ARG" -t 1h --noui --no-cooling --output "$GT_DIR"
                ;;
        esac
    )
    RAPS_EXIT=$?

    if [ $RAPS_EXIT -eq 0 ] && [ -f "$GT_SNAP" ]; then
        log "[$CASE] RAPS ✓ OK → $GT_DIR"
        RAPS_GT_PATHS[$CASE]="$GT_SNAP"
    else
        log "[$CASE] RAPS ✗ FAILED (exit $RAPS_EXIT)"
    fi
done

# Re-activate drai venv after raps subshells
source "$DRAI_VENV"
cd "$REDI_DIR"

# ─────────────────────────────────────────────────────────────────────────────
sep "PHASE 3 of 3 — redi validate"

for CASE in "${CASES[@]}"; do
    ABS_OUT="${REDI_OUT_DIRS[$CASE]:-}"
    GT_SNAP="${RAPS_GT_PATHS[$CASE]:-}"
    VALIDATE_JSON="$RESULTS_DIR/${CASE}_validate.json"

    if [ ! -d "$ABS_OUT" ]; then
        log "[$CASE] validate ✗ skipped — redi output dir not found: $ABS_OUT"
        continue
    fi

    if [ -z "$GT_SNAP" ]; then
        log "[$CASE] validate ✗ skipped — no ground truth available (redi validate requires --ground-truth)"
        continue
    fi

    log "[$CASE] Running: redi validate $ABS_OUT --ground-truth $GT_SNAP --style json"

    if redi validate "$ABS_OUT" \
        --ground-truth "$GT_SNAP" \
        --per-feature \
        --style json \
        > "$VALIDATE_JSON" 2>>"$LOG"; then
        log "[$CASE] validate ✓ OK → $VALIDATE_JSON"
    else
        log "[$CASE] validate ✗ FAILED"
        rm -f "$VALIDATE_JSON"
    fi
done

# ─────────────────────────────────────────────────────────────────────────────
sep "RESULTS SUMMARY"

python3 - << 'PYEOF'
import json, os, glob

results_dir = "/home/w1b/drai/apps/redi/validate_results"
cases = ["adastra_mi250", "aurora", "bluewaters", "frontier", "fugaku",
         "gcloud_v2", "lassen", "marconi100", "mit_supercloud", "philly"]

raps_cfgs = {
    "adastra_mi250": "adastraMI250.yaml",
    "aurora": None,
    "bluewaters": "bluewaters.yaml",
    "frontier": "frontier.yaml",
    "fugaku": "fugaku.yaml",
    "gcloud_v2": "gcloudv2.yaml",
    "lassen": "lassen.yaml",
    "marconi100": "marconi100.yaml",
    "mit_supercloud": "mit_supercloud/part-cpu.yaml",
    "philly": "philly/2-gpu.yaml",
}

summary = {}
for case in cases:
    vf = os.path.join(results_dir, f"{case}_validate.json")
    gt_dir = os.path.join(results_dir, f"{case}_raps_gt")
    gt_snap = os.path.join(gt_dir, "snapshot.npz")
    # run-parts systems produce partition-named files (e.g. part-cpu.npz, 2-gpu.npz)
    # rather than snapshot.npz — check for any .npz in the GT dir as fallback
    if not os.path.exists(gt_snap):
        npz_files = glob.glob(os.path.join(gt_dir, "*.npz"))
        if npz_files:
            gt_snap = sorted(npz_files)[0]
    entry = {
        "raps_config": raps_cfgs.get(case),
        "raps_gt_available": os.path.exists(gt_snap),
        "validate_done": os.path.exists(vf),
    }
    if os.path.exists(vf):
        try:
            with open(vf) as f:
                d = json.load(f)
            for k in ["readiness_score", "readiness_level", "overall_score",
                      "passed", "failed", "redi_shape", "ground_truth_shape"]:
                if k in d:
                    entry[k] = d[k]
            # Pull similarity metrics if present
            if "similarity" in d and d["similarity"]:
                entry["similarity"] = {
                    k: round(v, 4) for k, v in d["similarity"].items()
                    if isinstance(v, (int, float))
                }
        except Exception as e:
            entry["parse_error"] = str(e)
    summary[case] = entry

out_path = os.path.join(results_dir, "summary.json")
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)

# Print table
print(f"\n{'Case':<20s}  {'GT':^5s}  {'Validated':^9s}  {'Readiness':^10s}  {'MAE':^8s}  {'Corr':^8s}")
print("-" * 72)
for case, e in summary.items():
    gt = "✓" if e["raps_gt_available"] else "—"
    val = "✓" if e["validate_done"] else "—"
    rl = str(e.get("readiness_level", "—"))
    sim = e.get("similarity", {})
    mae = f"{sim.get('mae', float('nan')):.4f}" if sim.get("mae") is not None else "—"
    corr = f"{sim.get('corr', float('nan')):.4f}" if sim.get("corr") is not None else "—"
    print(f"{case:<20s}  {gt:^5s}  {val:^9s}  {rl:^10s}  {mae:^8s}  {corr:^8s}")

print(f"\nSummary written to: {out_path}")
PYEOF

echo ""
echo "All done! $(date)"
echo ""
echo "Key files:"
echo "  Progress log:  $LOG"
echo "  Results dir:   $RESULTS_DIR"
echo "  Summary:       $RESULTS_DIR/summary.json"
