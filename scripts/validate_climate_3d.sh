#!/usr/bin/env bash
# Validate REDI vs ClimaX for TaiESM1 3D variables (ta, ua, va).
# Run after raw files are downloaded to /opt/drai/CMIP6/TaiESM1/raw/<name>/
#
# Usage: bash scripts/validate_climate_3d.sh

set -e
cd "$(dirname "$0")/.."
source /opt/venvs/drai/bin/activate

YSTR="186001010600-187001010000"
CLIMAX_SRC="/home/w1b/drai/domains/climate/ClimaX/src/data_preprocessing/regrid.py"
GT_BASE="/opt/drai/datasets/CMIP6/TaiESM1/1.40625deg"
RAW_BASE="/opt/drai/CMIP6/TaiESM1/raw"
RESULTS="validate_results"

declare -A CMIP_NAME=( [temperature]=ta [u_component_of_wind]=ua [v_component_of_wind]=va )
declare -A ERA_NAME=(  [temperature]=t  [u_component_of_wind]=u  [v_component_of_wind]=v  )
declare -A REDI_CFG=(
  [temperature]=examples/climate_taiesm1_temperature_validate.yaml
  [u_component_of_wind]=examples/climate_taiesm1_uwind_validate.yaml
  [v_component_of_wind]=examples/climate_taiesm1_vwind_validate.yaml
)
declare -A REDI_OUT=(
  [temperature]=output/taiesm1_temperature_redi
  [u_component_of_wind]=output/taiesm1_uwind_redi
  [v_component_of_wind]=output/taiesm1_vwind_redi
)

for NAME in temperature u_component_of_wind v_component_of_wind; do
  CMIP="${CMIP_NAME[$NAME]}"
  ERA="${ERA_NAME[$NAME]}"
  RAW="${RAW_BASE}/${NAME}/${NAME}_${YSTR}_raw.nc"
  GT_DIR="${GT_BASE}/${NAME}"
  GT_FILE="${GT_DIR}/${NAME}_${YSTR}_1.40625deg.nc"
  REDI_LOG="${RESULTS}/taiesm1_${NAME}_redi_run.log"
  VAL_JSON="${RESULTS}/taiesm1_${NAME}_validate.json"

  echo "========================================"
  echo "Variable: $NAME  ($CMIP -> $ERA)"
  echo "========================================"

  # Check raw file exists
  if [ ! -f "$RAW" ]; then
    echo "  ERROR: raw file not found: $RAW"
    echo "  Run: wget <esgf_url> -O $RAW"
    continue
  fi
  echo "  Raw file: OK ($(du -sh "$RAW" | cut -f1))"

  # Step 1: ClimaX regrid to produce ground truth
  mkdir -p "$GT_DIR"
  if [ ! -f "$GT_FILE" ]; then
    echo "  Running ClimaX regrid.py ..."
    python "$CLIMAX_SRC" \
      --input_fns "$RAW" \
      --output_dir "$GT_DIR" \
      --ddeg_out 1.40625 \
      --reuse_weights 0 \
      --cmip 1 \
      --rename "$CMIP" "$ERA"
    # regrid.py writes a .nc.tmp file; rename it
    TMP="${GT_DIR}/${NAME}_${YSTR}_1.40625deg.nc.tmp"
    if [ -f "$TMP" ]; then mv "$TMP" "$GT_FILE"; fi
    echo "  ClimaX GT saved: $GT_FILE"
  else
    echo "  ClimaX GT: already exists, skipping"
  fi

  # Step 2: REDI pipeline
  echo "  Running REDI pipeline ..."
  redi run -c "${REDI_CFG[$NAME]}" -v > "$REDI_LOG" 2>&1
  REDI_RUN_DIR=$(ls -dt "${REDI_OUT[$NAME]}"/*/  2>/dev/null | head -1)
  echo "  REDI output: $REDI_RUN_DIR"

  # Step 3: Validate
  echo "  Running redi validate ..."
  redi validate "$REDI_RUN_DIR" \
    --ground-truth "$GT_FILE" \
    --gt-variable "$ERA" \
    --per-feature \
    --style json \
    > "$VAL_JSON" 2>&1

  # Print summary
  python3 - <<PYEOF
import json
with open("$VAL_JSON") as f:
    r = json.load(f)
if "error" in r:
    print(f"  ERROR: {r['error']}")
else:
    s = r["similarity"]
    print(f"  Shape REDI: {r['redi_shape']}  GT: {r['ground_truth_shape']}")
    print(f"  corr={s['corr']:.6f}  MAE={s['mae']:.4f}  RMSE={s['rmse']:.4f}")
PYEOF
  echo ""
done

echo "========================================"
echo "All done. Results in $RESULTS/"
echo "========================================"
