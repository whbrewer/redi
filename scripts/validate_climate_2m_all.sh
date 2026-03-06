#!/usr/bin/env bash
# Run REDI pipeline + redi validate for all available TaiESM1 2m_temperature chunks.
# Produces validate_results/taiesm1_2m_temp_<ystr>_validate.json for each.
#
# Usage: bash scripts/validate_climate_2m_all.sh

set -e
cd "$(dirname "$0")/.."
source /opt/venvs/drai/bin/activate

RAW_DIR="/opt/drai/datasets/CMIP6/TaiESM1/raw/2m_temperature"
GT_DIR="/opt/drai/datasets/CMIP6/TaiESM1/1.40625deg/2m_temperature"
RESULTS="validate_results"
BASE_CFG="examples/climate_taiesm1_validate.yaml"
mkdir -p "$RESULTS"

echo "TaiESM1 2m_temperature — all chunks"
echo "====================================="

SUMMARY_JSON="$RESULTS/taiesm1_2m_temp_all_validate.json"
echo "[" > "$SUMMARY_JSON"
FIRST=true

for GT_FILE in $(ls "$GT_DIR"/*.nc | sort); do
  GT_FNAME=$(basename "$GT_FILE")
  # Extract year string: 2m_temperature_<ystr>_1.40625deg.nc
  YSTR=$(echo "$GT_FNAME" | sed 's/2m_temperature_//;s/_1\.40625deg\.nc//')
  RAW_FILE="$RAW_DIR/2m_temperature_${YSTR}_raw.nc"

  if [ ! -f "$RAW_FILE" ]; then
    echo "[$YSTR] skip — no raw file"
    continue
  fi

  echo ""
  echo "[$YSTR]"
  echo "  Raw: $(du -sh $RAW_FILE | cut -f1)"

  # Generate per-chunk config from base config
  CHUNK_CFG="/tmp/redi_2m_${YSTR}.yaml"
  sed "s|input_path:.*|input_path: $RAW_FILE|;s|output_dir:.*|output_dir: ./output/taiesm1_2m_temp_${YSTR}|" \
    "$BASE_CFG" > "$CHUNK_CFG"

  # Run REDI pipeline
  RUN_LOG="$RESULTS/taiesm1_2m_temp_${YSTR}_run.log"
  echo -n "  REDI run ... "
  redi run -c "$CHUNK_CFG" > "$RUN_LOG" 2>&1
  REDI_RUN_DIR=$(ls -dt "output/taiesm1_2m_temp_${YSTR}"/*/  2>/dev/null | head -1)
  echo "done → $REDI_RUN_DIR"

  # Validate
  VAL_JSON="$RESULTS/taiesm1_2m_temp_${YSTR}_validate.json"
  echo -n "  Validate ... "
  redi validate "$REDI_RUN_DIR" \
    --ground-truth "$GT_FILE" \
    --gt-variable t2m \
    --style json \
    > "$VAL_JSON" 2>&1

  # Print and append to summary
  python3 - <<PYEOF
import json
with open("$VAL_JSON") as f:
    r = json.load(f)
if "error" in r:
    print(f"  ERROR: {r['error']}")
    result = {"ystr": "$YSTR", "error": r['error']}
else:
    s = r["similarity"]
    print(f"done  corr={s['corr']:.6f}  MAE={s['mae']:.4f} K  shape={r['redi_shape']}")
    result = {
        "ystr": "$YSTR",
        "corr": s["corr"],
        "mae": s["mae"],
        "rmse": s["rmse"],
        "redi_shape": r["redi_shape"],
        "gt_shape": r["ground_truth_shape"],
    }

first = open("$SUMMARY_JSON").read().strip() == "["
with open("$SUMMARY_JSON", "a") as f:
    if not first:
        f.write(",\n")
    import json
    f.write(json.dumps(result, indent=2))
PYEOF

done

echo "" >> "$SUMMARY_JSON"
echo "]" >> "$SUMMARY_JSON"

echo ""
echo "====================================="
echo "Summary table:"
python3 - <<'PYEOF'
import json
with open("validate_results/taiesm1_2m_temp_all_validate.json") as f:
    results = json.load(f)
print(f"  {'Year range':<38s}  {'Corr':>8s}  {'MAE (K)':>8s}  {'RMSE (K)':>9s}")
print(f"  {'-'*38}  {'-'*8}  {'-'*8}  {'-'*9}")
for r in results:
    if "error" in r:
        print(f"  {r['ystr']:<38s}  ERROR: {r['error']}")
    else:
        print(f"  {r['ystr']:<38s}  {r['corr']:>8.6f}  {r['mae']:>8.4f}  {r['rmse']:>9.4f}")
PYEOF

echo ""
echo "Full results in $RESULTS/"
