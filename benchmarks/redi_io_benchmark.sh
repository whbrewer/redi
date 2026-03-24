#!/bin/bash
#SBATCH --job-name=openfold-io-bench
#SBATCH --account=stf053
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=openfold-io-bench-%j.out

set -e
echo "== Start $(date) =="

source ~/venv/redi/bin/activate

BENCH=benchmarks/redi_io_benchmark.py
OUT=/lustre/orion/scratch/whbrewer/stf053/openfold_io_bench

# --- life_sciences (OpenFold) ---
echo ""
echo "=== LIFE_SCIENCES ==="
python $BENCH \
    --domain life_sciences \
    --npz-dir /lustre/orion/gen053/world-shared/openfold/redi_output/train \
    --output-dir $OUT/life_sciences \
    --n-samples 5000 \
    --n-reads 1000

# --- climate (ClimaX) — wait for job 3244958 to finish first; use single_file_test as fallback ---
echo ""
echo "=== CLIMATE ==="
CLIMAX_NPZ=/lustre/orion/scratch/whbrewer/stf053/ClimaX/data.npz
CLIMAX_SINGLE=/lustre/orion/scratch/whbrewer/stf053/ClimaX/single_file_test/data.npz
if [ -f "$CLIMAX_NPZ" ]; then
    NPZ_FILE=$CLIMAX_NPZ
else
    NPZ_FILE=$CLIMAX_SINGLE
    echo "Using single_file_test NPZ as fallback"
fi
python $BENCH \
    --domain climate \
    --npz-file $NPZ_FILE \
    --output-dir $OUT/climate \
    --n-reads 500 \
    --window 10

# --- fusion (XGC1) — input is HDF5, benchmark HDF5 vs Zarr ---
echo ""
echo "=== FUSION (XGC1) ==="
python $BENCH \
    --domain fusion \
    --h5-file /lustre/orion/stf053/proj-shared/ketan/redi-all-data-processed/xgc.3d.00002.bp.outdir/data.h5 \
    --output-dir $OUT/fusion \
    --n-reads 500 \
    --window 1

echo ""
echo "== Done $(date) =="
