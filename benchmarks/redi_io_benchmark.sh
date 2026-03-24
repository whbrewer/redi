#!/bin/bash
#SBATCH --job-name=redi-io-bench
#SBATCH --account=stf053
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --output=redi-io-bench-%j.out

echo "== Start $(date) =="

source ~/venv/redi/bin/activate

BENCH=benchmarks/redi_io_benchmark.py
OUT=/lustre/orion/scratch/whbrewer/stf053/openfold_io_bench

run_domain() {
    local domain=$1
    local result_json="$OUT/$domain/benchmark_${domain}.json"
    shift
    if [ -f "$result_json" ]; then
        echo "=== SKIP $domain (results already exist: $result_json) ==="
        return 0
    fi
    echo ""
    echo "=== $domain ==="
    python -u $BENCH --domain "$domain" "$@" || echo "WARNING: $domain benchmark failed, continuing..."
}

# --- life_sciences (OpenFold) ---
run_domain life_sciences \
    --npz-dir /lustre/orion/gen053/world-shared/openfold/redi_output/train \
    --alignment-dir /lustre/orion/gen053/world-shared/openfold/data/open_protein_set/pdb \
    --mmcif-dir /lustre/orion/gen053/world-shared/openfold/data/pdb_mmcif/mmcif_files \
    --output-dir $OUT/life_sciences \
    --n-samples 5000 \
    --n-reads 1000 \
    --skip-pack

# --- materials (OC22 / HydraGNN) ---
run_domain materials \
    --lmdb-dir /lustre/orion/stf053/proj-shared/drai/HydraGNN/data/oc22/s2ef_total_train_val_test_lmdbs/data/oc22/s2ef-total/train \
    --npz-file /lustre/orion/stf053/scratch/whbrewer/redi_oc22_bench_output/oc22_redi_output.npz \
    --output-dir $OUT/materials \
    --n-samples 100000 \
    --n-reads 1000 \
    --timeout 120

# --- climate (ClimaX) ---
# Run REDI on TaiESM1 data to produce train.zarr / train.npz, then benchmark.
CLIMATE_NC="/lustre/orion/stf053/world-shared/vga/projects/ClimaX/data/TaiESM1/raw/2m_temperature/2m_temperature_185001010000-186001010000_raw.nc"
CLIMATE_REDI_OUT="$OUT/climate/redi_output"
if [ ! -d "$CLIMATE_REDI_OUT/train.zarr" ]; then
    echo "Running REDI on TaiESM1 climate data..."
    redi run "$CLIMATE_NC" --domain climate --format zarr -o "$CLIMATE_REDI_OUT"
    redi run "$CLIMATE_NC" --domain climate --format npz  -o "$CLIMATE_REDI_OUT"
fi
run_domain climate \
    --redi-output-dir "$CLIMATE_REDI_OUT" \
    --output-dir $OUT/climate \
    --n-reads 500 \
    --window 10

# --- fusion (XGC1) ---
run_domain fusion \
    --h5-file /lustre/orion/stf053/proj-shared/ketan/redi-all-data-processed/xgc.3d.00002.bp.outdir/data.h5 \
    --bp-dir /lustre/orion/world-shared/fus184/fusiond-seed-xgc1-data/n560fr_ITER_PFPO_W_Ne \
    --output-dir $OUT/fusion \
    --n-reads 500 \
    --window 1

echo ""
echo "== Done $(date) =="
