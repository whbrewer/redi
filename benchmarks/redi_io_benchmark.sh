#!/bin/bash
#SBATCH --job-name=redi-io-bench
#SBATCH --account=stf053
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --output=redi-io-bench-%j.out

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
    --alignment-dir /lustre/orion/gen053/world-shared/openfold/data/open_protein_set/pdb \
    --mmcif-dir /lustre/orion/gen053/world-shared/openfold/data/pdb_mmcif/mmcif_files \
    --output-dir $OUT/life_sciences \
    --n-samples 5000 \
    --n-reads 1000 \
    --skip-pack

# --- materials (OC22 / HydraGNN) ---
echo ""
echo "=== MATERIALS (OC22 / HydraGNN) ==="
python $BENCH \
    --domain materials \
    --lmdb-dir /lustre/orion/stf053/proj-shared/drai/HydraGNN/data/oc22/s2ef_total_train_val_test_lmdbs/data/oc22/s2ef-total/train \
    --npz-file /lustre/orion/stf053/scratch/whbrewer/redi_oc22_bench_output/oc22_redi_output.npz \
    --output-dir $OUT/materials \
    --n-samples 100000 \
    --n-reads 1000

# --- climate (ClimaX) ---
echo ""
echo "=== CLIMATE ==="
python $BENCH \
    --domain climate \
    --npz-file /lustre/orion/scratch/whbrewer/stf053/ClimaX/data.npz \
    --output-dir $OUT/climate \
    --n-reads 500 \
    --window 10

# --- fusion (XGC1) ---
echo ""
echo "=== FUSION (XGC1) ==="
python $BENCH \
    --domain fusion \
    --h5-file /lustre/orion/stf053/proj-shared/ketan/redi-all-data-processed/xgc.3d.00002.bp.outdir/data.h5 \
    --bp-dir /lustre/orion/world-shared/fus184/fusiond-seed-xgc1-data/n560fr_ITER_PFPO_W_Ne \
    --output-dir $OUT/fusion \
    --n-reads 500 \
    --window 1

echo ""
echo "== Done $(date) =="
