#!/bin/bash
#SBATCH --job-name=redi-io-fusion
#SBATCH --account=stf053
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --output=openfold-io-bench-%j.out

set -e
echo "== Start $(date) =="

source ~/venv/redi/bin/activate

BENCH=benchmarks/redi_io_benchmark.py
OUT=/lustre/orion/scratch/whbrewer/stf053/openfold_io_bench

echo ""
echo "=== FUSION (XGC1) — HDF5 + Zarr + ADIOS2 native ==="
python $BENCH \
    --domain fusion \
    --h5-file /lustre/orion/stf053/proj-shared/ketan/redi-all-data-processed/xgc.3d.00002.bp.outdir/data.h5 \
    --bp-dir /lustre/orion/world-shared/fus184/fusiond-seed-xgc1-data/n560fr_ITER_PFPO_W_Ne \
    --output-dir $OUT/fusion_adios2 \
    --n-reads 500 \
    --window 1

echo ""
echo "== Done $(date) =="
