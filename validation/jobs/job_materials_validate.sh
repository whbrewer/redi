#!/bin/bash
#SBATCH --job-name=redi-val-materials
#SBATCH --account=stf053
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --output=validation/results/val_materials_%j.out

echo "== Start $(date) =="
source ~/venv/redi/bin/activate

python -u validation/runners/validate_materials.py \
    --lmdb-dir  /lustre/orion/stf053/proj-shared/drai/HydraGNN/data/oc22/s2ef_total_train_val_test_lmdbs/data/oc22/s2ef-total/train \
    --npz-file  /lustre/orion/stf053/scratch/whbrewer/redi_oc22_bench_output/oc22_redi_output.npz \
    --out validation/results/materials_validate.json

echo "== Done $(date) =="
