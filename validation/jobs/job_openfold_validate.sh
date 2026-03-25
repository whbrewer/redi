#!/bin/bash
#SBATCH --job-name=redi-val-openfold
#SBATCH --account=stf053
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --output=validation/results/val_openfold_%j.out

echo "== Start $(date) =="
source ~/venv/redi/bin/activate

python -u validation/runners/validate_openfold.py \
    --npz-dir  /lustre/orion/gen053/world-shared/openfold/redi_output/train \
    --mmcif-dir /lustre/orion/gen053/world-shared/openfold/data/pdb_mmcif/mmcif_files \
    --alignment-dir /lustre/orion/gen053/world-shared/openfold/data/open_protein_set/pdb \
    --n-samples 500 \
    --out validation/results/openfold_validate.json

echo "== Done $(date) =="
