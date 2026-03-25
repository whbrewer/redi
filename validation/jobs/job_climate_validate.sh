#!/bin/bash
#SBATCH --job-name=redi-val-climate
#SBATCH --account=stf053
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --output=validation/results/val_climate_%j.out

echo "== Start $(date) =="
source ~/venv/redi/bin/activate

# 2m temperature (all chunks)
bash validation/runners/validate_climate.sh

# 3D variables (ta, ua, va) for one representative chunk
bash validation/runners/validate_climate_3d.sh

echo "== Done $(date) =="
