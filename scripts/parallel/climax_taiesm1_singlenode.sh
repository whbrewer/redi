#!/bin/bash

#SBATCH -A stf053
#SBATCH -J redi-climax-taiesm1
#SBATCH -o redi-climax-taiesm1-%j.out
#SBATCH -e redi-climax-taiesm1-%j.out
#SBATCH -t 4:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

DATA_DIR=/lustre/orion/stf053/world-shared/vga/projects/ClimaX/data/TaiESM1/raw/2m_temperature
OUT_DIR=/lustre/orion/scratch/whbrewer/stf053/ClimaX
CONFIG=/ccs/home/whbrewer/drai/redi/examples/climate/climax_taiesm1.yaml

mkdir -p $OUT_DIR

source /ccs/home/whbrewer/venv/redi/bin/activate

echo "=== REDI ClimaX TaiESM1 pipeline === $(date)"
T0=$SECONDS

redi run "$DATA_DIR" \
    -c "$CONFIG" \
    -o "$OUT_DIR" \
    --style simple \
    -v

echo "== Done in $(($SECONDS - T0)) seconds ($(( ($SECONDS - T0) / 60 )) minutes) == $(date)"
