#!/bin/bash

#SBATCH -A gen053
#SBATCH -J redi-openfold
#SBATCH -o redi-openfold-%j.out
#SBATCH -e redi-openfold-%j.out
#SBATCH -t 24:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

DATA_DIR=/lustre/orion/gen053/world-shared/openfold/data
OUT_DIR=/lustre/orion/gen053/world-shared/openfold/redi_output

source /ccs/home/whbrewer/venv/redi/bin/activate

echo "=== REDI OpenFold pipeline === $(date)"
T0=$SECONDS

redi run "$DATA_DIR" \
    -c /ccs/home/whbrewer/drai/redi/examples/bio/openfold.yaml \
    -o "$OUT_DIR" \
    --style simple \
    -v

echo "== Done in $(($SECONDS - T0)) seconds ($(( ($SECONDS - T0) / 60 )) minutes) == $(date)"
