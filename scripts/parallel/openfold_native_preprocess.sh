#!/bin/bash

#SBATCH -A gen053
#SBATCH -J openfold-preprocess
#SBATCH -o openfold-preprocess-%j.out
#SBATCH -e openfold-preprocess-%j.out
#SBATCH -t 24:00:00
#SBATCH -p batch
#SBATCH -N 1

CODE=/lustre/orion/gen053/world-shared/openfold/code
DATA=/lustre/orion/gen053/world-shared/openfold/data
OUT=/lustre/orion/gen053/world-shared/openfold/out

source /ccs/home/whbrewer/venv/redi/bin/activate
export PYTHONPATH=/lustre/orion/gen053/world-shared/openfold/code:$PYTHONPATH

mkdir -p "$OUT/alignment_dbs"

TOTAL_START=$SECONDS

echo "=== Step 1: mmCIF cache === $(date)"
if [ -f "$OUT/mmcif_cache.json" ]; then
    echo "    -> already exists, skipping"
else
    T0=$SECONDS
    python "$CODE/scripts/generate_mmcif_cache.py" \
        "$DATA/pdb_mmcif/mmcif_files" \
        "$OUT/mmcif_cache.json" \
        --no_workers 32
    echo "    -> done in $(($SECONDS - T0)) seconds"
fi

echo "=== Step 2: Chain data cache === $(date)"
if [ -f "$OUT/chain_data_cache.json" ]; then
    echo "    -> already exists, skipping"
else
    T0=$SECONDS
    python "$CODE/scripts/generate_chain_data_cache.py" \
        "$DATA/pdb_mmcif/mmcif_files" \
        "$OUT/chain_data_cache.json" \
        --no_workers 32
    echo "    -> done in $(($SECONDS - T0)) seconds"
fi

echo "=== Step 3: Alignment DB === $(date)"
if [ -f "$OUT/alignment_dbs/alignment_db.index" ]; then
    echo "    -> already exists, skipping"
else
    T0=$SECONDS
    python "$CODE/scripts/alignment_db_scripts/create_alignment_db_sharded.py" \
        "$DATA/open_protein_set/pdb" \
        "$OUT/alignment_dbs" \
        alignment_db \
        --n_shards 10
    echo "    -> done in $(($SECONDS - T0)) seconds"
fi

echo "== Done in $(( ($SECONDS - TOTAL_START) / 60 )) minutes == $(date)"
