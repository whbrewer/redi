# OpenFold Multi-Node Parallel Processing Plan

## Core Idea

The existing `driver.sh` already handles multi-node round-robin distribution via
GNU parallel + SLURM. We restructure OpenFold processing so each **protein chain**
is an independent work unit, matching the existing `payload.sh` pattern.

## New Files Needed

### 1. `scripts/gen_openfold_chain_list.py`

Pre-generates `protein_list.txt` — scans OpenProteinSet once, assigns
train/val/test splits upfront (so each payload doesn't need to know the total
count), outputs one line per chain (4 tab-separated fields):

```
101m_A  train  /lustre/.../pdb_mmcif/mmcif_files/101m.cif  /lustre/.../open_protein_set/pdb/101m_A/a3m
2abc_B  val    /lustre/.../pdb_mmcif/mmcif_files/2abc.cif  /lustre/.../open_protein_set/pdb/2abc_B/a3m
```

Logic:
- Pre-scan `pdb_mmcif/mmcif_files/` into a set (one listing, not per-chain exists())
- Scan `open_protein_set/pdb/` for chain dirs
- Skip chains with no matching mmCIF
- Randomly assign 80/10/10 train/val/test split with fixed seed

---

### 2. `scripts/process_openfold_chain.py`

Processes a **single chain** end-to-end → one NPZ file.

Arguments: `--chain_id`, `--cif_path`, `--alignment_dir`, `--split`, `--out_dir`

Reuses existing helper functions from `redi/domains/openfold.py`:
- `_parse_mmcif(cif_path)`
- `_parse_a3m(a3m_path)`
- `_sequence_to_aatype(seq)`
- `_encode_msa_row(seq)`

Writes: `out_dir/{split}/{chain_id}.npz`

Skips silently if output NPZ already exists (safe to rerun/resume).

---

### 3. `parallelize/payload_openfold.sh`

```bash
#!/bin/bash
# Args: chain_id  split  cif_path  alignment_dir
CHAIN_ID=$1
SPLIT=$2
CIF=$3
ALN=$4

OUT_DIR=/lustre/orion/gen053/world-shared/openfold/redi_output

source /ccs/home/whbrewer/venv/redi/bin/activate

python /ccs/home/whbrewer/drai/redi/scripts/process_openfold_chain.py \
    --chain_id  "$CHAIN_ID" \
    --split     "$SPLIT" \
    --cif_path  "$CIF" \
    --alignment_dir "$ALN" \
    --out_dir   "$OUT_DIR"
```

---

### 4. `parallelize/openfold_multinode.sh`

```bash
#!/bin/bash
#SBATCH -A gen053
#SBATCH -J redi-openfold-multinode
#SBATCH -o redi-openfold-multinode-%j.out
#SBATCH -e redi-openfold-multinode-%j.out
#SBATCH -t 4:00:00
#SBATCH -p batch
#SBATCH -N 16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

WORK_DIR=/ccs/home/whbrewer/drai/redi
DATA_DIR=/lustre/orion/gen053/world-shared/openfold/data
CHAIN_LIST=$WORK_DIR/parallelize/protein_list.txt

source /ccs/home/whbrewer/venv/redi/bin/activate
module load parallel

echo "=== Step 1: Generate chain list === $(date)"
if [ ! -f "$CHAIN_LIST" ]; then
    python "$WORK_DIR/scripts/gen_openfold_chain_list.py" \
        "$DATA_DIR" "$CHAIN_LIST"
else
    echo "  Chain list already exists ($(wc -l < $CHAIN_LIST) chains), skipping."
fi

echo "=== Step 2: Distribute across nodes === $(date)"
T0=$SECONDS
srun --no-kill --ntasks-per-node=1 --wait=0 \
    "$WORK_DIR/parallelize/driver.sh" "$CHAIN_LIST"

echo "== Done in $(($SECONDS - T0)) seconds == $(date)"
```

---

### 5. Tweak to `parallelize/driver.sh`

Change `-j56 -N4` → `-j32 -N4` to match Andes' 32 cores per node:

```bash
cat $1 | \
awk -v NNODE="$SLURM_NNODES" -v NODEID="$SLURM_NODEID" \
'NR % NNODE == NODEID' | \
parallel -j32 -N4 ./payload_openfold.sh {}
```

Or better, use a separate `driver_openfold.sh` to avoid changing the original.

---

## Expected Performance

| Config | Chains | Workers | Time/chain | Wall time |
|---|---|---|---|---|
| Current (1 node, 1 worker) | 131,315 | 1 | ~2s | ~18h |
| Single node, 32 workers | 131,315 | 32 | ~2s | ~35 min |
| 16 nodes × 32 workers | 131,315 | 512 | ~2s | ~5 min |

---

## Resume Safety

`process_openfold_chain.py` should skip chains where the output NPZ already
exists. This makes the job safe to rerun if it fails partway through — just
resubmit and it will only process the missing chains.
