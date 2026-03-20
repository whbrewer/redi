#!/bin/bash

#SBATCH -A stf053
#SBATCH -J redi-multinode
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 0:20:00
#SBATCH -p batch
#SBATCH -N 4
#SBATCH --ntasks-per-node 1
#SBATCH -C nvme

#conda init
module load miniforge3/23.11.0-0
conda activate conda-redi
module load parallel 

srun --no-kill --ntasks-per-node=1 --wait=0 driver.sh ./inputdatalist.txt

cp -r /mnt/bb/ketan2/*  /lustre/orion/proj-shared/stf053/ketan/redi-dump/

echo "== Done in $(($SECONDS / 60)) minutes=="
