#!/bin/bash

#H="$(hostname)"
#echo This is the payload script. $1 is the argument passed to it. Ran on machine $H.

#set -x

redi --style=simple $1 $2 $3 $4 -o /mnt/bb/ketan2/$(basename $1).outdir &> /mnt/bb/ketan2/$(basename $1).stdoe

