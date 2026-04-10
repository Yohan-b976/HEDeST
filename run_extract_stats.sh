#!/bin/bash

#SBATCH --job-name=extract_stats
#SBATCH --output=/cluster/CBIO/home/ybeaumatin/HEDeST/log/extract_stats_%j.log
#SBATCH --error=/cluster/CBIO/home/ybeaumatin/HEDeST/log/extract_stats_%j.err
#SBATCH -p cbio-cpu
#SBATCH --cpus-per-task=8

echo 'Found a place!'

source /cluster/CBIO/home/ybeaumatin/miniconda3/etc/profile.d/conda.sh
conda activate hedest-env

export LD_LIBRARY_PATH=/cluster/CBIO/home/ybeaumatin/miniconda3/envs/hedest-env/lib:$LD_LIBRARY_PATH

FOLDER=$1
GT_CSV=$2

python3 -u hedest/compute_stats_processor.py "$FOLDER" "$GT_CSV"
