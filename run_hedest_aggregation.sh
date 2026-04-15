#!/bin/bash

#SBATCH --job-name=agg_HEDeST
#SBATCH --output=/cluster/CBIO/home/ybeaumatin/HEDeST/log/agg_hedest_%j.log
#SBATCH --error=/cluster/CBIO/home/ybeaumatin/HEDeST/log/agg_hedest_%j.err
#SBATCH --gres=gpu:1
#SBATCH -p cbio-gpu
#SBATCH --cpus-per-task=8

echo 'Found a place!'

source /cluster/CBIO/home/ybeaumatin/miniconda3/etc/profile.d/conda.sh
conda activate hedest-env

export LD_LIBRARY_PATH=/cluster/CBIO/home/ybeaumatin/miniconda3/envs/hedest-env/lib:$LD_LIBRARY_PATH

python3 -u hedest/aggregate_seeds.py \
    --run-dir /cluster/CBIO/home/ybeaumatin/HEDeST/models/Xenium_V1_humanLung_hierarchical_lv3 \
    --json-path /cluster/CBIO/data1/ybeaumatin/Xenium_V1_humanLung_Cancer_FFPE/hier_sim/lv3/seg_dict.json \
