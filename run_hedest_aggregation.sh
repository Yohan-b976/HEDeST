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
    --run-dir /cluster/CBIO/home/ybeaumatin/HEDeST/models/Xenium_FFPE_hierarchical_lv0 \
    --json-path /cluster/CBIO/data1/ybeaumatin/Xenium_FFPE_Human_Breast_Cancer_Rep1/hier_sim/lv0/seg_dict.json \

