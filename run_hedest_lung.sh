#!/bin/bash

#SBATCH --job-name=HEDeST
#SBATCH --output=/cluster/CBIO/home/ybeaumatin/HEDeST/log/hedest_%j.log
#SBATCH --error=/cluster/CBIO/home/ybeaumatin/HEDeST/log/hedest_%j.err
#SBATCH --gres=gpu:1
#SBATCH -p cbio-gpu
#SBATCH --cpus-per-task=8

echo 'Found a place!'

source /cluster/CBIO/home/ybeaumatin/miniconda3/etc/profile.d/conda.sh
conda activate hedest-env

export LD_LIBRARY_PATH=/cluster/CBIO/home/ybeaumatin/miniconda3/envs/hedest-env/lib:$LD_LIBRARY_PATH

for seed in {0..9}; do
  python3 -u hedest/main.py \
    /cluster/CBIO/data1/ybeaumatin/Xenium_V1_humanLung_Cancer_FFPE/sim/LuCA/moco_embed_HL_64px_20um.pt \
    /cluster/CBIO/data1/ybeaumatin/Xenium_V1_humanLung_Cancer_FFPE/hier_sim/lv0/proportions.csv \
    --json-path /cluster/CBIO/data1/ybeaumatin/Xenium_V1_humanLung_Cancer_FFPE/hier_sim/lv0/seg_dict.json \
    --path-st-adata /cluster/CBIO/data1/ybeaumatin/Xenium_V1_humanLung_Cancer_FFPE/sim/LuCA/pseudo_adata_real.h5ad \
    --adata-name Xenium_V1_humanLung_Cancer_FFPE \
    --spot-dict-file /cluster/CBIO/data1/ybeaumatin/Xenium_V1_humanLung_Cancer_FFPE/hier_sim/lv0/spot_dict.json \
    --model-name default \
    --norm \
    --dropout 0.0 \
    --batch-size 64 \
    --lr 1e-4 \
    --divergence l2 \
    --alpha 0 \
    --beta 0.0 \
    --epochs 100 \
    --train-size 0.8 \
    --val-size 0.1 \
    --out-dir models/Xenium_V1_humanLung_hierarchical_lv0/seed_${seed} \
    --save-geojson \
    --rs $seed
done
