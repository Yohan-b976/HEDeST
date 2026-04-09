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


python3 -u hedest/main.py \
  /cluster/CBIO/data1/ybeaumatin/Xenium_FFPE_Human_Breast_Cancer_Rep1/sim/moco_embed_HB_64px_20um.pt \
  /cluster/CBIO/data1/ybeaumatin/Xenium_FFPE_Human_Breast_Cancer_Rep1/sim/sim_Xenium_FFPE_Human_Breast_Cancer_Rep1_prop_real.csv \
  --json-path /cluster/CBIO/data1/ybeaumatin/Xenium_FFPE_Human_Breast_Cancer_Rep1/sim/pannuke_fast_mask_lvl3_annotated.json \
  --path-st-adata /cluster/CBIO/data1/ybeaumatin/Xenium_FFPE_Human_Breast_Cancer_Rep1/sim/pseudo_adata_real.h5ad \
  --adata-name Xenium_FFPE_Human_Breast_Cancer_Rep1 \
  --spot-dict-file /cluster/CBIO/data1/ybeaumatin/Xenium_FFPE_Human_Breast_Cancer_Rep1/sim/spot_dict_adjust_real.json \
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
  --out-dir models/Xenium_FFPE/Test1/model_default_hidden_dim_512-256_norm_True_dropout_0.0_alpha_0.0_lr_0.0001_divergence_l2_beta_0.0 \
  --save-geojson \
  --rs 42

