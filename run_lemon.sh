#!/bin/bash

#SBATCH --job-name=lemon
#SBATCH --output=/cluster/CBIO/home/lgortana/HEDeST/log/lemon_%j.log
#SBATCH --error=/cluster/CBIO/home/lgortana/HEDeST/log/lemon_%j.err
#SBATCH --gres=gpu:1
#SBATCH -p cbio-gpu
#SBATCH --cpus-per-task=4

echo 'Found a place!'

source /cluster/CBIO/home/lgortana/anaconda3/etc/profile.d/conda.sh
conda activate lemon-env

export LD_LIBRARY_PATH=/cluster/CBIO/home/lgortana/anaconda3/envs/lemon-env/lib:$LD_LIBRARY_PATH

python3 external/LEMON/run_lemon.py \
  --image-dict /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/image_dict_40.pt \
  --output-path /cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/lemon_embeddings_test.pt \
  --model-name vits8 \
  --cell-size 40 \
  --weights external/LEMON/pretrained/lemon.pth.tar \
  --stats external/LEMON/mean_std.json \
  --batch-size 2048 \
  --num-workers 4 \

python3 external/LEMON/run_lemon.py \
  --image-dict /cluster/CBIO/data1/lgortana/Xenium_FFPE_Human_Breast_Cancer_Rep1/image_dict_40.pt \
  --output-path /cluster/CBIO/data1/lgortana/Xenium_FFPE_Human_Breast_Cancer_Rep1/lemon_embeddings_test.pt \
  --model-name vits8 \
  --cell-size 40 \
  --weights external/LEMON/pretrained/lemon.pth.tar \
  --stats external/LEMON/mean_std.json \
  --batch-size 2048 \
  --num-workers 4 \

python3 external/LEMON/run_lemon.py \
  --image-dict /cluster/CBIO/data1/lgortana/Xenium_V1_humanLung_Cancer_FFPE/image_dict_40.pt \
  --output-path /cluster/CBIO/data1/lgortana/Xenium_V1_humanLung_Cancer_FFPE/lemon_embeddings_test.pt \
  --model-name vits8 \
  --cell-size 40 \
  --weights external/LEMON/pretrained/lemon.pth.tar \
  --stats external/LEMON/mean_std.json \
  --batch-size 2048 \
  --num-workers 4 \
