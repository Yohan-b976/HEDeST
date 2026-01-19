#!/bin/bash

#SBATCH --job-name=lemon
#SBATCH --output=/cluster/CBIO/home/lgortana/HEDeST/log/lemon_%j.log
#SBATCH --error=/cluster/CBIO/home/lgortana/HEDeST/log/lemon_%j.err
#SBATCH --gres=gpu:P100:1
#SBATCH -p cbio-gpu
#SBATCH --exclude=node005,node009
#SBATCH --cpus-per-task=4

echo 'Found a place!'

source /cluster/CBIO/home/lgortana/anaconda3/etc/profile.d/conda.sh
conda activate lemon-env

export LD_LIBRARY_PATH=/cluster/CBIO/home/lgortana/anaconda3/envs/lemon-env/lib:$LD_LIBRARY_PATH

python3 external/LEMON/run_lemon.py \
  --image-dict /cluster/CBIO/data1/lgortana/Visium_FFPE_Human_Breast_Cancer/image_dict_40.pt \
  --output-path /cluster/CBIO/data1/lgortana/Visium_FFPE_Human_Breast_Cancer/lemon_embeddings_test.pt \
  --model-name vits8 \
  --cell-size 40 \
  --weights external/LEMON/pretrained/lemon.pth.tar \
  --stats external/LEMON/mean_std.json \
  --batch-size 2048 \
  --num-workers 4 \
