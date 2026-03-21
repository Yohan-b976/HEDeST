#!/bin/bash

#SBATCH --job-name=hovernet
#SBATCH --output=/cluster/CBIO/home/lgortana/HEDeST/log/segment_%j.log
#SBATCH --error=/cluster/CBIO/home/lgortana/HEDeST/log/segment_%j.err
#SBATCH --gres=gpu:1
#SBATCH -p cbio-gpu
#SBATCH --cpus-per-task=8

echo 'Found a place!'

source /cluster/CBIO/home/lgortana/anaconda3/etc/profile.d/conda.sh
conda activate hovernet

export LD_LIBRARY_PATH=/cluster/CBIO/home/lgortana/anaconda3/envs/hovernet/lib:$LD_LIBRARY_PATH

python external/hovernet/run_infer.py \
    --gpu='0' \
    --nr_types=6 \
    --type_info_path=external/hovernet/type_info.json \
    --batch_size=16 \
    --model_mode=fast \
    --model_path=/cluster/CBIO/data1/lgortana/pretrained/seg_classif/hovernet_fast_pannuke_type_tf2pytorch.tar \
    --size_px=64 \
    --size_um=20 \
    --mpp=0.345 \
    wsi \
    --input_dir=/cluster/CBIO/data1/lgortana/Visium_FFPE_Human_Breast_Cancer/pyr_tif/ \
    --output_dir=/cluster/CBIO/data1/lgortana/Visium_FFPE_Human_Breast_Cancer/seg_json/ \
    --image_dict_path=/cluster/CBIO/data1/lgortana/Visium_FFPE_Human_Breast_Cancer/image_dict_64px_20um.pt \
    --input_mask_dir=/cluster/CBIO/data1/lgortana/Visium_FFPE_Human_Breast_Cancer/mask/lvl3/ \
    --cache_path=/cluster/CBIO/data1/lgortana/cache \
    --save_geojson
