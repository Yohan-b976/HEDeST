#!/bin/bash

python3 external/LEMON/run_lemon.py \
  --image-dict /home/luca/Documents/data/Visium_FFPE_Human_Breast_Cancer/image_dict_40.pt \
  --output-path /home/luca/Documents/data/Visium_FFPE_Human_Breast_Cancer/lemon_embeddings.pt \
  --device cpu \
  --model-name vits8 \
  --cell-size 40 \
  --weights external/LEMON/pretrained/lemon.pth.tar \
  --stats external/LEMON/mean_std.json \
  --batch-size 2048 \
