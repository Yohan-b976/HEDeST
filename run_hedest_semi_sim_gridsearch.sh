#!/bin/bash

# ---------------------
# Arguments for GPU job
# ---------------------

IMAGE_DICT="/cluster/CBIO/data1/ybeaumatin/Xenium_FFPE_Human_Breast_Cancer_Rep1/sim/moco_embed_HB_64px_20um.pt"
SIM_CSV="/cluster/CBIO/data1/ybeaumatin/Xenium_FFPE_Human_Breast_Cancer_Rep1/sim/sim_Xenium_FFPE_Human_Breast_Cancer_Rep1_prop_real.csv"
JSON_PATH="/cluster/CBIO/data1/ybeaumatin/Xenium_FFPE_Human_Breast_Cancer_Rep1/sim/pannuke_fast_mask_lvl3_annotated.json"
ADATA_PATH="/cluster/CBIO/data1/ybeaumatin/Xenium_FFPE_Human_Breast_Cancer_Rep1/sim/pseudo_adata_real.h5ad"
ADATA_NAME="Xenium_FFPE_Human_Breast_Cancer_Rep1"
SPOT_DICT="/cluster/CBIO/data1/ybeaumatin/Xenium_FFPE_Human_Breast_Cancer_Rep1/sim/spot_dict_adjust_real.json"

SEED_LIST=$(seq 0 2)

GPU_EXTRA_ARGS=(
    --models default
    --hidden_dims "512,256"
    --norm_options 0 1
    --dropouts 0.0 0.1
    --alphas 0 0.001 0.01 0.1
    --betas 0
    --learning_rates 1e-4 3e-4 1e-3 3e-3
    --divergences kl l2
    --seeds $SEED_LIST
    --batch_size 64
    --out_dir models/simulations-v2/moco-64px-20um-semi-sim-HL-7types
)

# ---------------------
# Arguments for CPU job
# ---------------------

FOLDER="models/simulations-v2/moco-64px-20um-semi-sim-HL-7types"
GT_CSV="/cluster/CBIO/data1/ybeaumatin/Xenium_FFPE_Human_Breast_Cancer_Rep1/sim/sim_Xenium_FFPE_Human_Breast_Cancer_Rep1_gt.csv"

# ---------------------
# Submit GPU job
# ---------------------

GPU_JOB_ID=$(sbatch --parsable run_gridsearch.sh "$IMAGE_DICT" "$SIM_CSV" "$JSON_PATH" "$ADATA_PATH" "$ADATA_NAME" "$SPOT_DICT" "${GPU_EXTRA_ARGS[@]}")

# ---------------------
# Submit CPU job (depends on GPU)
# ---------------------

CPU_JOB_ID=$(sbatch --parsable --dependency=afterok:$GPU_JOB_ID run_extract_stats.sh "$FOLDER" "$GT_CSV")
