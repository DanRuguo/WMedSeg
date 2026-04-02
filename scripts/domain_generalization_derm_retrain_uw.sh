#!/bin/bash

# Skin dermoscopy DG recipe v2
# Retrain on ISIC with stronger OOD-oriented derm augmentations,
# then evaluate on ISIC and UWaterlooSkinCancer.

run_id=$1
output_dir=$2
gpu_id=$3
prompt=${4:-original}

if [ -z "$run_id" ] || [ -z "$output_dir" ] || [ -z "$gpu_id" ]; then
    echo "Usage: bash scripts/domain_generalization_derm_retrain_uw_v2.sh <run_id> <output_dir> <gpu_id> [prompt_design]"
    exit 1
fi

set -e
mkdir -p "$output_dir"

CUDA_VISIBLE_DEVICES=$gpu_id python train.py \
    --seed "$run_id" \
    --config-file configs/ISIC.yaml \
    --output-dir "$output_dir"

for target in ISIC UWaterlooSkinCancer; do
    echo "Evaluating SOURCE=ISIC -> TARGET=${target}"
    CUDA_VISIBLE_DEVICES=$gpu_id python test.py \
        --seed "$run_id" \
        --config-file "configs/${target}.yaml" \
        --output-dir "$output_dir" \
        --source_dataset ISIC \
        --prompt_design "$prompt"

    CUDA_VISIBLE_DEVICES=$gpu_id python utils/eval.py \
        --seed "$run_id" \
        --config-file "configs/${target}.yaml" \
        --output-dir "$output_dir" \
        --prompt_design "$prompt"
done
