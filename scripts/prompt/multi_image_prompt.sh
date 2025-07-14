#!/bin/bash

# 设置参数
MODEL_NAME="dinov2_large"
#MODEL_NAME="siglip"
ROOT_DIR="/map-vepfs/dehua/code/visual-memory"

source /map-vepfs/dehua/anaconda3/bin/activate
conda activate vllm
# 数据集名称数组
DATASET_NAMES=("foodx251")
FEW_SHOTs=(None)
#FEW_SHOTs=(4)
# k 值数组
K_VALUES=(20)  # 如果计划使用多个 k 值，定义为数组

cd $ROOT_DIR

# 遍历每个数据集名称
for DATASET_NAME in "${DATASET_NAMES[@]}"; do
    for FEW_SHOT in "${FEW_SHOTs[@]}"; do
        for K in "${K_VALUES[@]}"; do  # 迭代 K_VALUES
            python code/prompt/get_multi_image_prompt.py \
                --model_name $MODEL_NAME \
                --dataset_name $DATASET_NAME \
                --k $K \
                --root_dir $ROOT_DIR \
		        --few_shot $FEW_SHOT
            echo "Completed processing for dataset: $DATASET_NAME with few shot $FEW_SHOT and k $K"
        done
    done
done

