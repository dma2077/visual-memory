#!/bin/bash

# 设置参数
#MODEL_NAME="dinov2_large"
MODEL_NAME="clip"
ROOT_DIR="/map-vepfs/dehua/code/visual-memory"

source /map-vepfs/dehua/anaconda3/bin/activate
conda activate vllm

# 数据集名称数组
DATASET_NAMES=("food101" "food172" "food2k" "fru92" "veg200" "foodx251")
#FEW_SHOTs=(None 4 8 16)
FEW_SHOTs=(8)
# k 值数组
K_VALUES=(5)  # 如果计划使用多个 k 值，定义为数组

# 添加训练集名称数组（用逗号分隔的字符串）
TRAIN_DATASETS="food101,food172,food2k,fru92,veg200,foodx251"  # 请替换为实际的训练集名称

cd $ROOT_DIR

# 遍历每个数据集名称
for DATASET_NAME in "${DATASET_NAMES[@]}"; do
    for FEW_SHOT in "${FEW_SHOTs[@]}"; do
        for K in "${K_VALUES[@]}"; do  # 迭代 K_VALUES
            python code/prompt/get_multi_image_prompt_mix_datasets.py \
                --model_name $MODEL_NAME \
                --dataset_name $DATASET_NAME \
                --k $K \
                --root_dir $ROOT_DIR \
                --few_shot $FEW_SHOT \
                --train_datasets $TRAIN_DATASETS
            echo "Completed processing for dataset: $DATASET_NAME with few shot $FEW_SHOT, k $K, and train datasets $TRAIN_DATASETS"
        done
    done
done

