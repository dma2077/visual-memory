#!/bin/bash


export ROOT_DIR="/home/madehua/code/visual-memory"  # 设置根目录路径
export PYTHONPATH=/map-vepfs/dehua/code/visual-memory/code
# 设置参数

# source /map-vepfs/dehua/anaconda3/bin/activate
# conda activate vllm
# DATASET_NAME="food101"  # 可选: food101, food2k, food172, foodx251, fru92, veg200
DATASETS=("food101-lt")
MODEL_NAME="siglip"
MODEL_PATH="/map-vepfs/models/google/siglip-so400m-patch14-384"  # 更新后的模型路径
OUTPUT_DIR="/map-vepfs/dehua/model/food_embeddings/siglip"  # 更新后的输出目录
FEW_SHOTs="$1"  # 第一组参数
SEED=42  # 随机种子
BATCH_SIZE=16
NUM_WORKERS=8
ROOT_DIR="/map-vepfs/dehua/code/visual-memory"

cd $ROOT_DIR
# 运行 Python 脚本

for DATASET_NAME in "${DATASETS[@]}"; do
    python code/encoder/get_embeddings_siglip.py \
        --dataset_name $DATASET_NAME \
        --model_name $MODEL_NAME \
        --model_path $MODEL_PATH \
        --output_dir $OUTPUT_DIR \
        --few_shot $FEW_SHOT \
        --seed $SEED \
        --batch_size $BATCH_SIZE \
        --num_workers $NUM_WORKERS

    # 打印完成消息
    echo "Embedding extraction completed for $DATASET_NAME using $CHECKPOINT_PATH with few shot $FEW_SHOT"
done

"""
gpu=0
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/encode/get_embedding_siglip.sh "None" > embedding_clip$gpu.log &
"""