#!/bin/bash


export ROOT_DIR="/map-vepfs/dehua/code/visual-memory"  # 设置根目录路径
export PYTHONPATH=/map-vepfs/dehua/code/visual-memory


source /map-vepfs/dehua/anaconda3/bin/activate
conda activate vllm
# 设置参数
# DATASET_NAME="food101"  # 可选: food101, food2k, food172, foodx251, fru92, veg200
#DATASETS=("food101-lt")
DATASETS=("foodx251")
MODEL_NAME="dinov2_large"
MODEL_PATH="/map-vepfs/models/facebook/dinov2-large"  # 更新后的模型路径
CHECKPOINT_PATH="/map-vepfs/dehua/model/dinov2_vitl14_reg4_pretrain.pth"  # 添加的 checkpoint 路径
OUTPUT_DIR="/map-vepfs/dehua/model/food_embeddings/dinov2_noreg"  # 更新后的输出目录
FEW_SHOTs=(None)  # 每个类别的样本数
SEED=42  # 随机种子
BATCH_SIZE=32
NUM_WORKERS=8
ROOT_DIR="/map-vepfs/dehua/code/visual-memory"

cd $ROOT_DIR/code/encoder
# 运行 Python 脚本

for DATASET_NAME in "${DATASETS[@]}"; do
    for FEW_SHOT in "${FEW_SHOTs[@]}"; do
        python get_embeddings_dinov2.py \
            --dataset_name $DATASET_NAME \
            --model_name $MODEL_NAME \
            --model_path $MODEL_PATH \
            --output_dir $OUTPUT_DIR \
            --checkpoint_path $CHECKPOINT_PATH \
            --few_shot $FEW_SHOT \
            --seed $SEED \
            --batch_size $BATCH_SIZE \
            --num_workers $NUM_WORKERS

        # 打印完成消息
        echo "Embedding extraction completed for $DATASET_NAME using $CHECKPOINT_PATH with few shot $FEW_SHOT"
    done
done

: '
gpu=0
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/encode/get_embedding_dinov2.sh > gpu_$gpu.log &
'