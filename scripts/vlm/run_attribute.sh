#!/bin/bash

DATASETS=("food101" "food172" "fru92" "veg200" "foodx251" "food2k")

MODEL_PATH=$1        # 第一组参数: MODEL_PATH
FEW_SHOT=$2          # 第二组参数: FEW_SHOT
MODE=$3              # 第三组参数: MODE
DATASET=$4           # 第四组参数: DATASET (Single string)
k1=$5                # 第五组参数: k1
GPU=$6

# Set constant values
K=5
DATA_PROPORTION=1
ROOT_DIR="/map-vepfs/dehua/code/visual-memory"  # 根目录路径
PYTHONPATH="/map-vepfs/dehua/code/visual-memory"

export ROOT_DIR
export PYTHONPATH
source /map-vepfs/dehua/anaconda3/bin/activate
conda activate vllm
# 设置模型路径
echo "Model path: $MODEL_PATH"
echo "Few-shot: $FEW_SHOT"
echo "Mode: $MODE"
echo "Dataset: $DATASET"
echo "K: $K"
echo "k1: $k1"

GOLD_ANSWER_FILE_PATH="/map-vepfs/dehua/code/visual-memory/answers/groundtruth/${DATASET}_answers.jsonl"


model_name=$(basename $MODEL_PATH)

SAVE_FILENAME="answers/${DATASET}/${model_name}_train_attribute.jsonl"
DINO_FILE="questions/${DATASET}/train_attribute.jsonl"


export CUDA_VISIBLE_DEVICES=$GPU  # 指定使用 GPU 0
echo "Using dataset: $DATASET"
echo "Gold answer file path: $GOLD_ANSWER_FILE_PATH"
echo "Save filename: $SAVE_FILENAME"
echo "DINO file: $DINO_FILE"
echo "Data proportion: $DATA_PROPORTION"

# 运行 Python 脚本
python /map-vepfs/dehua/code/visual-memory/code/vlm/inference_vllm.py \
    --model_path "$MODEL_PATH" \
    --save_filename "$SAVE_FILENAME" \
    --gold_answer_file_path "$GOLD_ANSWER_FILE_PATH" \
    --label_file "$LABEL_FILE" \
    --question_file "$DINO_FILE" \
    --data_proportion "$DATA_PROPORTION" \
    --siglip_file "$SIGLIP_FILE" \
    --mode "$MODE" \
    --k1 "$k1"

# 打印完成消息
echo "VLM inference for $DATASET completed. Results saved to $SAVE_FILENAME"

