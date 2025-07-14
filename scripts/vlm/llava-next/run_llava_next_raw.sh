#!/bin/bash

export ROOT_DIR="/map-vepfs/dehua/code/visual-memory"  # 设置根目录路径
export PYTHONPATH=/map-vepfs/dehua/code/visual-memory


source /map-vepfs/dehua/anaconda3/bin/activate
conda activate vllm
# 定义要处理的数据集名称列表
DATASETS=("food172")
# 设置其他参数

#MODEL_PATH='/map-vepfs/dehua/model/checkpoints/llava-next/food101_raw'
#MODEL_PATH='/map-vepfs/dehua/model/checkpoints/llava-next/food172_raw/checkpoint-1032'
MODEL_PATH='/map-vepfs/models/llava-hf/llama3-llava-next-8b-hf'
FEW_SHOT="$1"         # 第二组参数

echo $MODEL_PATH
#MODEL_PATH='/map-vepfs/dehua/model/checkpoints/qwen2vl/food172_5/checkpoint-2068'
# Array of k values
K_VALUES=(20)
MODE='siglip'
K=20
k1="$2"
DATASET_NAME="$3"

GOLD_ANSWER_FILE_PATH="/map-vepfs/dehua/code/visual-memory/answers/groundtruth/${DATASET_NAME}_answers.jsonl"
DATA_PROPORTION=1
if [ "$FEW_SHOT" == "None" ]; then
    #MODEL_PATH="/map-vepfs/dehua/model/checkpoints/qwen2vl/${DATASET_NAME}_raw"
    #SAVE_FILENAME="answers/${DATASET_NAME}/qwen2vl_${DATASET_NAME}_raw.jsonl"
    SAVE_FILENAME="answers/${DATASET_NAME}/llava-next-${k1}.jsonl"
    DINO_FILE="questions/${DATASET_NAME}/dinov2_large_test_${K}_softmax_old.jsonl"
    SIGLIP_FILE="questions/${DATASET_NAME}/siglip_test_${K}_softmax.jsonl"
else
    #MODEL_PATH="/map-vepfs/dehua/model/checkpoints/qwen2vl/${DATASET_NAME}_raw_fewshot${FEW_SHOT}"
    #SAVE_FILENAME="answers/${DATASET_NAME}/qwen2vl_${DATASET_NAME}_raw_fewshot${FEW_SHOT}.jsonl"
    SAVE_FILENAME="answers/${DATASET_NAME}/qwen2vl_fewshot${FEW_SHOT}.jsonl"
    DINO_FILE="questions/${DATASET_NAME}/dinov2_large_test_${K}_fewshot${FEW_SHOT}_old.jsonl"
    SIGLIP_FILE="questions/${DATASET_NAME}/siglip_test_${K}_fewshot${FEW_SHOT}.jsonl"
fi
echo "Using dataset: $DATASET_NAME"
echo "Model path: $MODEL_PATH"
echo "Gold answer file path: $GOLD_ANSWER_FILE_PATH"
echo "Label file: $LABEL_FILE"
echo "Save filename: $SAVE_FILENAME"
echo "Question file: $QUESTION_FILE"
echo "Data proportion: $DATA_PROPORTION"

# 运行 Python 脚本
python code/vlm/llava_next_text_raw.py \
    --model_path "$MODEL_PATH" \
    --save_filename "$SAVE_FILENAME" \
    --gold_answer_file_path "$GOLD_ANSWER_FILE_PATH" \
    --label_file "$LABEL_FILE" \
    --question_file "$DINO_FILE" \
    --data_proportion "$DATA_PROPORTION" \
    --siglip_file "$SIGLIP_FILE" \
    --mode $MODE \
    --k1 $k1

# 打印完成消息
echo "VLM inference for $DATASET_NAME with checkpoint_iter=$checkpoint_iter completed. Results saved to $SAVE_FILENAME"




: '

wandb login f3b76ea66a38b2a211dc706fa95b02c761994b73





DATASET_NAME=food101
fewshot_number=None
gpu=0
k1=5
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/vlm/llava-next/run_llava_next_raw.sh "$fewshot_number" "$k1" "$DATASET_NAME" > log/llava_next_101_${k1}.log 2>&1 &

fewshot_number=None
gpu=1
k1=10
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/vlm/llava-next/run_llava_next_raw.sh "$fewshot_number" "$k1" "$DATASET_NAME" > log/llava_next_101_${k1}.log 2>&1 &


fewshot_number=None
gpu=2
k1=15
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/vlm/llava-next/run_llava_next_raw.sh "$fewshot_number" "$k1" "$DATASET_NAME" > log/llava_next_101_${k1}.log 2>&1 &


fewshot_number=None
gpu=3
k1=20
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/vlm/llava-next/run_llava_next_raw.sh "$fewshot_number" "$k1" "$DATASET_NAME" > log/llava_next_101_${k1}.log 2>&1 &


DATASET_NAME=food172
fewshot_number=None
gpu=4
k1=5
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/vlm/llava-next/run_llava_next_raw.sh "$fewshot_number" "$k1" "$DATASET_NAME" > log/llava_next_172_${k1}.log 2>&1 &

fewshot_number=None
gpu=5
k1=10
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/vlm/llava-next/run_llava_next_raw.sh "$fewshot_number" "$k1" "$DATASET_NAME" > log/llava_next_172_${k1}.log 2>&1 &


fewshot_number=None
gpu=6
k1=15
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/vlm/llava-next/run_llava_next_raw.sh "$fewshot_number" "$k1" "$DATASET_NAME" > log/llava_next_172_${k1}.log 2>&1 &


fewshot_number=None
gpu=7
k1=20
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/vlm/llava-next/run_llava_next_raw.sh "$fewshot_number" "$k1" "$DATASET_NAME" > log/llava_next_172_${k1}.log 2>&1

sleep 1200

'