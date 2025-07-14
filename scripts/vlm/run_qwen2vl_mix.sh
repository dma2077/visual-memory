#!/bin/bash

export ROOT_DIR="/map-vepfs/dehua/code/visual-memory"  # 设置根目录路径
export PYTHONPATH=/map-vepfs/dehua/code/visual-memory
export TOKENIZERS_PARALLELISM=false

source /map-vepfs/dehua/anaconda3/bin/activate
conda activate vllm

# 定义要处理的数据集名称列表
DATASETS=("food101" "food172" "fru92" "veg200" "foodx251" "food2k")

checkpoint_iter="$1"  # 第一组参数
FEW_SHOT="$2"         # 第二组参数

# 使用第一个参数设置模型路径
if [ "$checkpoint_iter" == "None" ]; then
    MODEL_PATH='/map-vepfs/models/Qwen/Qwen2-VL-7B-Instruct-infer'
    echo $MODEL_PATH
else
    MODEL_PATH="/map-vepfs/dehua/model/checkpoints/qwen2vl/food2k_3_similarity/checkpoint-${checkpoint_iter}"
    echo $MODEL_PATH
fi

# Array of k values
K_VALUES=(5)
MODE='mix'
K=5
k1=3

for DATASET_NAME in "${DATASETS[@]}"; do
    GOLD_ANSWER_FILE_PATH="/map-vepfs/dehua/code/visual-memory/answers/groundtruth/${DATASET_NAME}_answers.jsonl"
    DATA_PROPORTION=1

    # 处理模型路径
    echo "Using dataset: $DATASET_NAME"
    echo "Model path: $MODEL_PATH"
    echo "Gold answer file path: $GOLD_ANSWER_FILE_PATH"

    if [ "$FEW_SHOT" == "None" ]; then
        SAVE_FILENAME="answers/${DATASET_NAME}/qwen2-vl-7b_food2k_3_similarity_${checkpoint_iter}.jsonl"
        DINO_FILE="questions/${DATASET_NAME}/dinov2_large_test_${K}_softmax.jsonl"
        SIGLIP_FILE="questions/${DATASET_NAME}/siglip_test_${K}_softmax.jsonl"
    else
        SAVE_FILENAME="answers/${DATASET_NAME}/qwen2-vl-7b_food2k_${FEW_SHOT}_3_similarity_${checkpoint_iter}.jsonl"
        DINO_FILE="questions/${DATASET_NAME}/dinov2_large_test_${K}_fewshot${FEW_SHOT}.jsonl"
        SIGLIP_FILE="questions/${DATASET_NAME}/siglip_test_${K}_fewshot${FEW_SHOT}.jsonl"
    fi

    echo "Save filename: $SAVE_FILENAME"
    echo "DINO file: $DINO_FILE"
    echo "SIGLIP file: $SIGLIP_FILE"
    echo "Data proportion: $DATA_PROPORTION"

    # 运行 Python 脚本
    python code/vlm/qwen2vl_vllm_text.py \
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
done



: '

wandb login f3b76ea66a38b2a211dc706fa95b02c761994b73
fewshot_number=None
gpu=0
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/llm/run_llama3_mix_old.sh $(( (gpu + 1) * 1000 )) "$fewshot_number" > gpu_$gpu.log &
gpu=1
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/llm/run_llama3_mix_old.sh $(( (gpu + 1) * 1000 )) "$fewshot_number" > gpu_$gpu.log &
gpu=2
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/llm/run_llama3_mix_old.sh $(( (gpu + 1) * 1000 )) "$fewshot_number" > gpu_$gpu.log &


wandb login f3b76ea66a38b2a211dc706fa95b02c761994b73
gpu=3
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/vlm/run_qwen2vl_mix.sh "None" "4" > gpu_$gpu.log &
'