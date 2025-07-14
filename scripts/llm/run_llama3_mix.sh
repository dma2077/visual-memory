#!/bin/bash

export ROOT_DIR="/map-vepfs/dehua/code/visual-memory"  # 设置根目录路径
export PYTHONPATH=/map-vepfs/dehua/code/visual-memory
export TOKENIZERS_PARALLELISM=false

source /map-vepfs/dehua/anaconda3/bin/activate
conda activate vllm

# 定义要处理的数据集名称列表
DATASETS=("veg200")
#DATASETS=("foodx251")
checkpoint_iter="$1"  # 第一组参数
FEW_SHOT="$2"         # 第二组参数

# 使用第一个参数设置模型路径
if [ "$checkpoint_iter" == "None" ]; then
    MODEL_PATH='/map-vepfs/models/meta-llama/Meta-Llama-3-8B-Instruct'
    echo $MODEL_PATH
else
    MODEL_PATH="/map-vepfs/dehua/model/checkpoints/llama3/food2k_3_similarity_text_fewshot_and_all/checkpoint-${checkpoint_iter}"
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
        SAVE_FILENAME="answers/${DATASET_NAME}/llama3_food2k_3_similarity_fewshot_and_all_${checkpoint_iter}.jsonl"
        DINO_FILE="questions/${DATASET_NAME}/dinov2_large_test_${K}_softmax.jsonl"
        SIGLIP_FILE="questions/${DATASET_NAME}/siglip_test_${K}_softmax.jsonl"
    else
        SAVE_FILENAME="answers/${DATASET_NAME}/llama3_food2k_3_similarity_fewshot_and_all_${checkpoint_iter}_fewshot${FEW_SHOT}.jsonl"
        DINO_FILE="questions/${DATASET_NAME}/dinov2_large_test_${K}_fewshot${FEW_SHOT}.jsonl"
        SIGLIP_FILE="questions/${DATASET_NAME}/siglip_test_${K}_fewshot${FEW_SHOT}.jsonl"
    fi

    echo "Save filename: $SAVE_FILENAME"
    echo "DINO file: $DINO_FILE"
    echo "SIGLIP file: $SIGLIP_FILE"
    echo "Data proportion: $DATA_PROPORTION"

    # 运行 Python 脚本
    python code/llm/llama3_vllm_text.py \
        --model_path "$MODEL_PATH" \
        --save_filename "$SAVE_FILENAME" \
        --gold_answer_file_path "$GOLD_ANSWER_FILE_PATH" \
        --label_file "$LABEL_FILE" \
        --question_file "$DINO_FILE" \
        --data_proportion "$DATA_PROPORTION" \
        --siglip_file "$SIGLIP_FILE" \
        --mode $MODE \
        --k1 $k1 \
        --dataset_name $DATASET_NAME
    # 打印完成消息
    echo "VLM inference for $DATASET_NAME with checkpoint_iter=$checkpoint_iter completed. Results saved to $SAVE_FILENAME"
done



: '
wandb login f3b76ea66a38b2a211dc706fa95b02c761994b73
fewshot_number=None
gpu=0
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/llm/run_llama3_mix.sh $(( (gpu + 1) * 3000 )) "$fewshot_number" > gpu_$gpu.log &
gpu=1
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/llm/run_llama3_mix.sh $(( (gpu + 1) * 3000 )) "$fewshot_number" > gpu_$gpu.log &

fewshot_number=4
gpu=2
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/llm/run_llama3_mix.sh $(( (gpu - 1) * 3000 )) "$fewshot_number" > gpu_$gpu.log &
gpu=3
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/llm/run_llama3_mix.sh $(( (gpu - 1) * 3000 )) "$fewshot_number" > gpu_$gpu.log &

fewshot_number=8
gpu=4
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/llm/run_llama3_mix.sh $(( (gpu - 3) * 3000 )) "$fewshot_number" > gpu_$gpu.log &
gpu=5
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/llm/run_llama3_mix.sh $(( (gpu - 3) * 3000 )) "$fewshot_number" > gpu_$gpu.log &
gpu=6
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/llm/run_llama3_mix.sh $(( (gpu + 1) * 1000 )) "$fewshot_number" > gpu_$gpu.log &
gpu=7
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/llm/run_llama3_mix.sh $(( (gpu + 1) * 1000 )) "$fewshot_number" > gpu_$gpu.log &



fewshot_number=None
gpu=3
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/llm/run_llama3_mix.sh $(( (gpu - 2) * 3000 )) "$fewshot_number" > gpu_$gpu.log &
gpu=4
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/llm/run_llama3_mix.sh $(( (gpu - 2) * 3000 )) "$fewshot_number" > gpu_$gpu.log &
gpu=2
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/llm/run_llama3_mix.sh $(( (gpu + 1) * 3000 )) "$fewshot_number" > gpu_$gpu.log &
gpu=3
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/llm/run_llama3_mix.sh $(( (gpu + 1) * 3000 )) "$fewshot_number" > gpu_$gpu.log &
gpu=4
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/llm/run_llama3_mix.sh $(( (gpu + 1) * 3000 )) "$fewshot_number" > gpu_$gpu.log &
gpu=5
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/llm/run_llama3_mix.sh $(( (gpu + 1) * 3000 )) "$fewshot_number" > gpu_$gpu.log &
gpu=6
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/llm/run_llama3_mix.sh $(( (gpu + 1) * 3000 )) "None" > gpu_$gpu.log &
gpu=7
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/llm/run_llama3_mix.sh $(( (gpu + 1) * 3000 )) "None" > gpu_$gpu.log &
'