#!/bin/bash

export ROOT_DIR="/home/madehua/code/visual-memory"  # 设置根目录路径
export PYTHONPATH=/home/madehua/code/visual-memory

# 定义要处理的数据集名称列表
DATASETS=("food101" "food172")

# 设置其他参数
MODEL_PATH='/mnt/madehua/model/llava-hf/llava-onevision-qwen2-7b-ov-hf'
MODEL_PATH='/mnt/madehua/model/Qwen/Qwen2-VL-7B-Instruct' 
# Array of k values
K_VALUES=(3 7 10 15 20)

for DATASET_NAME in "${DATASETS[@]}"; do
    GOLD_ANSWER_FILE_PATH="/mnt/madehua/fooddata/json_file/${DATASET_NAME}_answers.jsonl"
    LABEL_FILE="/mnt/madehua/fooddata/VireoFood172/SplitAndIngreLabel/FoodList.txt"
    DATA_PROPORTION=0.01

    # Loop through each k value
    for K in "${K_VALUES[@]}"; do
        SAVE_FILENAME="answers/multi_turn/${DATASET_NAME}/llava_onevision_k${K}.jsonl"
        QUESTION_FILE="questions/multi_image/${DATASET_NAME}/test_${K}.jsonl"

        # 打印参数以供调试
        echo "Using dataset: $DATASET_NAME"
        echo "Model path: $MODEL_PATH"
        echo "Gold answer file path: $GOLD_ANSWER_FILE_PATH"
        echo "Label file: $LABEL_FILE"
        echo "Save filename: $SAVE_FILENAME"
        echo "Question file: $QUESTION_FILE"
        echo "Data proportion: $DATA_PROPORTION"

        # 运行 Python 脚本
        python code/vlm/llava_onevison.py \
            --model_path "$MODEL_PATH" \
            --save_filename "$SAVE_FILENAME" \
            --gold_answer_file_path "$GOLD_ANSWER_FILE_PATH" \
            --label_file "$LABEL_FILE" \
            --question_file "$QUESTION_FILE" \
            --data_proportion "$DATA_PROPORTION"

        # 打印完成消息
        echo "VLM inference for $DATASET_NAME with k=$K completed. Results saved to $SAVE_FILENAME"
    done
done



"""
gpu=6
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/vlm/run_llava_onevision.sh > gpu_$gpu.log &
"""