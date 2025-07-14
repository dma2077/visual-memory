#!/bin/bash

export ROOT_DIR="/map-vepfs/dehua/code/visual-memory"  # 设置根目录路径
export PYTHONPATH=/map-vepfs/dehua/code/visual-memory


source /map-vepfs/dehua/anaconda3/bin/activate
conda activate vllm
# 定义要处理的数据集名称列表
DATASETS=("food101" "food172")

# 设置其他参数
MODEL_PATH='/map-vepfs/dehua/model/checkpoints/qwen2vl/food_mix_5/checkpoint-5000'
MODEL_PATH='/map-vepfs/models/Qwen/Qwen2-VL-7B-Instruct'
checkpoint_iters=(1032 2064)
MODEL_PATH="/map-vepfs/dehua/model/checkpoints/qwen2vl/food172_dinov2_ns/checkpoint-${checkpoint_iter}"
echo $MODEL_PATH
#MODEL_PATH='/map-vepfs/dehua/model/checkpoints/qwen2vl/food172_5/checkpoint-2068'
# Array of k values
K_VALUES=(5)
MODE='dinov2'
K=5

for DATASET_NAME in "${DATASETS[@]}"; do
    GOLD_ANSWER_FILE_PATH="/map-vepfs/dehua/code/visual-memory/answers/groundtruth/${DATASET_NAME}_answers.jsonl"
    DATA_PROPORTION=1

    # Loop through each k value
    for checkpoint_iter in "${checkpoint_iters[@]}"; do
        SAVE_FILENAME="answers/${DATASET_NAME}/qwen2-vl-7b_food172_dinov2_ns_${checkpoint_iter}.jsonl"
        QUESTION_FILE="questions/multi_image/${DATASET_NAME}/test_${K}_softmax.jsonl"
        MODEL_PATH="/map-vepfs/dehua/model/checkpoints/qwen2vl/food172_dinov2_ns/checkpoint-${checkpoint_iter}"
        echo "Using dataset: $DATASET_NAME"
        echo "Model path: $MODEL_PATH"
        echo "Gold answer file path: $GOLD_ANSWER_FILE_PATH"
        echo "Label file: $LABEL_FILE"
        echo "Save filename: $SAVE_FILENAME"
        echo "Question file: $QUESTION_FILE"
        echo "Data proportion: $DATA_PROPORTION"

        # 运行 Python 脚本
        python code/vlm/qwen2vl_vllm_text.py \
            --model_path "$MODEL_PATH" \
            --save_filename "$SAVE_FILENAME" \
            --gold_answer_file_path "$GOLD_ANSWER_FILE_PATH" \
            --label_file "$LABEL_FILE" \
            --question_file "$QUESTION_FILE" \
            --data_proportion "$DATA_PROPORTION" \
            --clip_file "$CLIP_FILE" \
            --mode $MODE

        # 打印完成消息
        echo "VLM inference for $DATASET_NAME with checkpoint_iter=$checkpoint_iter completed. Results saved to $SAVE_FILENAME"
    done
done



"""
wandb login f3b76ea66a38b2a211dc706fa95b02c761994b73
gpu=1
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/vlm/run_qwen2vl_dinov2.sh > gpu_$gpu.log &
"""