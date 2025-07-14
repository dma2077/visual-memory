#!/bin/bash

export ROOT_DIR="/map-vepfs/dehua/code/visual-memory"  # 设置根目录路径
export PYTHONPATH=/map-vepfs/dehua/code/visual-memory


source /map-vepfs/dehua/anaconda3/bin/activate
conda activate vllm
# 定义要处理的数据集名称列表
DATASETS=("food172")
checkpoints=("258" "516" "774" "1032")
# 设置其他参数

#MODEL_PATH='/map-vepfs/models/Qwen/Qwen2-VL-7B-Instruct-infer'
# MODEL_PATH='/map-vepfs/dehua/model/checkpoints/qwen2vl/food101_raw/checkpoint-1184'
# MODEL_PATH='/map-vepfs/dehua/model/checkpoints/qwen2vl/raw_mix'
FEW_SHOT="$1"         # 第二组参数

echo $MODEL_PATH
#MODEL_PATH='/map-vepfs/dehua/model/checkpoints/qwen2vl/food172_5/checkpoint-2068'
# Array of k values
K_VALUES=(5)
MODE='siglip'
K=5
k1=3

for DATASET_NAME in "${DATASETS[@]}"; do
    GOLD_ANSWER_FILE_PATH="/map-vepfs/dehua/code/visual-memory/answers/groundtruth/${DATASET_NAME}_answers.jsonl"
    DATA_PROPORTION=1
    for checkpoint_iter in "${checkpoints[@]}"; do
        if [ "$FEW_SHOT" == "None" ]; then
            MODEL_PATH="/map-vepfs/dehua/model/checkpoints/qwen2vl/food172_raw/checkpoint-${checkpoint_iter}"
            SAVE_FILENAME="answers/${DATASET_NAME}/qwen2vl_${DATASET_NAME}_${checkpoint_iter}_${DATASET_NAME}.jsonl"
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
        python code/vlm/qwen2vl_vllm_text_raw.py \
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
done



: '

wandb login f3b76ea66a38b2a211dc706fa95b02c761994b73


fewshot_number=None
gpu=1
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/vlm/run_qwen2vl_raw_food172.sh "$fewshot_number" > log/qwen2vl_raw_food172_None.log &

fewshot_number=4
gpu=1
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/vlm/run_qwen2vl_raw.sh "$fewshot_number" > log/qwen2vl_raw_4.log &

fewshot_number=8
gpu=2
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/vlm/run_qwen2vl_mix_raw.sh "$fewshot_number" > log/qwen2vl_raw_8.log &
'