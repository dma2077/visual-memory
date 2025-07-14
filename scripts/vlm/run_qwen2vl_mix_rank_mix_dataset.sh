#!/bin/bash

export ROOT_DIR="/map-vepfs/dehua/code/visual-memory"  # 设置根目录路径
export PYTHONPATH=/map-vepfs/dehua/code/visual-memory


source /map-vepfs/dehua/anaconda3/bin/activate
conda activate vllm
# 定义要处理的数据集名称列表
DATASETS=("food101" "food172" "fru92" "veg200" "foodx251" "food2k")
# 设置其他参数

MODEL_PATH='/map-vepfs/models/Qwen/Qwen2-VL-7B-Instruct-infer'
checkpoint_iter="$1"  # 第一组参数
FEW_SHOT="$2"         # 第二组参数
MODEL_PATH="/map-vepfs/dehua/model/checkpoints/qwen2vl/food2k_3_image_rank/checkpoint-${checkpoint_iter}"
echo $MODEL_PATH
#MODEL_PATH='/map-vepfs/dehua/model/checkpoints/qwen2vl/food172_5/checkpoint-2068'
# Array of k values
K_VALUES=(5)
MODE='mix'
K=5
k1=3

for DATASET_NAME in "${DATASETS[@]}"; do
    GOLD_ANSWER_FILE_PATH="/map-vepfs/dehua/code/visual-memory/answers/groundtruth/${DATASET_NAME}_answers.jsonl"
    DATA_PROPORTION=1
    if [ "$FEW_SHOT" == "None" ]; then
        SAVE_FILENAME="answers/${DATASET_NAME}/qwen2vl_food2k_3_image_rank_checkpoint-${checkpoint_iter}_mixedtrain.jsonl"
        DINO_FILE="questions/${DATASET_NAME}/dinov2_large_test_${K}_mixedtrain.jsonl"
        SIGLIP_FILE="questions/${DATASET_NAME}/siglip_test_${K}_mixedtrain.jsonl"
    else
        SAVE_FILENAME="answers/${DATASET_NAME}/qwen2vl_food2k_3_image_rank_checkpoint-${checkpoint_iter}_fewshot${FEW_SHOT}_mixedtrain.jsonl"
        DINO_FILE="questions/${DATASET_NAME}/dinov2_large_test_${K}_mixedtrain_fewshot${FEW_SHOT}.jsonl"
        SIGLIP_FILE="questions/${DATASET_NAME}/siglip_test_${K}_mixedtrain_fewshot${FEW_SHOT}.jsonl"
    fi
    echo "Using dataset: $DATASET_NAME"
    echo "Model path: $MODEL_PATH"
    echo "Gold answer file path: $GOLD_ANSWER_FILE_PATH"
    echo "Label file: $LABEL_FILE"
    echo "Save filename: $SAVE_FILENAME"
    echo "Question file: $QUESTION_FILE"
    echo "Data proportion: $DATA_PROPORTION"

    # 运行 Python 脚本
    python code/vlm/qwen2vl_vllm_text_rank.py \
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
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/vlm/run_qwen2vl_mix_rank_mix_dataset.sh $(( (gpu + 1) * 1000 )) "$fewshot_number" > mixdataset_ac_$gpu.log &
gpu=1
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/vlm/run_qwen2vl_mix_rank_mix_dataset.sh $(( (gpu + 1) * 1000 )) "$fewshot_number" > mixdataset_ac_$gpu.log &
gpu=2
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/vlm/run_qwen2vl_mix_rank_mix_dataset.sh $(( (gpu + 1) * 1000 )) "$fewshot_number" > mixdataset_ac_$gpu.log &
gpu=3
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/vlm/run_qwen2vl_mix_rank_mix_dataset.sh $(( (gpu + 1) * 1000 )) "$fewshot_number" > mixdataset_ac_$gpu.log &

fewshot_number=4
gpu=4
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/vlm/run_qwen2vl_mix_rank_mix_dataset.sh $(( (gpu - 3) * 1000 )) "$fewshot_number" > mixdataset_ac_4_$gpu.log &
gpu=5
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/vlm/run_qwen2vl_mix_rank_mix_dataset.sh $(( (gpu - 3) * 1000 )) "$fewshot_number" > mixdataset_ac_4_$gpu.log &
gpu=6
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/vlm/run_qwen2vl_mix_rank_mix_dataset.sh $(( (gpu - 3) * 1000 )) "$fewshot_number" > mixdataset_ac_4_$gpu.log &
gpu=7
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/vlm/run_qwen2vl_mix_rank_mix_dataset.sh $(( (gpu - 3) * 1000 )) "$fewshot_number" > mixdataset_ac_4_$gpu.log &
'
