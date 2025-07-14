#!/bin/bash
export PYTHONPATH=/home/madehua/code/visual-memory
export ROOT_DIR=/home/madehua/code/visual-memory

# Define the paths to be used as arguments
#MODEL_PATH='/mnt/madehua/model/checkpoints/llava1.5-7b-retrieval-no-similarity-food101/checkpoint-533'
MODEL_PATH='/mnt/madehua/model/Qwen/Qwen2-VL-7B-Instruct'
SAVE_FILENAME='answers/food172/llava1.5-7b-similarity-retrieval_5.jsonl'
GOLD_ANSWER_FILE_PATH="/mnt/madehua/fooddata/json_file/172_answers.jsonl"
QUESTION_FILE='questions/food172/172_test_retrieval_category_and_similarity.jsonl'
TEMPLATE_NAME='qwen2_vl'


DATA_PROPORTION=0.1  # Use 1% of the data
LABEL_FILE='/mnt/madehua/fooddata/VireoFood172/SplitAndIngreLabel/FoodList.txt'
TAMPLATE_FILE='template.json'
# Run the Python script with the specified arguments
python code/vlm/vlm_inference.py \
    --model_path "$MODEL_PATH" \
    --save_filename "$SAVE_FILENAME" \
    --gold_answer_file_path "$GOLD_ANSWER_FILE_PATH" \
    --label_file "$LABEL_FILE" \
    --question_file "$QUESTION_FILE" \
    --data_proportion "$DATA_PROPORTION" \
    --template_file "$TAMPLATE_FILE" \
    --template_name "$TEMPLATE_NAME" \

"""
gpu_id=1
CUDA_VISIBLE_DEVICES=$gpu_id nohup sh scripts/run_vlm_qwen2vl.sh > output/gpt_$gpu_id.log 2>&1 &
"""