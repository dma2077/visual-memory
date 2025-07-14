# #!/bin/bash

# 设置基础路径
BASE_PATH="/llm_reco/dehua/code/visual-memory"
METRIC_PATH="$BASE_PATH/code/metric"

# 运行 Food101 评估
# echo "Evaluating Food101..."
# python $METRIC_PATH/get_101_acc.py $BASE_PATH/answers/food101/food101_1e-5_3r_step_50_food101_None_att_k5.jsonl

# # 运行 Food101 评估
# echo "Evaluating Food101..."
# python $METRIC_PATH/get_101_acc.py $BASE_PATH/answers/food101/food101_1e-5_3r_step_150_food101_None_att_k5.jsonl

# # 运行 Food101 评估
# echo "Evaluating Food101..."
# python $METRIC_PATH/get_101_acc.py $BASE_PATH/answers/food101/food101_1e-5_3r_step_150_food101_None_att_k5.jsonl

# 运行 Food101 评估
echo "Evaluating Food101..."
python $METRIC_PATH/get_101_acc.py $BASE_PATH/answers/food101/Qwen2.5-VL-7B-Instruct_food101_None_mix_k5.jsonl

echo "Evaluating Food172..."
python $METRIC_PATH/get_172_acc.py $BASE_PATH/answers/food172/Qwen2.5-VL-7B-Instruct_food172_None_mix_k5.jsonl

# # 运行 FoodX-251 评估
# echo "Evaluating FoodX-251..."
# python $METRIC_PATH/get_251_acc.py $BASE_PATH/answers/foodx251/Qwen2.5-VL-foodx251_cold_sft_nofreeze_foodx251_None_att_k5.jsonl

# # 运行 Food172 评估
# echo "Evaluating Food172..."
# python $METRIC_PATH/get_172_acc.py $BASE_PATH/answers/food172/checkpoint-5906_food172_None_att_k5.jsonl



# # 运行 Fru92 评估
# echo "Evaluating Fru92..."
# python $METRIC_PATH/get_92_acc.py $BASE_PATH/answers/fru92/Qwen2.5-VL-fru92_attribute_0613_fru92_None_att_k5.jsonl

# # 运行 Veg200 评估
# echo "Evaluating Veg200..."
# python $METRIC_PATH/get_200_acc.py $BASE_PATH/answers/veg200/Qwen2.5-VL-veg200_attribute_0613_veg200_None_att_k5.jsonl
# #!/bin/bash

# # 设置基础路径
# BASE_PATH="/llm_reco/dehua/code/visual-memory"
# METRIC_PATH="$BASE_PATH/code/metric"

# # 要遍历的 checkpoint 列表
# CKPTS=(50 100 150 200 250)

# # 评估 Food101
# for ckpt in "${CKPTS[@]}"; do
#   echo "Evaluating Food101 — ckpt ${ckpt}..."
#   python "$METRIC_PATH/get_101_acc.py" \
#     "$BASE_PATH/answers/food101/qwen2_5_vl_7b_${ckpt}_food101_None_att_k5.jsonl"
# done

# # 评估 Food172
# for ckpt in "${CKPTS[@]}"; do
#   echo "Evaluating Food172 — ckpt ${ckpt}..."
#   python "$METRIC_PATH/get_172_acc.py" \
#     "$BASE_PATH/answers/food172/qwen2_5_vl_7b_${ckpt}_food172_None_att_k5.jsonl"
# done

# # 评估 FoodX-251
# for ckpt in "${CKPTS[@]}"; do
#   echo "Evaluating FoodX-251 — ckpt ${ckpt}..."
#   python "$METRIC_PATH/get_251_acc.py" \
#     "$BASE_PATH/answers/foodx251/qwen2_5_vl_7b_${ckpt}_foodx251_None_att_k5.jsonl"
# done