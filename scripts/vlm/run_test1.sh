#!/bin/bash

model_path="/mmu_mllm_hdd_2/madehua/model/CKPT/food_model/Qwen2.5-VL-all_cold_sft_nosample"
# model_path="/llm_reco/dehua/code/Visual-RFT/outputs/Qwen2.5-VL-food101_raw_food101_all_shot_att"
# model_path="/llm_reco/dehua/model/food_model/Qwen2.5-VL-food_pretrain/checkpoint-19099"
# model_path="/mmu_mllm_hdd_2/madehua/model/CKPT/verl/DAPO/Qwen2.5-VL-food101_1e-5_3r/global_step_150/food101_1e-5_3r_step_150"
# model_name=$(basename $model_path)
# nohup bash run_vllm.sh $model_path "None" "att" "food101" 5 7 > logs/${model_name}_food101.out 2>&1 & 

# model_path="/mmu_mllm_hdd_2/madehua/model/CKPT/verl/DAPO/Qwen2.5-VL-food101_1e-5_3r/global_step_200/food101_1e-5_3r_step_200"
# model_name=$(basename $model_path)
# nohup bash run_vllm.sh $model_path "None" "att" "food101" 5 6 > logs/${model_name}_food101.out 2>&1 & 

model_path="/mmu_mllm_hdd_2/madehua/model/CKPT/food_model/Qwen2.5-VL-food2k_cold_sft_nofreeze"
model_name=$(basename $model_path)
nohup bash run_vllm.sh $model_path "None" "att" "food2k" 5 7 > logs/${model_name}_food2k.out 2>&1 & 
