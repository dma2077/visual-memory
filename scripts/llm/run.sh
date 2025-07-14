#!/bin/bash

fewshot_number=None
gpu=0
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/llm/run_llama3_mix_old.sh $(( (gpu + 1) * 1000 )) "$fewshot_number" > log/llama3_food2k_3_rank_old_$gpu.log 2>&1 &
gpu=1
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/llm/run_llama3_mix_old.sh $(( (gpu + 1) * 1000 )) "$fewshot_number" > log/llama3_food2k_3_rank_old_$gpu.log 2>&1 &
gpu=2
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/llm/run_llama3_mix_old.sh $(( (gpu + 1) * 1000 )) "$fewshot_number" > log/llama3_food2k_3_rank_old_$gpu.log 2>&1 &
gpu=3
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/llm/run_llama3_mix_old.sh $(( (gpu + 1) * 1000 )) "$fewshot_number" > log/llama3_food2k_3_rank_old_$gpu.log 2>&1 &

fewshot_number=4
gpu=4
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/llm/run_llama3_mix_old.sh $(( (gpu - 3) * 1000 )) "$fewshot_number" > log/llama3_food2k_3_rank_old_$gpu.log 2>&1 &
gpu=5
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/llm/run_llama3_mix_old.sh $(( (gpu - 3) * 1000 )) "$fewshot_number" > log/llama3_food2k_3_rank_old_$gpu.log 2>&1 &
gpu=6
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/llm/run_llama3_mix_old.sh $(( (gpu - 3) * 1000 )) "$fewshot_number" > log/llama3_food2k_3_rank_old_$gpu.log 2>&1 &
gpu=7
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/llm/run_llama3_mix_old.sh $(( (gpu - 3) * 1000 )) "$fewshot_number" > log/llama3_food2k_3_rank_old_$gpu.log 2>&1
sleep 600