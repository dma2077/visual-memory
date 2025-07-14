cd /map-vepfs/dehua/code/visual-memory

fewshot_number=None
MIXDATASET=False
checkpointname="food2k_3_fewshot_and_all_nocategory/checkpoint-1000"
gpu=0
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/vlm/run_qwen2vl_mix_rank.sh "$fewshot_number" "$MIXDATASET" "$checkpointname" > log/qwen2vl_${fewshot_number}_${MIXDATASET}_ac_1000.log 2>&1 &

fewshot_number=None
MIXDATASET=False
checkpointname="food2k_3_fewshot_and_all_nocategory/checkpoint-2000"
gpu=1
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/vlm/run_qwen2vl_mix_rank.sh "$fewshot_number" "$MIXDATASET" "$checkpointname" > log/qwen2vl_${fewshot_number}_${MIXDATASET}_ac_2000.log 2>&1 &

fewshot_number=None
MIXDATASET=False
checkpointname="food2k_3_fewshot_and_all_nocategory/checkpoint-3000"
gpu=2
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/vlm/run_qwen2vl_mix_rank.sh "$fewshot_number" "$MIXDATASET" "$checkpointname" > log/qwen2vl_${fewshot_number}_${MIXDATASET}_ac_3000.log 2>&1 &

fewshot_number=None
MIXDATASET=False
checkpointname="food2k_3_fewshot_and_all_nocategory/checkpoint-4000"
gpu=3
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/vlm/run_qwen2vl_mix_rank.sh "$fewshot_number" "$MIXDATASET" "$checkpointname" > log/qwen2vl_${fewshot_number}_${MIXDATASET}_ac_4000.log 2>&1 &



fewshot_number=4
MIXDATASET=False
checkpointname="food2k_3_fewshot_and_all_nocategory/checkpoint-1000"
gpu=4
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/vlm/run_qwen2vl_mix_rank.sh "$fewshot_number" "$MIXDATASET" "$checkpointname" > log/qwen2vl_${fewshot_number}_${MIXDATASET}_ac_1000.log 2>&1 &

fewshot_number=4
MIXDATASET=False
checkpointname="food2k_3_fewshot_and_all_nocategory/checkpoint-2000"
gpu=5
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/vlm/run_qwen2vl_mix_rank.sh "$fewshot_number" "$MIXDATASET" "$checkpointname" > log/qwen2vl_${fewshot_number}_${MIXDATASET}_ac_2000.log 2>&1 &

fewshot_number=4
MIXDATASET=False
checkpointname="food2k_3_fewshot_and_all_nocategory/checkpoint-3000"
gpu=6
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/vlm/run_qwen2vl_mix_rank.sh "$fewshot_number" "$MIXDATASET" "$checkpointname" > log/qwen2vl_${fewshot_number}_${MIXDATASET}_ac_3000.log 2>&1 &

fewshot_number=4
MIXDATASET=False
checkpointname="food2k_3_fewshot_and_all_nocategory/checkpoint-4000"
gpu=7
CUDA_VISIBLE_DEVICES=$gpu nohup bash scripts/vlm/run_qwen2vl_mix_rank.sh "$fewshot_number" "$MIXDATASET" "$checkpointname" > log/qwen2vl_${fewshot_number}_${MIXDATASET}_ac_4000.log 2>&1

sleep 600