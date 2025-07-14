
model_path="/map-vepfs/huggingface/models/Qwen2.5-VL-7B-Instruct-infer"
model_path="/map-vepfs/huggingface/models/InternVL2_5-8B"
few_shot="None"
inference_method="cot"
k1=5
model_name=$(basename $model_path)
# bash run_vllm.sh $model_path $few_shot $inference_method "food101" $k1 0
pids=()

model_path="/mmu_mllm_hdd_2/madehua/model/InternVL3-8B"
model_name=$(basename $model_path)
nohup bash run_vllm.sh $model_path "None" "mix" "food101" 5 3 > logs/${model_name}_food101.out 2>&1 & 
pids+=($!)

# model_path="/map-vepfs/dehua/model/checkpoints/qwen2vl/qwen2vl_food172_attribute"
model_name=$(basename $model_path)
nohup bash run_vllm.sh $model_path "None" "mix" "food172" 5 4 > logs/${model_name}_food172.out 2>&1 &
pids+=($!)

# model_path="/map-vepfs/dehua/model/checkpoints/qwen2vl/qwen2vl_fru92_attribute"
model_name=$(basename $model_path)
nohup bash run_vllm.sh $model_path "None" "mix" "fru92" 5 5 > logs/${model_name}_fur92.out 2>&1 &
pids+=($!)

# model_path="/map-vepfs/dehua/model/checkpoints/qwen2vl/qwen2vl_veg200_attribute"
model_name=$(basename $model_path)
nohup bash run_vllm.sh $model_path "None" "mix" "veg200" 5 6 > logs/${model_name}_veg200.out 2>&1 &
pids+=($!)

# model_path="/map-vepfs/dehua/model/checkpoints/qwen2vl/qwen2vl_foodx251_attribute"
model_name=$(basename $model_path)
nohup bash run_vllm.sh $model_path "None" "mix" "foodx251" 5 2 > logs/${model_name}_foodx251.out 2>&1 &

model_name=$(basename $model_path)
nohup bash run_vllm.sh $model_path "None" "mix" "food2k" 5 1 > logs/${model_name}_food2k.out 2>&1 &
pids+=($!)

# # 等待所有后台进程完成
# for pid in "${pids[@]}"; do
#     wait $pid
# done
