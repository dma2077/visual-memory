
model_path="/map-vepfs/huggingface/models/Qwen2.5-VL-7B-Instruct-infer"
# model_path="/map-vepfs/huggingface/models/InternVL2_5-8B"
few_shot="None"
inference_method="raw"
k1=5
model_name=$(basename $model_path)
# bash run_vllm.sh $model_path $few_shot $inference_method "food101" $k1 0
pids=()

model_path="/map-vepfs/dehua/model/checkpoints/qwen2vl/all_raw/checkpoint-9550"
model_name=$(basename $model_path)
nohup bash run_vllm.sh $model_path "None" "raw" "food101" 5 6 > logs/${model_name}_food101.out 2>&1 & 
pids+=($!)

model_name=$(basename $model_path)
nohup bash run_vllm.sh $model_path "None" "raw" "food172" 5 1 > logs/${model_name}_food172.out 2>&1 &
pids+=($!)

model_name=$(basename $model_path)
nohup bash run_vllm.sh $model_path "None" "raw" "fru92" 5 2 > logs/${model_name}_fur92.out 2>&1 &
pids+=($!)

model_name=$(basename $model_path)
nohup bash run_vllm.sh $model_path "None" "raw" "veg200" 5 7 > logs/${model_name}_veg200.out 2>&1 &
pids+=($!)

model_name=$(basename $model_path)
nohup bash run_vllm.sh $model_path "None" "raw" "foodx251" 5 4 > logs/${model_name}_foodx251.out 2>&1 &
pids+=($!)

model_name=$(basename $model_path)
nohup bash run_vllm.sh $model_path "None" "raw" "food2k" 5 5 > logs/${model_name}_food2k.out 2>&1 &
pids+=($!)

# 等待所有后台进程完成
for pid in "${pids[@]}"; do
    wait $pid
done

# 所有进程结束后
echo "All tasks finished. Proceeding with next steps."

#nohup bash run_vllm.sh $model_path $few_shot $retrieval_method "food2k" $k1 2 > logs/${model_name}_food2k.out 2>&1 &
# nohup bash run_vllm.sh $model_path $few_shot $retrieval_method "fru92" $k1 3 > logs/${model_name}_fur92.out 2>&1 &
# nohup bash run_vllm.sh $model_path $few_shot $retrieval_method "veg200" $k1 4 > logs/${model_name}_veg200.out 2>&1 &
# nohup bash run_vllm.sh $model_path $few_shot $retrieval_method "foodx251" $k1 5 > logs/${model_name}_foodx251.out 2>&1 &

# # 等待所有后台进程完成
wait

# 所有后台进程完成后，运行后续内容
# 在这里添加后续的命令
echo "All tasks finished. Proceeding with next steps."