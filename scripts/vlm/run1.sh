#!/bin/bash

model_path="/map-vepfs/huggingface/models/Qwen2.5-VL-72B-Instruct-infer"
few_shot="None"
retrieval_method="dinov2"
k1=5
model_name=$(basename $model_path)

# nohup bash run_attribute.sh $model_path $few_shot $retrieval_method "food2k" $k1 "0,1,2,3" > logs/${model_name}_food2k_attribute.out 2>&1 &

nohup bash run_attribute.sh $model_path $few_shot $retrieval_method "veg200" $k1 "0,1,2,3" > logs/${model_name}_veg200_attribute.out 2>&1 &

nohup bash run_attribute.sh $model_path $few_shot $retrieval_method "foodx251" $k1 "4,5,6,7" > logs/${model_name}_foodx251_attribute.out 2>&1 &

# # First two jobs run immediately
# nohup bash run_attribute.sh $model_path $few_shot $retrieval_method "food101" $k1 "0,1,2,3" > logs/${model_name}_food101_attribute.out 2>&1 &
# PID1=$!  # Save the process ID

# nohup bash run_attribute.sh $model_path $few_shot $retrieval_method "food172" $k1 "4,5,6,7" > logs/${model_name}_food172_attribute.out 2>&1 &
# PID2=$!  # Save the process ID

# # Function to wait for a process to finish before launching a new one
# wait_for_free_gpus() {
#     while true; do
#         if ! ps -p $1 > /dev/null; then  # Check if process with PID $1 has finished
#             echo "Process $1 has completed. Assigning its GPUs to the next job."
#             break
#         fi
#         sleep 30  # Check every 30 seconds
#     done
# }

# # # Run jobs sequentially when previous jobs finish
# wait_for_free_gpus $PID1
# nohup bash run_attribute.sh $model_path $few_shot $retrieval_method "veg200" $k1 "0,1,2,3" > logs/${model_name}_veg200_attribute.out 2>&1 &
# PID3=$!

# wait_for_free_gpus $PID2
# nohup bash run_attribute.sh $model_path $few_shot $retrieval_method "fru92" $k1 "4,5,6,7" > logs/${model_name}_fru92_attribute.out 2>&1 &
# PID4=$!

# wait_for_free_gpus $PID3
# nohup bash run_attribute.sh $model_path $few_shot $retrieval_method "food2k" $k1 "0,1,2,3" > logs/${model_name}_food2k_attribute.out 2>&1 &
# PID5=$!

# wait_for_free_gpus $PID4
# nohup bash run_attribute.sh $model_path $few_shot $retrieval_method "foodx251" $k1 "4,5,6,7" > logs/${model_name}_foodx251_attribute.out 2>&1 &
# PID6=$!

# wait_for_free_gpus $PID6