#!/usr/bin/env bash

# 根目录，请根据实际路径调整
BASE_ROOT="/mmu_mllm_hdd_2/madehua/model/CKPT/verl/DAPO"
mkdir -p logs

# 数据集和检查点列表
DATASETS=(food101 food172 foodx251)
STEPS=(50 100 150 200 250)

# 构建所有要评估的 model_path 和对应的 ds 名称
model_paths=()
dss=()
for ds in "${DATASETS[@]}"; do
  for step in "${STEPS[@]}"; do
    model_paths+=("${BASE_ROOT}/Qwen2.5-VL-${ds}_cold_start-${ds}-0702/global_step_${step}/qwen2_5_vl_7b_${step}")
    dss+=("${ds}")
  done
done

# GPU 分配
# 组 1 用 GPU 0-7（共 8 张）
gpu_group1=(0 1 2 3 4 5 6 7)
# 组 2 用 GPU 0-6（共 7 张），你也可以换成 1-7 / 2-8 等
gpu_group2=(0 1 2 3 4 5 6)

# 记录组 1 的 PID
PIDS_GROUP1=()

# echo "===== Launching Group 1 (8 jobs on GPUs ${gpu_group1[*]}) ====="
# for idx in $(seq 0 7); do
#   model_path="${model_paths[$idx]}"
#   ds="${dss[$idx]}"
#   gpu_id="${gpu_group1[$idx]}"
#   model_name=$(basename "$model_path")
#   log_file="logs/${model_name}_${ds}.out"

#   echo "Job $((idx+1)): ${model_name}, dataset=${ds}, GPU=${gpu_id}"
#   nohup bash run_vllm.sh \
#     "$model_path" "None" "att" "$ds" 5 "$gpu_id" \
#     > "$log_file" 2>&1 &
#   PIDS_GROUP1+=($!)
# done

# echo "Waiting for Group 1 jobs to finish..."
# for pid in "${PIDS_GROUP1[@]}"; do
#   wait "$pid"
# done

echo "===== Group 1 completed. Launching Group 2 (7 jobs on GPUs ${gpu_group2[*]}) ====="
for j in $(seq 8 14); do
  idx=$((j-8))  # 0..6
  model_path="${model_paths[$j]}"
  ds="${dss[$j]}"
  gpu_id="${gpu_group2[$idx]}"
  model_name=$(basename "$model_path")
  log_file="logs/${model_name}_${ds}.out"

  echo "Job $((j+1)): ${model_name}, dataset=${ds}, GPU=${gpu_id}"
  nohup bash run_vllm.sh \
    "$model_path" "None" "att" "$ds" 5 "$gpu_id" \
    > "$log_file" 2>&1 &
done

echo "All 15 evaluation jobs have been dispatched."
