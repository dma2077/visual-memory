#!/usr/bin/env bash

# ------------------------
# 1) 准备变量
# ------------------------
CKPT_DIR="/mmu_mllm_hdd_2/madehua/model/CKPT/food_model/Qwen2.5-VL-food172_cold_sft_v1"
LOG_DIR="logs"
BASE_PATH="/llm_reco/dehua/code/visual-memory"
METRIC_PATH="$BASE_PATH/code/metric"
NUM_GPUS=8    # GPU 索引 0..7

mkdir -p "${LOG_DIR}"

# ------------------------
# 2) 并行启动所有 ckpt 的评估，并记录 PID
# ------------------------
declare -a PIDS=()
idx=0

for model_path in "${CKPT_DIR}"/checkpoint-* "${CKPT_DIR}"/ckpt-*; do
  [ -e "${model_path}" ] || continue

  model_name=$(basename "${model_path}")
  gpu_id=$(( idx % NUM_GPUS ))
  idx=$(( idx + 1 ))

  echo "[`date +'%H:%M:%S'`] Launching ${model_name} on GPU ${gpu_id}..."
  CUDA_VISIBLE_DEVICES="${gpu_id}" nohup bash run_vllm.sh \
    "${model_path}" "None" "att" "food172" 5 "${gpu_id}" \
    > "${LOG_DIR}/${model_name}_food172.out" 2>&1 &

  PIDS+=("$!")   # 将刚才后台任务的 PID 存入数组
done

# ------------------------
# 3) 按 PID 等待所有后台任务完成
# ------------------------
echo "[`date +'%H:%M:%S'`] Waiting for all run_vllm.sh jobs to finish..."
for pid in "${PIDS[@]}"; do
  if kill -0 "$pid" 2>/dev/null; then
    wait "$pid"
    echo "[`date +'%H:%M:%S'`] Job PID $pid completed."
  else
    echo "[`date +'%H:%M:%S'`] Job PID $pid already exited or not found."
  fi
done
echo "[`date +'%H:%M:%S'`] All run_vllm.sh jobs completed."

# ------------------------
# 4) 评估新生成的 JSONL
# ------------------------
echo "[`date +'%H:%M:%S'`] Evaluating newly generated JSONL files (last 20 minutes)..."
find "$BASE_PATH/answers/food172" -maxdepth 1 -type f -name "*.jsonl" -mmin -20 | while read -r file; do
  fname=$(basename "$file")
  echo "[`date +'%H:%M:%S'`]  → Evaluating ${fname}"
  python "$METRIC_PATH/get_172_acc.py" "$file"
done

echo "[`date +'%H:%M:%S'`] All evaluations done."
