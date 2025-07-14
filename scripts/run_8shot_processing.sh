#!/bin/bash
# 运行8-shot数据处理脚本

echo "=== 开始处理8-shot数据 ==="
cd /Users/dehua/code/visual-memory
python code/data_pre_process/cold_sft_data_8shot.py
echo "=== 8-shot数据处理完成 ===" 