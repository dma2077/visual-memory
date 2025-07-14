#!/bin/bash
# 运行4-shot数据处理脚本

echo "=== 开始处理4-shot数据 ==="
cd /Users/dehua/code/visual-memory
python code/data_pre_process/cold_sft_data_4shot.py
echo "=== 4-shot数据处理完成 ===" 