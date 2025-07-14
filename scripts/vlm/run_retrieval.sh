#!/bin/bash

# 设置变量
export CUDA_VISIBLE_DEVICES=6  # 只使用第0号GPU

dataset_name="food172"
model_name="dinov2_large"
model_path="/mnt/madehua/model/facebook/dinov2-large"
output_dir="/mnt/madehua/food_embeddings"
aggregate_method="GeM"
embedding_dir="/mnt/madehua/food_embeddings"
train_index_file="${embedding_dir}/${dataset_name}_train_${model_name}_cos_${aggregate_method}.bin"
test_index_file="${embedding_dir}/${dataset_name}_test_${model_name}_cos_${aggregate_method}.bin"


python get_embeddings_cos.py --dataset_name $dataset_name --model_name $model_name --model_path $model_path --output_dir $output_dir --aggregate_method $aggregate_method
