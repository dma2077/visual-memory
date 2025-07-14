#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
model_path=/mnt/madehua/model/clip-vit-large-patch14-336
file_path=/mnt/madehua/fooddata/VireoFood172/SplitAndIngreLabel/TE.txt
root_dir=/mnt/madehua/fooddata/VireoFood172/ready_chinese_food
batch_size=16
k=5
template="a photo of a {}"  # Use double quotes to handle spaces
output_file=/mnt/madehua/fooddata/food_recognition/clip_retrieval/food172_results.json
label_file=/mnt/madehua/fooddata/VireoFood172/SplitAndIngreLabel/FoodList.txt

python code/encoder/clip.py --model_path $model_path --file_path $file_path --root_dir $root_dir --batch_size $batch_size --k $k --output_file $output_file --template "$template" --label_file $label_file