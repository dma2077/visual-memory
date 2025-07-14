#!/bin/bash

model_path=/map-vepfs/models/google/siglip-so400m-patch14-384
file_path=/map-vepfs/dehua/data/data/food-101/meta/train.txt
root_dir=/map-vepfs/dehua/data/data/food-101/images
batch_size=2048
k=5
template="a photo of a {}"  # Use double quotes to handle spaces
output_file=/map-vepfs/dehua/code/visual-memory/answers/food101/food101_train_results_softmax.jsonl
label_file=/map-vepfs/dehua/data/data/food-101/meta/labels.txt

python code/encoder/siglip.py --model_path $model_path --file_path $file_path --root_dir $root_dir --batch_size $batch_size --k $k --output_file $output_file --template "$template" --label_file $label_file