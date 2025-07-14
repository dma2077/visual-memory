#!/bin/bash
source /map-vepfs/dehua/anaconda3/bin/activate
conda activate vllm
# Set the model path for siglip
model_path=/map-vepfs/models/openai/clip-vit-large-patch14-336

# Set the file path for the test data
file_path=/map-vepfs/dehua/data/data/Food2k_complete_jpg/test.txt

# Set the root directory for the images
root_dir=/map-vepfs/dehua/data/data/Food2k_complete_jpg

# Set the batch size
batch_size=2048

# Set the number of top predictions to retrieve
k=5

# Define the template for zero-shot classification
template="a photo of a {}"  # Use double quotes to handle spaces

# Set the output file path for the results
output_file=/map-vepfs/dehua/code/visual-memory/answers/food2k/food2k_clip_results_softmax.jsonl

# Set the label file path
label_file=/map-vepfs/dehua/data/data/Food2k_complete_jpg/food2k_label2name_en.txt
dataset=food2k


# Run the zero-shot classification script
python code/encoder/siglip.py --model_path $model_path --file_path $file_path --root_dir $root_dir --batch_size $batch_size --k $k --output_file $output_file --template "$template" --label_file $label_file --dataset $dataset


# file_path=/map-vepfs/dehua/data/data/Food2k_complete_jpg/train.txt

# output_file=/map-vepfs/dehua/code/visual-memory/answers/food2k/food2k_clip_train_results_softmax.jsonl

# python code/encoder/siglip.py --model_path $model_path --file_path $file_path --root_dir $root_dir --batch_size $batch_size --k $k --output_file $output_file --template "$template" --label_file $label_file --dataset $dataset
