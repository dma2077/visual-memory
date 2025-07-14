#!/bin/bash

# Set the model path for siglip
model_path=/map-vepfs/models/google/siglip-so400m-patch14-384

# Set the file path for the test data
file_path=/map-vepfs/dehua/data/data/VireoFood172/SplitAndIngreLabel/TR.txt

# Set the root directory for the images
root_dir=/map-vepfs/dehua/data/data/VireoFood172/ready_chinese_food

# Set the batch size
batch_size=32

# Set the number of top predictions to retrieve
k=5

# Define the template for zero-shot classification
template="a photo of a {}"  # Use double quotes to handle spaces

# Set the output file path for the results
output_file=/map-vepfs/dehua/code/visual-memory/answers/food172/food172_train_results_softmax.jsonl

# Set the label file path
label_file=/map-vepfs/dehua/data/data/VireoFood172/SplitAndIngreLabel/FoodList.txt

# Run the zero-shot classification script
python code/encoder/siglip.py --model_path $model_path --file_path $file_path --root_dir $root_dir --batch_size $batch_size --k $k --output_file $output_file --template "$template" --label_file $label_file