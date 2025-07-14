#!/bin/bash

# Set the model path for siglip
model_path=/mnt/madehua/model/google/siglip-so400m-patch14-384

# Set the file path for the test data
file_path=/mnt/madehua/fooddata/vegfru-dataset/veg200_lists/veg_test.txt

# Set the root directory for the images
root_dir=/mnt/madehua/fooddata/vegfru-dataset/veg200_images

# Set the batch size
batch_size=32

# Set the number of top predictions to retrieve
k=5

# Define the template for zero-shot classification
template="a photo of a {}"  # Use double quotes to handle spaces

# Set the output file path for the results
output_file=/mnt/madehua/fooddata/food_recognition/slip_retrieval/veg200_results_softmax.jsonl

# Set the label file path
label_file=/mnt/madehua/fooddata/vegfru-dataset/veg200_lists/veg_subclasses.txt

# Run the zero-shot classification script
python code/encoder/siglip.py --model_path $model_path --file_path $file_path --root_dir $root_dir --batch_size $batch_size --k $k --output_file $output_file --template "$template" --label_file $label_file