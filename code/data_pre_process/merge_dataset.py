# Define the paths to the input files and their corresponding root directories
datasets_train = [
    ("/mnt/madehua/fooddata/vegfru-dataset/veg200_lists/processed_veg_train.txt", "/mnt/madehua/fooddata/vegfru-dataset/veg200_images"),
    ("/mnt/madehua/fooddata/vegfru-dataset/fru92_lists/processed_fru_train.txt", "/mnt/madehua/fooddata/vegfru-dataset/fru92_images"),
    ("/mnt/madehua/fooddata/Food2k_complete/meta/train.txt", "/mnt/madehua/fooddata/Food2k_complete"),
    ("/mnt/madehua/fooddata/food-101/meta/train.txt", "/mnt/madehua/fooddata/food-101/images"),
    ("/mnt/madehua/fooddata/VireoFood172/SplitAndIngreLabel/TR.txt", "/mnt/madehua/fooddata/VireoFood172/ready_chinese_food"),
    ("/mnt/madehua/fooddata/FoodX-251/annot/train.txt", "/mnt/madehua/fooddata/FoodX-251/images")   
]

datasets_test = [
    ("/mnt/madehua/fooddata/vegfru-dataset/veg200_lists/processed_veg_test.txt", "/mnt/madehua/fooddata/vegfru-dataset/veg200_images"),
    ("/mnt/madehua/fooddata/vegfru-dataset/fru92_lists/processed_fru_test.txt", "/mnt/madehua/fooddata/vegfru-dataset/fru92_images"),
    ("/mnt/madehua/fooddata/FoodX-251/annot/val.txt", "/mnt/madehua/fooddata/FoodX-251/images"),
    ("/mnt/madehua/fooddata/Food2k_complete/meta/test.txt", "/mnt/madehua/fooddata/Food2k_complete"),
    ("/mnt/madehua/fooddata/food-101/meta/test.txt", "/mnt/madehua/fooddata/food-101/images"),
    ("/mnt/madehua/fooddata/VireoFood172/SplitAndIngreLabel/TE.txt", "/mnt/madehua/fooddata/VireoFood172/ready_chinese_food"),    
]

datasets = datasets_train + datasets_test
# Define the path to the output file
output_file = "/mnt/madehua/fooddata/six_dataset_full.txt"

# Open the output file in write mode
with open(output_file, 'w') as outfile:
    # Iterate over each dataset
    for file_path, root_dir in datasets:
        image_count = 0  # Initialize image count for each dataset
        try:
            # Open each input file in read mode
            with open(file_path, 'r') as infile:
                for line in infile:
                    relative_path = line.strip()
                    if relative_path[0] == '/':
                        relative_path = relative_path[1:]
                    absolute_path = f"{root_dir}/{relative_path}"
                    
                    # Check if the path ends with .jpg or .jpeg, if not, append .jpg
                    if not absolute_path.lower().endswith(('.jpg', '.jpeg')):
                        absolute_path += '.jpg'
                    
                    # Write the absolute path to the output file
                    outfile.write(absolute_path + '\n')
                    image_count += 1  # Increment image count
            print(f"Merged {file_path} with {image_count} images.")
        except FileNotFoundError:
            print(f"File {file_path} does not exist.")

print(f"All files have been merged into {output_file}")

import random

# Define the path to the file
file_path = '/mnt/madehua/fooddata/six_dataset_full.txt'

# Read all lines from the file
with open(file_path, 'r') as file:
    lines = file.readlines()

# Shuffle the lines
random.shuffle(lines)

# Write the shuffled lines back to the file
with open(file_path, 'w') as file:
    file.writelines(lines)

print(f"The file {file_path} has been shuffled.")