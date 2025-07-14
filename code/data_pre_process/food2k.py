import json
import random
import os

# def get_jpg_files(dir_path):
#     with open('/media/fast_data/Food2k_complete/all_images.txt', 'w') as f:
#         for root, dirs, files in os.walk(dir_path):
#             for file in files:
#                 if file.endswith('.jpg'):
#                     relative_path = '/' + root.split('/')[-1] + '/' + file
#                     print(relative_path)
#                     f.write(f'/{relative_path}\n')

# get_jpg_files('/media/fast_data/Food2k_complete')  # 替换为你的目录路径



def split_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    random.shuffle(lines)  # 打乱顺序
    split_index = int(len(lines) * 0.9)  # 90%为训练集

    train_set = lines[:split_index]
    test_set = lines[split_index:]

    with open('/media/fast_data/Food2k_complete/train.txt', 'w') as train_file, open('/media/fast_data/Food2k_complete/test.txt', 'w') as test_file:
        train_file.writelines(train_set)
        test_file.writelines(test_set)



# split_data('/media/fast_data/Food2k_complete/all_images.txt')



# file_path = '/media/fast_data/Food2k_complete/all_images.txt'
# with open(file_path, 'r') as f:
#     lines = f.readlines()

# set = [line.replace('//', '/') for line in lines]
# with open('/media/fast_data/Food2k_complete/all_images.txt', 'w') as file:
#     file.writelines(set)


# import os
# import shutil
# from tqdm import tqdm

# # Define the source file and destination directory
# source_file = '/mnt/madehua/fooddata/Food2k_complete/all_images.txt'
# destination_root = '/mnt/madehua/fooddata/Food2k_complete_jpg'

# # Ensure the destination directory exists
# os.makedirs(destination_root, exist_ok=True)

# # Read all lines from the source file
# with open(source_file, 'r') as file:
#     lines = file.readlines()

# # Process each line with a progress bar
# for line in tqdm(lines, desc="Copying files", unit="file"):
#     # Get the relative path and strip any leading/trailing whitespace
#     relative_path = line.strip()
    
#     # Construct the full source path
#     source_path ='/mnt/madehua/fooddata/Food2k_complete' + relative_path
    
#     # Construct the full destination path
#     destination_path = destination_root + relative_path
    
#     # Ensure the destination directory exists
#     os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    
#     # Copy the file
#     shutil.copy2(source_path, destination_path)

# print("All files have been copied.")
