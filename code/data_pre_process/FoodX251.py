import json
import random
import os
import pandas as pd

from utils import *


import shutil

def move_file(source, destination):
    try:
        shutil.move(source, destination)
        print(f"文件已成功移动到 {destination}")
    except Exception as e:
        print(f"移动文件时出错: {e}")
        
        
def load_label(file):
    label_dict = {}
    with open(file , 'r') as f:
        for line in f.readlines():
            line = line.strip()
            line = line.split(' ')
            label_dict[line[0]] = line[1]
    return label_dict

def create_directories(label_dict, base_path):
    for label in set(label_dict.values()):  # 使用 set 去重
        dir_path = os.path.join(base_path, label)
        os.makedirs(dir_path, exist_ok=True)  # 如果目录不存在则创建
        
lable_path = '/media/fast_data/food_recognition_dataset/FoodX-251/label.txt'
base_image_path = '/media/fast_data/food_recognition_dataset/FoodX-251/images'

label = load_label(lable_path)
# create_directories(label, base_image_path)

train_path_new = '/media/fast_data/food_recognition_dataset/FoodX-251/annot/train.txt'
val_path_new  = '/media/fast_data/food_recognition_dataset/FoodX-251/annot/val.txt'

with open(val_path_new, 'r') as f:
    train_list = f.readlines()
    for train_data in train_list:
        train_data = train_data.strip()
        train_data = train_data.split(',')
        relative_path = train_data[0]
        label_id = train_data[1]
        source_path = '/media/fast_data/food_recognition_dataset/FoodX-251/train_set/' + relative_path
        target_path = '/media/fast_data/food_recognition_dataset/FoodX-251/images/' + label[label_id] + '/' + relative_path
        move_file(source_path, target_path)


train_path = '/media/fast_data/food_recognition_dataset/FoodX-251/annot/train_info.csv'
test_path = '/media/fast_data/food_recognition_dataset/FoodX-251/annot/test_info.csv'
val_path = '/media/fast_data/food_recognition_dataset/FoodX-251/annot/val_info.csv'


train_path_new = '/media/fast_data/food_recognition_dataset/FoodX-251/annot/train.txt'
val_path_new  = '/media/fast_data/food_recognition_dataset/FoodX-251/annot/val.txt'

train_path_new1 = '/media/fast_data/food_recognition_dataset/FoodX-251/annot/train1.txt'
val_path_new1  = '/media/fast_data/food_recognition_dataset/FoodX-251/annot/val1.txt'

# relative_paths = []
# with open(val_path_new, 'r') as f:
#     train_list = f.readlines()
#     for data in train_list:
#         data = data.strip()
#         data = data.split(',')
#         relative_path = data[0]
#         label_id = data[1]
#         relative_path = label[label_id] + '/' + relative_path
#         print(relative_path)
#         relative_path = str(relative_path)
#         relative_paths.append(relative_path)

# with open(val_path_new, 'w') as f:
#     for relative_path in relative_paths:
#         f.write(relative_path)
#         f.write('\n')

# df = pd.read_csv(val_path)



# df.to_csv(val_path_new, index=False)


