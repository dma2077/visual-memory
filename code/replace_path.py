import json
from utils import *
import copy
import os

# root_dir = '/map-vepfs/dehua/model/food_embeddings/six_dataset_trainset/63659_student'

# # Get all JSON files in the root_dir
# json_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.json')]

# for filename in json_files:
#     results = load_json(filename)
#     new_results = copy.copy(results)

#     image_path = results["image_path"]
#     new_image_paths = []
#     for path in image_path:
#         new_path = path.replace('/mnt/madehua/fooddata', '/map-vepfs/dehua/data/data')
#         new_image_paths.append(new_path)

#     new_results["image_path"] = new_image_paths

#     save_json(filename, new_results)


with open('/map-vepfs/dehua/data/data/FoodX-251/annot/class_list.txt', 'r') as file:
    lines = file.readlines()

for line in lines:
    print(line.strip().split(' ')[1])