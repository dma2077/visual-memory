
import json
from tqdm import tqdm
file_name = '/map-vepfs/dehua/code/visual-memory/questions/veg200/dinov2_large_train_5_softmax_old.json'
groundtruth_file = "/map-vepfs/dehua/code/visual-memory/answers/groundtruth/veg200_train_groundtruth.jsonl"
food2k_label_file = '/map-vepfs/dehua/data/data/Food2k_complete_jpg/food2k_label2name_en.txt'
# food172_label_file = '/map-vepfs/dehua/data/data/VireoFood172/SplitAndIngreLabel/FoodList.txt'

def get_food2k_label(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        category_dict = {}
        for data in file.readlines():
            category_id, category = data.split('--')[0], data.split('--')[1].strip()
            category_id = str(int(category_id)) 
            category_dict[category_id] = category
    return category_dict

def get_food172_label(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        category_dict = {}
        for idx, data in enumerate(file.readlines()):
            category = data.strip()
            category_id = str(int(idx + 1)) 
            category_dict[category_id] = category
    return category_dict


# category_dict = get_food2k_label(food2k_label_file)
# category_dict = get_food172_label(food172_label_file)
datas = []
# Open and load the JSON file
with open(file_name, 'r', encoding='utf-8') as file:
    data = json.load(file)
    for idx, d in enumerate(tqdm(data, total=len(data))):
        image_path = d["images"][0]  # Ensure 'images' key exists in d
        groundtruth = image_path.split('/')[-2].replace('_', " ")

        # Handle cases where groundtruth_id is not in category_dict
        # groundtruth = category_dict.get(groundtruth_id, "unknown")

        # Construct new data entry
        new_data = {
            "question_id": idx,
            "image": image_path,
            "categories": ["" for _ in range(5)],  # Fixed list comprehension
            "text": "",
            "category": 'default',
            "groundtruth": groundtruth
        }

        # Append new_data to datas
        datas.append(new_data)
    
with open(groundtruth_file, 'w', encoding='utf-8') as file:
    for data in datas:
        data = json.dumps(data)
        file.write(data)
        file.write("\n")