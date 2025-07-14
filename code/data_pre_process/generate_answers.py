
import json

import os


"""
food2k
"""
# root_dir = "/mnt/madehua/fooddata/Food2k_complete"


# label_path = '/mnt/madehua/fooddata/Food2k_complete/food2k_label2name_en.txt'
# with open(label_path, 'r') as f:
#     lines = f.readlines()
#     labels = [line.strip().split('--')[1] for line in lines]
    
# with open("/mnt/madehua/fooddata/Food2k_complete/test.txt", "r") as f:
#     lines = f.readlines()
    
# conversations = []
# for idx, line in enumerate(lines):
#     relative_path = line.strip()
#     category_id = relative_path.split("/")[1]

#     conversation = {
#     "question_id": idx, 
#     "image": root_dir + relative_path, 
#     "text": labels[int(category_id)], 
#     "category": "default"
#     }
#     conversations.append(conversation)

# with open("/mnt/madehua/fooddata/json_file/food2k_answers.jsonl", "w") as f:
#     for conversation in conversations:
#         f.write(json.dumps(conversation) + "\n")
        
        

"""
veg200 and fru92
"""

# import json

# import os



# root_dir = "/mnt/madehua/fooddata/vegfru-dataset/veg200_images"
    
# with open("/mnt/madehua/fooddata/vegfru-dataset/veg200_lists/processed_veg_test.txt", "r") as f:
#     lines = f.readlines()
    
# conversations = []
# for idx, line in enumerate(lines):
#     relative_path = line.strip()
    
#     category = relative_path.split("/")[0].replace("_", " ")


#     conversation = {
#     "question_id": idx, 
#     "image": root_dir + '/' + relative_path, 
#     "text": category, 
#     "category": "default"
#     }
#     conversations.append(conversation)

# with open("/mnt/madehua/fooddata/json_file/answers/veg200_answers.jsonl", "w") as f:
#     for conversation in conversations:
#         f.write(json.dumps(conversation) + "\n")

"""
foodx251
"""

import json

import os


root_dir = "/mnt/madehua/fooddata/FoodX-251/images"
    
with open("/mnt/madehua/fooddata/FoodX-251/annot/val.txt", "r") as f:
    lines = f.readlines()
    
conversations = []
for idx, line in enumerate(lines):
    relative_path = line.strip()
    
    category = relative_path.split("/")[0].replace("_", " ")

    conversation = {
    "question_id": idx, 
    "image": root_dir + '/' + relative_path, 
    "text": category, 
    "category": "default"
    }
    conversations.append(conversation)

with open("/mnt/madehua/fooddata/json_file/answers/foodx251_answers.jsonl", "w") as f:
    for conversation in conversations:
        f.write(json.dumps(conversation) + "\n")