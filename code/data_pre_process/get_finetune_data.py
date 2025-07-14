import json
from tqdm import tqdm
import re
import random
def load_jsonl(filename):
    results = []
    with open(filename, 'r') as f:
        for line in f:
            line = json.loads(line)
            results.append(line)
    return results

def load_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

data_dinov2 = [
    '/map-vepfs/dehua/code/visual-memory/questions/multi_image/food101/train_5_softmax.json',
    '/map-vepfs/dehua/code/visual-memory/questions/multi_image/food172/train_5_softmax.json',
    '/map-vepfs/dehua/code/visual-memory/questions/multi_image/food2k/train_5_softmax.json'
]

label_file = '/map-vepfs/dehua/data/data/Food2k_complete_jpg/food2k_label2name_en.txt'

prompt_template = """
The categories of the 5 images most similar to this image are: {content}. Based on the information above, please answer the following questions. What dish is this? Just provide its category.
"""

with open(label_file, 'r') as f:
    lines = f.readlines()
labels = []
for line in lines:
    label = line.strip().split('--')[1]
    labels.append(label)
    

def merge_food_prompts_multi_model(filename1):
    data1 = load_json(filename1)
    train_datas = []
    for idx, data in tqdm(enumerate(data1), total=len(data1), desc="Loading test data"):
        prompt = data["conversations"][0]["value"].replace('<image> ', '')
        label = data["conversations"][1]["value"]
        question = "What dish is this?"
        if label.isdigit():
            label = labels[int(label)].strip()
   
        train_data = {
            "messages":  [
            {
                "content": f"<image>{prompt}",
                "role": "user"
            },
            {
                "content": label,
                "role": "assistant"
                }
            ],
            "images": [data["images"][0].replace('/mnt/madehua/fooddata', '/map-vepfs/dehua/data/data').replace('Food2k_complete', 'Food2k_complete_jpg')]
        }
        train_datas.append(train_data)
        train_data = {
            "messages":  [
            {
                "content": f"<image>{question}",
                "role": "user"
            },
            {
                "content": label,
                "role": "assistant"
                }
            ],
            "images": [data["images"][0].replace('/mnt/madehua/fooddata', '/map-vepfs/dehua/data/data').replace('Food2k_complete', 'Food2k_complete_jpg')]
        }
        
        train_datas.append(train_data)
        
    return train_datas

prompts_1 = merge_food_prompts_multi_model(data_dinov2[0])
total_prompts = prompts_1
random.shuffle(total_prompts)
with open('/map-vepfs/dehua/data/food101_dinov2_ns.json', 'w') as f:
    json.dump(total_prompts, f, indent=4)   


prompts_2 = merge_food_prompts_multi_model(data_dinov2[1])

total_prompts = prompts_2
random.shuffle(total_prompts)
with open('/map-vepfs/dehua/data/food172_dinov2_ns.json', 'w') as f:
    json.dump(total_prompts, f, indent=4)  

prompts_3 = merge_food_prompts_multi_model(data_dinov2[2])
total_prompts = prompts_2
random.shuffle(total_prompts)
with open('/map-vepfs/dehua/data/food2k_dinov2_ns.json', 'w') as f:
    json.dump(total_prompts, f, indent=4)

total_prompts = prompts_1 + prompts_2 + prompts_3
random.shuffle(total_prompts)
with open('/map-vepfs/dehua/data/food_mix_dinov2_ns.json', 'w') as f:
    json.dump(total_prompts, f, indent=4)       