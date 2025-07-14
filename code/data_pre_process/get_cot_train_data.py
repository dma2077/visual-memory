import json
from tqdm import tqdm
import re
import random
def load_jsonl(filename):
    results = []
    with open(filename, 'r') as f:
        for line in f:
            try:
                line = json.loads(line)
                results.append(line)
            except:
                continue
    return results

def load_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

attribute_train_data = [
    '/map-vepfs/dehua/code/visual-memory/answers/food101/Qwen2.5-VL-72B-Instruct-infer_train_attribute.jsonl',
    '/map-vepfs/dehua/code/visual-memory/answers/food172/Qwen2.5-VL-72B-Instruct-infer_train_attribute.jsonl',
    '/map-vepfs/dehua/code/visual-memory/answers/veg200/Qwen2.5-VL-72B-Instruct-infer_train_attribute.jsonl',
    '/map-vepfs/dehua/code/visual-memory/answers/fru92/Qwen2.5-VL-72B-Instruct-infer_train_attribute.jsonl',
    '/map-vepfs/dehua/code/visual-memory/answers/foodx251/Qwen2.5-VL-72B-Instruct-infer_train_attribute.jsonl',
]
groundtruth_data = [
    '/map-vepfs/dehua/code/visual-memory/answers/groundtruth/food101_train_groundtruth.jsonl',
    '/map-vepfs/dehua/code/visual-memory/answers/groundtruth/food172_train_groundtruth.jsonl',
    '/map-vepfs/dehua/code/visual-memory/answers/groundtruth/veg200_train_groundtruth.jsonl',
    '/map-vepfs/dehua/code/visual-memory/answers/groundtruth/fru92_train_groundtruth.jsonl',
    '/map-vepfs/dehua/code/visual-memory/answers/groundtruth/foodx251_train_groundtruth.jsonl',
]

question_template = """
"This is an image containing a food. Please identify the categories of the food based on the image.\n"
"Output the thinking process in <think> </think> and final answer in <answer> </answer> tags. "
"The output answer format should be as follows:\n"
"<think> ... </think> <answer>category name</answer>\n"
"Please strictly follow the format."
"""

label_template = """
"<think> {attribute} </think> <answer> {label} </answer>\n"
"""

def merge_food_prompts_multi_model(filename1, filename2):
    data1 = load_jsonl(filename1)
    data2 = load_jsonl(filename2)
    train_datas = []
    for idx, data in tqdm(enumerate(data1), total=len(data1), desc="Loading test data"):
        attribute = data['text']
        label = data2[idx]["groundtruth"]
        train_data = {
            "messages":  [
            {
                "content": f"<image>{question_template}",
                "role": "user"
            },
            {
                "content": label_template.format(attribute=attribute, label=label),
                "role": "assistant"
                }
            ],
            "images": [data["image"].replace('/mnt/madehua/fooddata', '/map-vepfs/dehua/data/data')]
        }
        train_datas.append(train_data)
    return train_datas

# datasets = ["food101", "food172", 'veg200', "fru92", "foodx251"]
# for idx, dataset in enumerate(datasets):
#     prompts_1 = merge_food_prompts_multi_model(attribute_train_data[idx], groundtruth_data[idx])

#     total_prompts = prompts_1
#     random.shuffle(total_prompts)
#     with open(f'/map-vepfs/dehua/data/{dataset}_attribute_train.json', 'w') as f:
#         json.dump(total_prompts, f, indent=4)   


prompts_1 = merge_food_prompts_multi_model(attribute_train_data[0], groundtruth_data[0])
prompts_2 = merge_food_prompts_multi_model(attribute_train_data[1], groundtruth_data[1])
prompts_3 = merge_food_prompts_multi_model(attribute_train_data[2], groundtruth_data[2])
prompts_4 = merge_food_prompts_multi_model(attribute_train_data[3], groundtruth_data[3])
prompts_5 = merge_food_prompts_multi_model(attribute_train_data[4], groundtruth_data[4])
# total_prompts = prompts_1 + prompts_2 + prompts_3 + prompts_4 + prompts_5
random.shuffle(prompts_1)
random.shuffle(prompts_2)
random.shuffle(prompts_3)
random.shuffle(prompts_4)
random.shuffle(prompts_5)

with open(f'/map-vepfs/dehua/data/food101_attribute_train.json', 'w') as f:
    json.dump(prompts_1, f, indent=4)   

with open(f'/map-vepfs/dehua/data/food172_attribute_train.json', 'w') as f:
    json.dump(prompts_2, f, indent=4)   

with open(f'/map-vepfs/dehua/data/veg200_attribute_train.json', 'w') as f:
    json.dump(prompts_3, f, indent=4)   

with open(f'/map-vepfs/dehua/data/fru92_attribute_train.json', 'w') as f:
    json.dump(prompts_4, f, indent=4)   

with open(f'/map-vepfs/dehua/data/foodx251_attribute_train.json', 'w') as f:
    json.dump(prompts_5, f, indent=4)   