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

data_dinov2 = [
    '/map-vepfs/dehua/code/visual-memory/questions/food2k/dinov2_large_train_5_softmax_old.json',
    '/map-vepfs/dehua/code/visual-memory/questions/food2k/dinov2_large_train_5_fewshot4_old.json',
    '/map-vepfs/dehua/code/visual-memory/questions/food2k/dinov2_large_train_5_fewshot8.json',
    '/map-vepfs/dehua/code/visual-memory/questions/food2k/dinov2_large_train_5_fewshot16_old.json',
]

data_siglip = [
    '/map-vepfs/dehua/code/visual-memory/questions/food2k/siglip_train_5_softmax.json',
    '/map-vepfs/dehua/code/visual-memory/questions/food2k/siglip_train_5_fewshot4.json',
    '/map-vepfs/dehua/code/visual-memory/questions/food2k/siglip_train_5_fewshot8.json',
    '/map-vepfs/dehua/code/visual-memory/questions/food2k/siglip_train_5_fewshot16.json',
]

label_file = '/map-vepfs/dehua/data/data/Food2k_complete_jpg/food2k_label2name_en.txt'

prompt_template = """
We use two retrievers to identify the category of the current food. The food categories and similarity scores returned by the first retriever are {content1}, and those returned by the second retriever are {content2}. Based the information provided by both retrievers, please make a comprehensive judgment on the category of the current food.
"""

with open(label_file, 'r') as f:
    lines = f.readlines()
labels = []
used_categories = {}
for idx, line in enumerate(lines):
    label = line.strip().split('--')[1]
    labels.append(label)
    used_categories[label.lower()] = f"category{idx}"

def merge_food_prompts_multi_model(filename1, filename2, k1):

    data1 = load_json(filename1)
    print(filename1)
    data2 = load_json(filename2)
    print(filename2)
    train_datas = []
    

    for idx, data in tqdm(enumerate(data1), total=len(data1), desc="Loading test data"):
        question = data["conversations"][0]["value"]
        categories1 = re.search(r'are:\s*(.*?)(?:\.|$)', question).group(1)
        categories_list = categories1.split(', ')[:k1]

        # 更新 categories_list 为对应的 category ID
        categories_list = [used_categories[category.lower()] for category in categories_list]

        try:
            question2 = data2[idx]["conversations"][0]["value"]
            categories2 = re.search(r'are:\s*(.*?)(?:\.|$)', question2).group(1)
            categories2_list = categories2.split(', ')[:k1]

            # 更新 categories2_list 为对应的 category ID
            categories2_list = [used_categories[category.lower()] for category in categories2_list]
        except:
            print(data2[idx])

        content1 = ", ".join([f"{category}" for category in categories_list])
        content2 = ", ".join([f"{category}" for category in categories2_list])
        prompt = prompt_template.format(content1=content1, content2=content2)
        
        # 将 label 与 categories_list 进行对应
        label = data["conversations"][1]["value"]
        label = used_categories[label.lower()]
        
        train_data = {
            "instruction": prompt,
            "input": "",
            "output": label  # 使用替代的 label
        }
        train_datas.append(train_data)
    return train_datas

k1 = 3
prompts_1 = merge_food_prompts_multi_model(data_dinov2[0], data_siglip[0], k1)
total_prompts = prompts_1
random.shuffle(total_prompts)
with open('/map-vepfs/dehua/data/food2k_3_rank.json', 'w') as f:
    json.dump(total_prompts, f, indent=4)

prompts_2 = merge_food_prompts_multi_model(data_dinov2[1], data_siglip[1], k1)
total_prompts = prompts_2
random.shuffle(total_prompts)
with open('/map-vepfs/dehua/data/food2k_3_rank_fewshot4.json', 'w') as f:
    json.dump(total_prompts, f, indent=4)
    
# prompts_3 = merge_food_prompts_multi_model(data_dinov2[1], data_siglip[1], k1)
# total_prompts = prompts_3
# random.shuffle(total_prompts)
# with open('/map-vepfs/dehua/data/food2k_3_rank_fewshot8.json', 'w') as f:
#     json.dump(total_prompts, f, indent=4)