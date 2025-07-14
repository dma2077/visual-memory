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
    '/map-vepfs/dehua/code/visual-memory/questions/food2k/dinov2_large_train_5_softmax.json',
    '/map-vepfs/dehua/code/visual-memory/questions/food2k/dinov2_large_train_5_fewshot4.json',
    '/map-vepfs/dehua/code/visual-memory/questions/food2k/dinov2_large_train_5_fewshot8.json',
    '/map-vepfs/dehua/code/visual-memory/questions/food2k/dinov2_large_train_5_fewshot16.json',
]

data_siglip = [
    '/map-vepfs/dehua/code/visual-memory/questions/food2k/siglip_train_5_softmax.json',
    '/map-vepfs/dehua/code/visual-memory/questions/food2k/siglip_train_5_fewshot4.json',
    '/map-vepfs/dehua/code/visual-memory/questions/food2k/siglip_train_5_fewshot8.json',
    '/map-vepfs/dehua/code/visual-memory/questions/food2k/siglip_train_5_fewshot16.json',
]

label_file = '/map-vepfs/dehua/data/data/Food2k_complete_jpg/food2k_label2name_en.txt'

prompt_template = """
We use two retrievers to identify the category of the current image. The food categories returned by the first retriever are {content1}, and those returned by the second retriever are {content2}. Based on the current image and the information provided by both retrievers, please make a comprehensive judgment on the category of the current image.
"""

with open(label_file, 'r') as f:
    lines = f.readlines()
labels = []
for line in lines:
    label = line.strip().split('--')[1]
    labels.append(label)


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
        categories_list = [used_categories[category.lower()] for category in categories_list]
        try:
            question2 = data2[idx]["conversations"][0]["value"]
            categories2 = re.search(r'are:\s*(.*?)(?:\.|$)', question2).group(1)
            categories2_list = categories2.split(', ')[:k1]
            categories2_list = [used_categories[category.lower()] for category in categories2_list]
        except:
            print(data2[idx])
        
        similarity_list1 = data["similarities"]
        similarity_list2 = data2[idx]["similarities"]
        content1 = ", ".join([f"{category}" for category in categories_list])
        content2 = ", ".join([f"{category}" for category in categories2_list])
        prompt = prompt_template.format(content1=content1, content2=content2)
        label = data["conversations"][1]["value"]
        label = used_categories[label.lower()]

        
        if label.isdigit():
            label = used_categories[int(label)].strip()
   
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
            "images": [data["images"][0].replace('/mnt/madehua/fooddata', '/map-vepfs/dehua/data/data')]
        }
        
        
        train_datas.append(train_data)
        
    return train_datas


k1 = 3
prompts_1 = merge_food_prompts_multi_model(data_dinov2[0], data_siglip[0], k1)
total_prompts = prompts_1
random.shuffle(total_prompts)
with open('/map-vepfs/dehua/data/food2k_3_image_rank_nocategory.json', 'w') as f:
    json.dump(total_prompts, f, indent=4)   


# prompts_2 = merge_food_prompts_multi_model(data_dinov2[1], data_siglip[1], k1)

# total_prompts = prompts_2
# random.shuffle(total_prompts)
# with open('/map-vepfs/dehua/data/food2k_3_image_rank_fewshot4_nocategory.json', 'w') as f:
#     json.dump(total_prompts, f, indent=4)  


# prompts_3 = merge_food_prompts_multi_model(data_dinov2[2], data_siglip[2], k1)

# total_prompts = prompts_3
# random.shuffle(total_prompts)
# with open('/map-vepfs/dehua/data/food2k_3_image_rank_fewshot8_nocategory.json', 'w') as f:
#     json.dump(total_prompts, f, indent=4)  

# prompts_4 = merge_food_prompts_multi_model(data_dinov2[3], data_siglip[3], k1)  

# total_prompts = prompts_4
# random.shuffle(total_prompts)
# with open('/map-vepfs/dehua/data/food2k_3_image_rank_fewshot16_nocategory.json', 'w') as f:
#     json.dump(total_prompts, f, indent=4)  

# total_prompts = prompts_1 + prompts_2 + prompts_3 + prompts_4
# # #total_prompts = prompts_1 + prompts_2
# # print(len(total_prompts))
# # print(total_prompts[0])
# random.shuffle(total_prompts)
# with open('/map-vepfs/dehua/data/food2k_3_similarity_fewshot.json', 'w') as f:
#     json.dump(total_prompts, f, indent=4)       
