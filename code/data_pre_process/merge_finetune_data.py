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

data_siglip = [
    '/map-vepfs/dehua/code/visual-memory/answers/food101/food101_train_results_softmax.jsonl',
    '/map-vepfs/dehua/code/visual-memory/answers/food172/food172_train_results_softmax.jsonl',
    '/map-vepfs/dehua/code/visual-memory/answers/food2k/food2k_train_results_softmax.jsonl'
]

data_dinov2_few_shot = [
    
]


label_file = '/map-vepfs/dehua/data/data/Food2k_complete_jpg/food2k_label2name_en.txt'

prompt_template = """
We use two retrievers to identify the category of the current image. The first retriever provided food categories based on similarity scores ({content1}), while the second retriever used Softmax normalization to retrieve food categories ({content2}). Based on the results from both retrievers, determine the most appropriate category for the image. Please provide only the category name, without including similarity scores or additional details.
"""

with open(label_file, 'r') as f:
    lines = f.readlines()
labels = []
for line in lines:
    label = line.strip().split('--')[1]
    labels.append(label)
    

def merge_food_prompts_multi_model(filename1, filename2):

    data1 = load_json(filename1)
    data2 = load_jsonl(filename2)
    train_datas = []
    for idx, data in tqdm(enumerate(data1), total=len(data1), desc="Loading test data"):
        
        question = data["conversations"][0]["value"]
        categories1 = re.search(r'are:\s*(.*?)(?:\.|$)', question).group(1)
        categories_list = categories1.split(', ')

        # Add final user question
        similarities1 = data["similarities"]    
        categories2 = data2[idx]["categories"]
        similarities2 = data2[idx]["similarities"]
        content1 = ", ".join([f"{category}: {similarity:.2f}" for category, similarity in zip(categories_list, similarities1)])
        content2 = ", ".join([f"{category}: {similarity:.2f}" for category, similarity in zip(categories2, similarities2)])

        prompt = prompt_template.format(content1=content1, content2=content2)
        
        
        
        label = data["conversations"][1]["value"]
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
        
    return train_datas

prompts_1 = merge_food_prompts_multi_model(data_dinov2[0], data_siglip[0])
# total_prompts = prompts_1
# random.shuffle(total_prompts)
# with open('/map-vepfs/dehua/data/food101_5.json', 'w') as f:
#     json.dump(total_prompts, f, indent=4)   


prompts_2 = merge_food_prompts_multi_model(data_dinov2[1], data_siglip[1])

total_prompts = prompts_2
random.shuffle(total_prompts)
with open('/map-vepfs/dehua/data/food172_5.json', 'w') as f:
    json.dump(total_prompts, f, indent=4)  


prompts_3 = merge_food_prompts_multi_model(data_dinov2[2], data_siglip[2])

# total_prompts = prompts_1 + prompts_2 + prompts_3
# #total_prompts = prompts_1 + prompts_2
# print(len(total_prompts))
# print(total_prompts[0])
# random.shuffle(total_prompts)
# with open('/map-vepfs/dehua/data/food_mix_5.json', 'w') as f:
#     json.dump(total_prompts, f, indent=4)       