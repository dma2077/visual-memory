import json
from tqdm import tqdm
import re
import random
from code.metric_calculator import get_label_file, build_id2category_dict
from code.utils import from_path2category

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

dataset_names = ["food101", "food172", "food2k", "fru92", "veg200", "foodx251"]


def merge_food_prompts_multi_model(filename):
    train_data = load_json(filename)
    train_num = train_data['train_size']
    train_paths = train_data['image_path'][:train_num]
    train_datas = []
    for idx, data in tqdm(enumerate(train_paths), total=len(train_paths), desc="Loading test data"):
        label = from_path2category(dataset_name, data, id2category_dict)
   
        prompt = "What is the category of the food?"
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
            "images": [data]
        }
        train_datas.append(train_data)
        
    return train_datas


for dataset_name in dataset_names:
    label_file = get_label_file(dataset_name=dataset_name)
    id2category_dict = build_id2category_dict(label_file, dataset_name)
    print(id2category_dict)
    data_dinov2_full = f'/map-vepfs/dehua/code/visual-memory/questions/{dataset_name}/dinov2_large_train_5_softmax_old.json'
    data_dinov2_fs = f'/map-vepfs/dehua/code/visual-memory/questions/{dataset_name}/dinov2_large_train_5_fewshot4_old.json'

    data_all = f'/map-vepfs/dehua/model/food_embeddings/siglip/{dataset_name}_data.json'
    data_fewshot = f'/map-vepfs/dehua/model/food_embeddings/siglip/{dataset_name}_data_fewshot4.json'
    prompts_1 = merge_food_prompts_multi_model(data_all)
    total_prompts = prompts_1
    random.shuffle(total_prompts)
    with open(f'/map-vepfs/dehua/data/{dataset_name}_finetune_raw.json', 'w') as f:
        json.dump(total_prompts, f, indent=4)   


    prompts_2 = merge_food_prompts_multi_model(data_fewshot)

    total_prompts = prompts_2
    random.shuffle(total_prompts)
    with open(f'/map-vepfs/dehua/data/{dataset_name}_finetune_raw_fewshot4.json', 'w') as f:
        json.dump(total_prompts, f, indent=4)  
