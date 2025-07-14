import json
from utils import *

def calculate_accuracy(jsonl_file_path):
    total_count = 0
    match_count = 0

    with open(jsonl_file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            image_path = data['image']
            text_value = data['text']

            # 获取倒数第二个元素并替换下划线为空格
            image_category = image_path.split('/')[-2].replace('_', ' ')

            # 检查 text 是否与处理后的 image_category 相同
            if text_value == image_category:
                match_count += 1
            total_count += 1

    # 计算准确率
    accuracy = match_count / total_count if total_count > 0 else 0
    print(f"Matched: {match_count}, Total: {total_count}, Accuracy: {accuracy:.2%}")

# 使用函数并传入文件路径
jsonl_file_path = 'answers/multi_turn/food101/qwen2-vl-7b-3-shot.jsonl'
# calculate_accuracy(jsonl_file_path)


def calculate_accuracy(jsonl_file_path):
    total_count = 0
    match_count = 0    
    with open(jsonl_file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            truth_category = data['image'].split('/')[-2]
            predict_category = data['text']
            if truth_category == predict_category:
                match_count += 1
            total_count += 1
    accuracy = match_count / total_count if total_count > 0 else 0
    print(f"Matched: {match_count}, Total: {total_count}, Accuracy: {accuracy:.2%}")

# jsonl_file_path = '/map-vepfs/dehua/code/visual-memory/answers/food172/qwen2-vl-7b-2w_mix_k5_softmax.jsonl'
# calculate_accuracy(jsonl_file_path)
# jsonl_file_path = '/map-vepfs/dehua/code/visual-memory/answers/food172/qwen2-vl-7b-3w_mix_k5_softmax.jsonl'
# calculate_accuracy(jsonl_file_path)



dataname = 'food101'
filename = '/map-vepfs/dehua/code/visual-memory/answers/food101/qwen2-vl-7b_food172_2068_mix_k5_softmax.jsonl'
filename1 = '/map-vepfs/dehua/code/visual-memory/answers/food101/qwen2-vl-7b_food172_2068_mix_k5_softmax.jsonl'
label_file = '/map-vepfs/dehua/data/data/food-101/meta/classes.txt'


with open(label_file, 'r') as file:
    lines = file.readlines()
    
id2category_dict = {}

if dataname == 'food101':
    for idx, line in enumerate(lines):
        line = line.strip()
        id2category_dict[idx] = line
elif dataname == "food172":
    for idx, line in enumerate(lines):
        line = line.strip()
        id2category_dict[idx + 1] = line

print(id2category_dict)
    
import copy
def replace_data(filename, id2category_dict):
    lines = load_jsonl(filename)
    new_lines = []
    for line in lines:
        new_line = copy.copy(line)  # 创建字典的副本
        new_line["text"] = id2category_dict[int(line["text"])]
        new_lines.append(new_line)
    return new_lines


new_lines = replace_data(filename, id2category_dict)
save_jsonl(filename1, new_lines)
