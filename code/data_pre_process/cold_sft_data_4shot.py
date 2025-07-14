import json
import os
from tqdm import tqdm
import random
import re
from collections import defaultdict

# === 设置随机种子以确保结果可复现 ===
random.seed(42)

PROMPT = "<think>{think}</think><answer>{answer}</answer>"
SYSTEM_PROMPT = """
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>
"""

# === 配置路径 ===
file_paths = [
    '/llm_reco/dehua/code/visual-memory/questions/food101/dinov2_large_train_5_softmax.json',
    '/llm_reco/dehua/code/visual-memory/questions/food172/dinov2_large_train_5_softmax.json',
    '/llm_reco/dehua/code/visual-memory/questions/food2k/dinov2_large_train_5_softmax.json',
    '/llm_reco/dehua/code/visual-memory/questions/fru92/dinov2_large_train_5_softmax.json',
    '/llm_reco/dehua/code/visual-memory/questions/veg200/dinov2_large_train_5_softmax.json',
    '/llm_reco/dehua/code/visual-memory/questions/foodx251/dinov2_large_train_5_softmax.json',
]
result_paths = {
    "food101": "/llm_reco/dehua/data/food_finetune_data/food101_cold_sft_4shot.json",
    "food172": "/llm_reco/dehua/data/food_finetune_data/food172_cold_sft_4shot.json",
    "foodx251": "/llm_reco/dehua/data/food_finetune_data/foodx251_cold_sft_4shot.json",
    "food2k": "/llm_reco/dehua/data/food_finetune_data/food2k_cold_sft_4shot.json",
    "veg200": "/llm_reco/dehua/data/food_finetune_data/veg200_cold_sft_4shot.json",
    "fru92": "/llm_reco/dehua/data/food_finetune_data/fru92_cold_sft_4shot.json"
}
file_paths1 = {
    "food101": "/llm_reco/dehua/data/food_data/food-101/Qwen2.5-VL-72B-Instruct-cot.jsonl",
    "food172": "/llm_reco/dehua/data/food_data/VireoFood172/Qwen2.5-VL-72B-Instruct-cot.jsonl",
    "foodx251": "/llm_reco/dehua/data/food_data/FoodX-251/Qwen2.5-VL-72B-Instruct-cot.jsonl",
    "food2k": "/llm_reco/dehua/data/food_data/Food2k_complete_jpg/Qwen2.5-VL-72B-Instruct-cot.jsonl",
    "veg200": "/llm_reco/dehua/data/food_data/veg200_lists/Qwen2.5-VL-72B-Instruct-cot.jsonl",
    "fru92": "/llm_reco/dehua/data/food_data/fru92_lists/Qwen2.5-VL-72B-Instruct-cot.jsonl"
}

# === 4-shot 配置 ===
FEW_SHOT = 4

def sample_few_shot_data(data_list, few_shot=4, seed=42):
    """
    对数据进行4-shot采样，确保每个类别最多包含4个样本
    
    Args:
        data_list: 原始数据列表
        few_shot: 每个类别的最大样本数量，固定为4
        seed: 随机种子
    
    Returns:
        采样后的数据列表
    """
    # 设置随机种子
    random.seed(seed)
    
    # 按类别分组
    category_data = defaultdict(list)
    for item in data_list:
        category = item["conversations"][1]["value"]
        category_data[category].append(item)
    
    # 对每个类别进行采样
    sampled_data = []
    total_original = 0
    total_sampled = 0
    
    for category, items in category_data.items():
        total_original += len(items)
        if len(items) <= few_shot:
            # 如果样本数量少于或等于few_shot，使用全部样本
            sampled_items = items
        else:
            # 随机采样few_shot个样本
            sampled_items = random.sample(items, few_shot)
        
        sampled_data.extend(sampled_items)
        total_sampled += len(sampled_items)
        print(f"[INFO] Category '{category}': {len(sampled_items)}/{len(items)} samples selected")
    
    print(f"[INFO] Total samples: {total_sampled}/{total_original}")
    
    # 随机打乱最终结果
    random.shuffle(sampled_data)
    return sampled_data

# === Attribute 解析函数 ===
def parse_attribute(attribute):
    color, texture, shape = None, None, None
    lines = [line.strip() for line in attribute.strip().split('\n') if line.strip()]
    
    for line in lines:
        if "Color:" in line:
            color = line.split("Color:")[1].strip()
        elif "Texture:" in line:
            texture = line.split("Texture:")[1].strip()
        elif "Shape:" in line:
            shape = line.split("Shape:")[1].strip()
        elif "Composition" in line:
            composition = line.split("Composition:")[1].strip()
        elif "Cooking Style" in line:
            cooking_style = line.split("Cooking Style:")[1].strip()
    
    missing = []
    if color is None:
        missing.append("Color")
    if texture is None:
        missing.append("Texture")
    if shape is None:
        missing.append("Shape")

    if missing:
        return False, f"Missing fields: {', '.join(missing)}", (None, None, None)
    
    return True, "OK", (shape, texture, composition, color, cooking_style)

# === Attribute 解析函数 ===
def parse_attribute1(attribute: str):
    shape = texture = composition = color = cooking_style = None

    # 按行拆分，并去掉空行
    lines = [line.strip() for line in attribute.strip().split('\n') if line.strip()]

    # 用正则匹配 "数字. 文本" 形式
    for line in lines:
        m = re.match(r'^(\d+)\.\s*(.+)$', line)
        if not m:
            continue
        idx = int(m.group(1))
        text = m.group(2).strip()

        if idx == 1:
            shape = text
        elif idx == 2:
            texture = text
        elif idx == 3:
            composition = text
        elif idx == 4:
            color = text
        elif idx == 5:
            cooking_style = text

    # 检查缺失项
    missing = []
    if shape is None:
        missing.append("Shape")
    if texture is None:
        missing.append("Texture")
    if composition is None:
        missing.append("Composition")
    if color is None:
        missing.append("Color")
    if cooking_style is None:
        missing.append("Cooking Style")

    if missing:
        return False, f"Missing fields: {', '.join(missing)}", (None, None, None, None, None)

    return True, "OK", (shape, texture, composition, color, cooking_style)

# === 构建 category -> attribute 映射 ===
dataset_c2a = {}
for name, path in file_paths1.items():
    c2a = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            c2a[data['category']] = data['output']
    dataset_c2a[name] = c2a

# === 主程序执行 ===
if __name__ == "__main__":
    print("=== Food Dataset 4-Shot Processing ===")
    
    missing_paths = []

    # === 主循环：处理每个数据文件 ===
    for path in tqdm(file_paths, desc="处理 JSON 文件", unit="file"):
        with open(path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)

        dataset_name = path.split('/')[-2]
        print(f"[INFO] Processing dataset: {dataset_name}")
        print(f"[INFO] Original data size: {len(data_list)}")

        # 进行4-shot采样
        print(f"[INFO] Applying 4-shot sampling...")
        sampled_data = sample_few_shot_data(data_list, few_shot=4)
        print(f"[INFO] Sampled data size: {len(sampled_data)}")

        merged_conversations = []

        for idx, item in tqdm(enumerate(sampled_data), desc=f"读取 {os.path.basename(path)}", unit="item", leave=False):
            orig_image_path = item["images"][0]
            new_image_path = orig_image_path \
                .replace("/map-vepfs/dehua/data/data/", "/llm_reco/dehua/data/food_data/") \
                .replace("/vegfru-dataset", "")
            if not os.path.exists(new_image_path):
                missing_paths.append(new_image_path)
                continue

            category = item["conversations"][1]["value"]
            attribute = dataset_c2a[dataset_name].get(category, "")
            valid, message, text_tuple = parse_attribute(attribute)

            if not valid:
                valid, message, text_tuple = parse_attribute1(attribute)
                if not valid:
                    print(f"line {idx}, [Format Error] {category}: {message}")
                    continue
            (shape, texture, composition, color, cooking_style) = text_tuple
            think = f"Shape is {shape}. Texture is {texture}. Composition is {composition}. Color is {color}. Cooking Style is {cooking_style}."
            answer = category
            answer_text = PROMPT.format(think=think, answer=answer)
            merged_conversations.append({
                "messages": [
                    {"role": "user", "content": "<image>Please analyze these food attributes in the image: shape, texture, composition, color, and cooking style. Then identify the food category."},
                    {"role": "assistant", "content": answer_text}
                ],
                "images": [new_image_path]  
            })

        # === 保存结果 ===
        output_path = result_paths[dataset_name]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_conversations, f, ensure_ascii=False, indent=2)
        
        print(f"[INFO] Saved {len(merged_conversations)} conversations to {output_path}")

    print(f"\n✅ 4-Shot processing completed. Missing image count: {len(missing_paths)}") 