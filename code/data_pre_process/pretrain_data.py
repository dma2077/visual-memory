import json
import random
import os
from tqdm import tqdm

# 列出所有要处理的 JSON 文件路径
file_paths = [
    '/llm_reco/dehua/code/visual-memory/questions/food101/dinov2_large_train_5_softmax.json',
    '/llm_reco/dehua/code/visual-memory/questions/food172/dinov2_large_train_5_softmax.json',
    # '/llm_reco/dehua/code/visual-memory/questions/food2k/dinov2_large_train_5_softmax.json',
    '/llm_reco/dehua/code/visual-memory/questions/fru92/dinov2_large_train_5_softmax.json',
    '/llm_reco/dehua/code/visual-memory/questions/veg200/dinov2_large_train_5_softmax.json',
    '/llm_reco/dehua/code/visual-memory/questions/foodx251/dinov2_large_train_5_softmax.json',
]

merged_conversations = []
missing_paths = []

# 第一层：遍历文件路径，显示文件处理进度
for path in tqdm(file_paths, desc="处理 JSON 文件", unit="file"):
    # 依次打开每个 JSON 文件
    with open(path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    # 第二层：遍历每个文件中的条目，显示条目处理进度
    for item in tqdm(data_list, desc=f"读取 {os.path.basename(path)} 中条目", unit="item", leave=False):
        # 原始图片路径
        orig_image_path = item["images"][0]
        # 路径替换逻辑
        new_image_path = orig_image_path \
            .replace("/map-vepfs/dehua/data/data/", "/llm_reco/dehua/data/food_data/") \
            .replace("/vegfru-dataset", "")

        # 判断文件是否存在
        if not os.path.exists(new_image_path):
            missing_paths.append(new_image_path)
            continue

        # 假设第二条对话（index=1）的 'value' 就是食物类别
        category = item["conversations"][1]["value"]

        single_conversation = {
            "messages": [
                {
                    "content": "<image>What is the category of the food?",
                    "role": "user"
                },
                {
                    "content": category,
                    "role": "assistant"
                }
            ],
            "images": [
                new_image_path
            ]
        }
        merged_conversations.append(single_conversation)

# 如果有缺失的路径，在控制台打印部分示例
if missing_paths:
    print(f"警告：共发现 {len(missing_paths)} 个不存在的图片路径。")
    for p in missing_paths[:10]:
        print(f"  - {p}")
    if len(missing_paths) > 10:
        print("  ...")

# 打乱顺序
random.shuffle(merged_conversations)

# 将打乱后的结果写入新的 JSON 文件
output_path = '/llm_reco/dehua/code/visual-memory/questions/merged_food_pretrain.json'
with open(output_path, 'w', encoding='utf-8') as out_f:
    json.dump(merged_conversations, out_f, ensure_ascii=False, indent=2)

print(f"已打乱并合并 {len(merged_conversations)} 条对话，保存到：{output_path}")
