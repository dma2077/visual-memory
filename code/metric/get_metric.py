import json
import tqdm
import re
import json
from tqdm import tqdm
file_name = '/map-vepfs/dehua/code/visual-memory/questions/veg200/dinov2_large_train_5_softmax_old.json'
groundtruth_file = "/map-vepfs/dehua/code/visual-memory/answers/groundtruth/veg200_train_groundtruth.jsonl"
food2k_label_file = '/map-vepfs/dehua/data/data/Food2k_complete_jpg/food2k_label2name_en.txt'
food172_label_file = '/map-vepfs/dehua/data/data/VireoFood172/SplitAndIngreLabel/FoodList.txt'

def get_food2k_label(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        category_dict = {}
        for data in file.readlines():
            category_id, category = data.split('--')[0], data.split('--')[1].strip()
            category_id = str(int(category_id)) 
            category_dict[category_id] = category
    return category_dict

def get_food172_label(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        category_dict = {}
        for idx, data in enumerate(file.readlines()):
            category = data.strip()
            category_id = str(int(idx + 1)) 
            category_dict[category_id] = category
    return category_dict


food2k_category_dict = get_food2k_label(food2k_label_file)
food172_category_dict = get_food172_label(food172_label_file)

def get_label_from_path(image_path, food2k_dict=None, food172_dict=None):
    """
    从 image_path 解析类别标签，适用于不同数据集的两类处理方式：
    1. food101, veg200, foodx251, fru92 这类数据集，类别信息在路径倒数第二级目录。
    2. food2k, food172 这类数据集，类别 ID 在路径倒数第二级目录，需通过映射字典转换为名称。

    参数:
      image_path (str): 图像文件的路径。
      food2k_dict (dict): food2k 数据集的类别映射字典 {类别ID: 类别名称}。
      food172_dict (dict): food172 数据集的类别映射字典 {类别ID: 类别名称}。

    返回:
      str: 解析出的类别名称或 ID。
    """
    # 获取路径倒数第二级目录作为类别标识
    category_id = image_path.split('/')[-2]

    # 处理方式 1: 直接使用路径中的类别名称（food101, veg200, foodx251, fru92）
    if any(dataset in image_path for dataset in ["food-101", "veg200", "foodx251", "fru92"]):
        return category_id.replace('_', ' ')  # 替换 `_` 为空格，确保格式一致

    # 处理方式 2: 解析 food2k 数据集（需要 ID 到名称的转换）
    elif "food2k" in image_path.lower():
        return food2k_dict.get(category_id, "unknown")  # 查找 ID 对应的类别名

    # 处理方式 3: 解析 food172 数据集（需要 ID 到名称的转换）
    elif "food172" in image_path.lower():
        return food172_dict.get(category_id, "unknown")  # 查找 ID 对应的类别名

    # 默认返回原始类别
    return category_id


def extract_food_category(text):
    """
    从文本中提取 <food category>: 后面的食物名称。
    """
    match = re.search(r"<food category>:\s*(.+)", text)
    return match.group(1).strip() if match else None

filename = '/map-vepfs/dehua/code/visual-memory/answers/food101/qwen2vl_food101_attribute_food101_None_cot_k5.jsonl'

import json

def calculate_accuracy(filename, output_filename="output.txt"):
    """
    计算 food category 预测的准确率，并将不匹配的结果写入文件。

    参数:
      filename (str): JSONL 格式的文件路径，每行是一个 JSON 对象。
      output_filename (str): 结果输出文件路径，记录预测错误的情况，每行格式为: 真实类别, 预测类别。

    返回:
      tuple: (正确匹配数量, 总数量, 预测准确率)
    """
    with open(filename, 'r', encoding='utf-8') as file:
        data_list = [json.loads(line) for line in file]  # 先解析整个文件，减少 I/O 操作

    correct_count = 0
    total_count = len(data_list)

    with open(output_filename, 'w', encoding='utf-8') as output_file:
        for data in data_list:
            true_label = get_label_from_path(data["image"], food2k_category_dict, food172_category_dict)
            predicted_label = extract_food_category(data["text"])

            if true_label == predicted_label:
                correct_count += 1
            else:
                # 只写入不匹配的情况
                output_file.write(f"{true_label},{predicted_label}\n")

    accuracy = correct_count / total_count if total_count > 0 else 0.0  # 避免除零错误
    return correct_count, total_count, accuracy

# 示例调用
correct, total, acc = calculate_accuracy(filename)
print(f"Correct: {correct}, Total: {total}, Accuracy: {acc:.2%}")  # 显示百分比格式
accuracy = calculate_accuracy(filename)
print(accuracy)