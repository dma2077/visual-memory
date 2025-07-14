import json
import os
import re
from tqdm import tqdm

# === 配置路径 ===
file_paths = {
    "food101": "/llm_reco/dehua/data/food_data/food-101/Qwen2.5-VL-72B-Instruct-cot.jsonl",
    "food172": "/llm_reco/dehua/data/food_data/VireoFood172/Qwen2.5-VL-72B-Instruct-cot.jsonl",
    "foodx251": "/llm_reco/dehua/data/food_data/FoodX-251/Qwen2.5-VL-72B-Instruct-cot.jsonl",
    "food2k": "/llm_reco/dehua/data/food_data/Food2k_complete_jpg/Qwen2.5-VL-72B-Instruct-cot.jsonl",
    "veg200": "/llm_reco/dehua/data/food_data/veg200_lists/Qwen2.5-VL-72B-Instruct-cot.jsonl",
    "fru92": "/llm_reco/dehua/data/food_data/fru92_lists/Qwen2.5-VL-72B-Instruct-cot.jsonl"
}

# === Attribute 解析函数 ===
def parse_attribute(attribute):
    color, texture, shape, composition, cooking_style = None, None, None, None, None
    lines = [line.strip() for line in attribute.strip().split('\n') if line.strip()]
    
    for line in lines:
        if "Color:" in line:
            color = line.split("Color:")[1].strip()
        elif "Texture:" in line:
            texture = line.split("Texture:")[1].strip()
        elif "Shape:" in line:
            shape = line.split("Shape:")[1].strip()
        elif "Composition:" in line:
            composition = line.split("Composition:")[1].strip()
        elif "Cooking Style:" in line:
            cooking_style = line.split("Cooking Style:")[1].strip()
    
    missing = []
    if color is None:
        missing.append("Color")
    if texture is None:
        missing.append("Texture")
    if shape is None:
        missing.append("Shape")
    if composition is None:
        missing.append("Composition")
    if cooking_style is None:
        missing.append("Cooking Style")

    if missing:
        return False, f"Missing fields: {', '.join(missing)}", (None, None, None, None, None)
    
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

# === 统计每个类别每个属性的最高频描述 ===
def get_most_frequent_attributes(all_category_data):
    """
    统计每个类别每个属性的最高频描述
    """
    truth_attrs = {}
    
    for category, attr_lists in all_category_data.items():
        print(f"[INFO] 统计类别 '{category}' 的属性频率，共有 {len(attr_lists)} 条数据")
        
        # 统计每个属性的频率
        attr_counts = {
            "Shape": {},
            "Texture": {},
            "Composition": {},
            "Color": {},
            "Cooking Style": {}
        }
        
        # 计算每个属性值的出现次数
        for attrs in attr_lists:
            shape, texture, composition, color, cooking_style = attrs
            
            attr_counts["Shape"][shape] = attr_counts["Shape"].get(shape, 0) + 1
            attr_counts["Texture"][texture] = attr_counts["Texture"].get(texture, 0) + 1
            attr_counts["Composition"][composition] = attr_counts["Composition"].get(composition, 0) + 1
            attr_counts["Color"][color] = attr_counts["Color"].get(color, 0) + 1
            attr_counts["Cooking Style"][cooking_style] = attr_counts["Cooking Style"].get(cooking_style, 0) + 1
        
        # 选择每个属性的最高频描述
        most_frequent_attrs = {}
        for attr_name, value_counts in attr_counts.items():
            if value_counts:  # 确保不为空
                most_frequent_value = max(value_counts.items(), key=lambda x: x[1])
                most_frequent_attrs[attr_name] = most_frequent_value[0]
                
                # 显示统计信息
                print(f"  {attr_name}: '{most_frequent_value[0]}' (出现 {most_frequent_value[1]} 次)")
                if len(value_counts) > 1:
                    print(f"    其他选项: {dict(sorted(value_counts.items(), key=lambda x: x[1], reverse=True)[1:])}")
        
        truth_attrs[category] = most_frequent_attrs
        print()
    
    return truth_attrs

# === 主函数：生成truth_attrs字典 ===
def generate_truth_attrs():
    # 收集所有数据，按类别分组
    all_category_data = {}
    
    for dataset_name, file_path in file_paths.items():
        print(f"[INFO] Processing dataset: {dataset_name}")
        
        if not os.path.exists(file_path):
            print(f"[WARNING] File not found: {file_path}")
            continue
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(tqdm(f, desc=f"读取 {dataset_name}", unit="line"), 1):
                try:
                    data = json.loads(line.strip())
                    category = data['category']
                    output = data['output']
                    
                    # 先尝试第一个解析函数
                    valid, message, text_tuple = parse_attribute(output)
                    
                    if not valid:
                        # 第一个函数失败，尝试第二个函数
                        valid, message, text_tuple = parse_attribute1(output)
                        if not valid:
                            print(f"line {line_num}, [Format Error] {category}: {message}")
                            continue
                    
                    # 解构元组获取5个属性
                    (shape, texture, composition, color, cooking_style) = text_tuple
                    
                    # 收集数据，按类别分组
                    if category not in all_category_data:
                        all_category_data[category] = []
                    
                    all_category_data[category].append((shape, texture, composition, color, cooking_style))
                        
                except json.JSONDecodeError as e:
                    print(f"[JSON Error] {dataset_name} line {line_num}: {e}")
                    continue
                except KeyError as e:
                    print(f"[Key Error] {dataset_name} line {line_num}: Missing key {e}")
                    continue
    
    print(f"\n📊 数据收集完成，共有 {len(all_category_data)} 个类别")
    
    # 统计每个类别每个属性的最高频描述
    print("\n🔍 开始统计每个类别的最高频属性描述...")
    truth_attrs = get_most_frequent_attributes(all_category_data)
    
    return truth_attrs

# === 保存结果 ===
def save_truth_attrs(truth_attrs, output_path="/llm_reco/dehua/data/food_data/truth_attrs.json"):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(truth_attrs, f, ensure_ascii=False, indent=2)
    print(f"✅ Truth attrs saved to: {output_path}")

# === 主程序 ===
if __name__ == "__main__":
    print("开始生成 truth_attrs 字典...")
    truth_attrs = generate_truth_attrs()
    
    print(f"\n📊 最终统计信息:")
    print(f"总类别数: {len(truth_attrs)}")
    
    # 显示前5个类别作为示例
    print(f"\n🔍 前5个类别的最高频属性示例:")
    for i, (category, attrs) in enumerate(list(truth_attrs.items())[:5]):
        print(f"{i+1}. {category}:")
        for attr_name, attr_value in attrs.items():
            print(f"   {attr_name}: {attr_value}")
        print()
    
    # 保存到文件
    save_truth_attrs(truth_attrs)
    
    # 额外保存详细统计信息
    save_truth_attrs(truth_attrs, "/llm_reco/dehua/data/food_data/truth_attrs_most_frequent.json")
    print("✅ 完成！基于最高频属性的 truth_attrs 已生成！")