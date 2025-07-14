import os

# def find_files_in_subdirectories(file_names, root_dir):
#     found_files = []
    
#     # 遍历当前目录及其子目录
#     for dirpath, dirnames, filenames in os.walk(root_dir):
#         for file_name in file_names:
#             if file_name in filenames:
#                 # 如果找到文件，构建完整路径并添加到列表
#                 full_path = os.path.join(dirpath, file_name)
#                 full_path = full_path.replace(root_dir + '/', '')
#                 found_files.append(full_path)
    
#     return found_files

# def save_found_files(found_files, output_file):
#     with open(output_file, 'w') as f:
#         for file_path in found_files:
#             f.write(file_path + '\n')

# def main():
#     # 指定包含所有文件名的文件和根目录
#     input_file = '/map-vepfs/dehua/data/data/vfn_1_0/Meta/training.txt'  # 包含文件名的文件
#     root_dir = '/map-vepfs/dehua/data/data/vfn_1_0/Images'  # 当前目录
#     output_file = '/map-vepfs/dehua/data/data/vfn_1_0/Meta/training_new.txt'  # 输出文件名

#     # 读取文件名列表
#     with open(input_file, 'r') as f:
#         file_names = [line.strip() for line in f.readlines()]

#     found_files = find_files_in_subdirectories(file_names, root_dir)
#     save_found_files(found_files, output_file)

#     print(f"Found {len(found_files)} files. Paths saved to {output_file}.")

import json
# category_frequency_dict = {}

# with open('/map-vepfs/dehua/data/data/vfn_1_0/Meta/category.txt', 'r') as f:
#     category_list = f.readlines()
#     category_list = [line.strip().replace(' ', '_').lower() for line in category_list]
# with open('/map-vepfs/dehua/data/data/vfn_1_0/Meta/frequency.txt', 'r') as f:
#     frequency_list = f.readlines()
#     frequency_list = [int(line.strip()) for line in frequency_list]
# category_frequency_dict = dict(zip(category_list, frequency_list))

# with open('/map-vepfs/dehua/data/data/vfn_1_0/Meta/category_frequency.json', 'w') as f:
#     json.dump(category_frequency_dict, f)
# print(category_frequency_dict)

# def load_category_ids(category_file):
#     category_ids = {}
#     with open(category_file, 'r') as f:
#         for line in f:
#             parts = line.strip().split(' ', 1)  # 只分割一次，获取 ID 和类别名称
#             if len(parts) == 2:
#                 category_id, category_name = parts
#                 category_ids[category_name] = category_id
#     return category_ids

# category_ids = load_category_ids('/map-vepfs/dehua/data/data/vfn_1_0/Meta/category_ids.txt')

# print(category_ids)
# id_category_frequency_dict = {}
# matched_count = 0  # 初始化匹配计数器
# for category, frequency in category_frequency_dict.items():
#     if category in category_ids:  # 检查类别是否在 category_ids 中
#         item = {}
#         item["id"] = category_ids[category]
#         item["frequency"] = frequency
#         item["category"] = category
#         id_category_frequency_dict[category_ids[category]] = item
#         matched_count += 1  # 增加匹配计数

# print(f"Matched {matched_count} categories.")
# with open('/map-vepfs/dehua/data/data/vfn_1_0/Meta/id_category_frequency.json', 'w') as f:
#     json.dump(id_category_frequency_dict, f)
    

import json

# 读取类别频率字典
category_frequency_dict = {}
with open('/map-vepfs/dehua/data/data/vfn_1_0/Meta/category.txt', 'r') as f:
    category_list = f.readlines()
    category_list = [line.strip().replace(' ', '_').lower() for line in category_list]
with open('/map-vepfs/dehua/data/data/vfn_1_0/Meta/frequency.txt', 'r') as f:
    frequency_list = f.readlines()
    frequency_list = [int(line.strip()) for line in frequency_list]
category_frequency_dict = dict(zip(category_list, frequency_list))

# 读取类别 ID 字典
def load_category_ids(category_file):
    category_ids = {}
    with open(category_file, 'r') as f:
        for line in f:
            parts = line.strip().split(' ', 1)  # 只分割一次，获取 ID 和类别名称
            if len(parts) == 2:
                category_id, category_name = parts
                category_ids[category_name] = category_id
    return category_ids

category_ids = load_category_ids('/map-vepfs/dehua/data/data/vfn_1_0/Meta/category_ids.txt')

# 找出不同的键
sorted_category_frequency_keys = sorted(category_frequency_dict.keys())
sorted_category_ids_keys = sorted(category_ids.keys())

# 找出在 category_frequency_dict 中但不在 category_ids 中的键
diff_keys_in_frequency = set(sorted_category_frequency_keys) - set(sorted_category_ids_keys)
# 找出在 category_ids 中但不在 category_frequency_dict 中的键
diff_keys_in_ids = set(sorted_category_ids_keys) - set(sorted_category_frequency_keys)

# 输出不同的键
print("Keys in category_frequency_dict but not in category_ids:")
for key in sorted(diff_keys_in_frequency):
    print(key)

print("\nKeys in category_ids but not in category_frequency_dict:")
for key in sorted(diff_keys_in_ids):
    print(key)