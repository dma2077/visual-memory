import numpy as np
import random
import os
# # Parameters
# N = 101      # Total number of categories
# n_max = 750  # Maximum number of samples per category
# n_min = 5    # Minimum number of samples per category
# alpha = 6    # Power-law distribution exponent

# # Calculate the power-law exponent beta
# beta = np.log(n_max / n_min) / np.log(N)

# # Generate category indices (from 1 to N)
# class_indices = np.arange(1, N + 1)

# # Calculate the number of samples for each category
# n_i = n_max * (class_indices ** (-beta))
# n_i = np.round(n_i).astype(int)  # Round sample counts to integers
# n_i = np.clip(n_i, n_min, n_max)  # Ensure sample counts are within [n_min, n_max]

# # Write the sample counts to a file
# output_file_path = '/map-vepfs/dehua/data/data/food-101/meta/train_lt_number.txt'
# with open(output_file_path, 'w') as file:
#     for idx, samples in zip(class_indices, n_i):
#         file.write(f"{idx} {samples}\n")

# 读取类别编号对应的采样数量
# category_id_number = {}
# with open('/map-vepfs/dehua/data/data/food-101/meta/train_lt_number.txt', 'r') as file:
#     lines = file.readlines()
#     for line in lines:
#         idx, samples = line.strip().split()
#         category_id_number[int(idx)] = int(samples)  # idx 为类别编号，samples 为采样数量

# # 读取训练集文件
# with open("/map-vepfs/dehua/data/data/food-101/meta/train.txt", "r") as f:
#     data = f.readlines()

# # 按类别存储样本
# category_samples = {}
# category_number = {}
# added_category = []
# category_id = 1

# for line in data:
#     category, sample_id = line.strip().split("/")
#     if category not in category_samples:
#         category_samples[category] = []
        
#     # 将类别名称映射到编号，并记录采样数量
#     if category not in added_category:
#         if category_id in category_id_number:
#             category_number[category] = category_id_number[category_id]
#         added_category.append(category)
#         category_id += 1

#     category_samples[category].append(line.strip())

# # 进行采样
# sampled_data = []
# for category, samples in category_samples.items():
#     sample_count = category_number.get(category, len(samples))  # 获取该类别的采样数量，如果没有指定则保留所有样本
#     sampled_data.extend(random.sample(samples, min(sample_count, len(samples))))  # 随机采样

# # 将采样结果保存到新文件
# output_file_path = "/map-vepfs/dehua/data/data/food-101/meta/train_lt.txt"
# with open(output_file_path, "w") as f:
#     for item in sampled_data:
#         f.write(f"{item}\n")

# print(f"采样完成，采样数据已保存到 {output_file_path}")

import os

# 指定目录和文件路径
directory_path = "/map-vepfs/dehua/data/data/vegfru-dataset/veg200_images"
file_path = "/map-vepfs/dehua/data/data/vegfru-dataset/veg200_lists/veg_subclasses.txt"

def get_subdirectories(path):
    # 获取指定目录下的所有子目录名称
    subdirectories = {name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))}
    return subdirectories

def get_file_classes(file_path):
    # 从文件中读取每一行并去掉换行符，存为列表
    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file]
    # 创建集合来移除重复项
    classes_set = set(lines)
    return lines, classes_set

# 获取子目录名称集合和文件中的类名集合
subdirectories = get_subdirectories(directory_path)
file_lines, file_classes = get_file_classes(file_path)

# 输出文件的原始行数和集合的大小
print(f"Original number of lines in file: {len(file_lines)}")
print(f"Unique classes in file (after removing duplicates): {len(file_classes)}")

# 检查是否有重复项
if len(file_lines) != len(file_classes):
    print("There are duplicates in the file.")

# 输出子目录集合的大小
print(f"Number of subdirectories: {len(subdirectories)}")

# 计算交集
intersection = subdirectories & file_classes

# # 输出交集大小和重合部分
# print(f"\nNumber of common items: {len(intersection)}")
# print("Common items:")
# for name in intersection:
#     print(name)

# 计算差集
only_in_subdirectories = subdirectories - file_classes
only_in_file_classes = file_classes - subdirectories

# 输出差集
print("\nOnly in subdirectories:")
for name in only_in_subdirectories:
    print(name)

print("\nOnly in file classes:")
for name in only_in_file_classes:
    print(name)



        