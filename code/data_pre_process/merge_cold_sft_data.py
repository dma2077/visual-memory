#!/usr/bin/env python3
import json
import os
import random

def merge_food_datasets():
    # 定义文件路径
    datasets = {
        "food101": "/llm_reco/dehua/data/food_finetune_data/food101_cold_sft.json",
        "food172": "/llm_reco/dehua/data/food_finetune_data/food172_cold_sft.json",
        "foodx251": "/llm_reco/dehua/data/food_finetune_data/foodx251_cold_sft.json",
        "food2k": "/llm_reco/dehua/data/food_finetune_data/food2k_cold_sft.json",
        "veg200": "/llm_reco/dehua/data/food_finetune_data/veg200_cold_sft.json",
        "fru92": "/llm_reco/dehua/data/food_finetune_data/fru92_cold_sft.json"
    }
    
    # 输出文件路径
    output_file = "/llm_reco/dehua/data/food_finetune_data/merged_food_datasets.json"
    
    merged_data = []
    
    for dataset_name, file_path in datasets.items():
        print(f"Processing {dataset_name}...")
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found, skipping...")
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if dataset_name == "food2k":
                random.shuffle(data)
                half_size = len(data) // 2
                data = data[:half_size]
                print(f"Taking 50% ({half_size} out of {len(data) * 2}) entries from {dataset_name} after random shuffle")
            
            # 添加数据集标识
            for item in data:
                item['dataset_source'] = dataset_name
                
            merged_data.extend(data)
            print(f"Added {len(data)} entries from {dataset_name}")
            
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {file_path}: {e}")
            continue
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    # 保存合并后的数据
    print(f"\nSaving merged data to {output_file}...")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)
        
        print(f"Successfully merged {len(merged_data)} total entries")
        print(f"Output saved to: {output_file}")
        
        # 统计每个数据集的数量
        print("\nDataset statistics:")
        dataset_counts = {}
        for item in merged_data:
            source = item.get('dataset_source', 'unknown')
            dataset_counts[source] = dataset_counts.get(source, 0) + 1
        
        for dataset, count in dataset_counts.items():
            print(f"  {dataset}: {count} entries")
            
    except Exception as e:
        print(f"Error saving merged data: {e}")

if __name__ == "__main__":
    merge_food_datasets()