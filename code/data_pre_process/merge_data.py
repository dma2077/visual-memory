#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random
from pathlib import Path

def merge_and_shuffle_datasets(result_paths, output_path):
    """
    Merge multiple JSON files and shuffle the combined data.
    
    Args:
        result_paths (dict): Dictionary mapping dataset names to their JSON file paths
        output_path (str): Path to save the merged and shuffled JSON file
    """
    # List to store all data
    all_data = []
    
    # Read and combine data from all files
    for dataset_name, file_path in result_paths.items():
        print(f"Reading {dataset_name} from {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.extend(data)
                print(f"Added {len(data)} entries from {dataset_name}")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    # Shuffle the combined data
    random.shuffle(all_data)
    
    # Save the merged and shuffled data
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nTotal entries in merged dataset: {len(all_data)}")
    print(f"Merged and shuffled data saved to: {output_path}")

if __name__ == "__main__":
    # Define the paths
    result_paths = {
        "food101": "/llm_reco/dehua/data/food_finetune_data/food101_cold_sft.json",
        "food172": "/llm_reco/dehua/data/food_finetune_data/food172_cold_sft.json",
        "foodx251": "/llm_reco/dehua/data/food_finetune_data/foodx251_cold_sft.json",
        "veg200": "/llm_reco/dehua/data/food_finetune_data/veg200_cold_sft.json",
        "fru92": "/llm_reco/dehua/data/food_finetune_data/fru92_cold_sft.json"
    }
    
    # Output path for the merged file
    output_path = "/llm_reco/dehua/data/food_finetune_data/cold_sft_all.json"
    
    # Merge and shuffle the datasets
    merge_and_shuffle_datasets(result_paths, output_path)