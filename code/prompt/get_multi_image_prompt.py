import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import faiss
import random
import argparse
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_jsonl, load_json, save_json, save_jsonl

def MapIdx2Category(idx, food_data, dataset_name):
    image_paths = food_data["image_path"]
    image_path = image_paths[idx]
    category = image_path.split('/')[-2]
    return category

def MapIdx2Category_172(idx, food_data, dataset_name):
    if dataset_name == 'food172':
        label_file = '/map-vepfs/dehua/data/data/VireoFood172/SplitAndIngreLabel/FoodList.txt'
        with open(label_file, 'r') as file:
            lines = file.readlines()
        image_paths = food_data["image_path"]
        image_path = image_paths[idx]
        category_id = int(image_path.split('/')[-2])
        if '--' in lines[category_id - 1]:
            category = lines[category_id - 1].strip().split('--')[1]
        else:
            category = lines[category_id - 1].strip()
        return category
    else:
        label_file = '/map-vepfs/dehua/data/data/Food2k_complete_jpg/food2k_label2name_en.txt'
        with open(label_file, 'r') as file:
            lines = file.readlines()
        image_paths = food_data["image_path"]
        image_path = image_paths[idx]
        category_id = int(image_path.split('/')[-2])
        if '--' in lines[category_id]:
            category = lines[category_id].strip().split('--')[1]
        return category

def get_similarity_data(retrieval_matrix, similarity_matrix, query_category, value_category, k_max, food_data, food_data_few_shot ,train_num):
    result_data = {}
    is_train = len(query_category) == train_num
    
    for i in range(len(query_category)):
        if is_train:
            retrieved_indices = retrieval_matrix[i, 1:k_max+1]
            retrieved_similarities = similarity_matrix[i, 1:k_max+1]
            image_path = food_data['image_path'][i]
        else:
            retrieved_indices = retrieval_matrix[i, :k_max]
            retrieved_similarities = similarity_matrix[i, :k_max]
            image_path = food_data['image_path'][i + train_num]
        
        retrieved_data = [
            {
                'category': value_category[idx],
                'path': food_data_few_shot['image_path'][idx],
                'similarity': round(float(sim), 4)
            }
            for idx, sim in zip(retrieved_indices, retrieved_similarities)
        ]
        
        result_data[image_path] = retrieved_data
    
    return result_data

def save_retrieval_prompt(similarity_data, retrieval_save_path, category_set, is_train=True):
    conversation_list = []
    for idx, (target_image, similar_images) in enumerate(similarity_data.items()):
        prompt = "The categories of the 5 images most similar to this image are: "
        categories = [img['category'].replace('_', ' ') for img in similar_images]
        prompt += ", ".join(categories)
        prompt += ". Based on the information above, please answer the following questions. What dish is this? Just provide its category."

        if is_train:
            conversation = {
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image> {prompt}"
                    },
                    {
                        "from": "gpt",
                        "value": category_set[idx].replace('_', ' ')
                    }
                ],
                "images": [target_image] + [img['path'] for img in similar_images],
                "similarities": [img['similarity'] for img in similar_images]
            }
        else:
            conversation = {
                "question_id": str(idx),
                "image": target_image,
                "categories": [img['category'].replace('_', ' ') for img in similar_images],
                "similarities": [img['similarity'] for img in similar_images],
                "retrieval_images": [img['path'] for img in similar_images],
                "text": prompt,
                "category": "default"
            }
        conversation_list.append(conversation)
    
    if is_train:
        save_json(retrieval_save_path, conversation_list)
    else:
        save_jsonl(retrieval_save_path, conversation_list)




def main(args):
    model_name = args.model_name
    dataset_name = args.dataset_name
    k = args.k
    root_dir = args.root_dir
    few_shot = None if args.few_shot == "None" else args.few_shot
    args.few_shot = few_shot
    if model_name == "dinov2_large":
        #embedding_root = '/map-vepfs/dehua/model/food_embeddings/six_dataset_all'
        embedding_root = '/map-vepfs/dehua/model/food_embeddings/dinov2'
    elif model_name == "siglip":
        embedding_root = '/map-vepfs/dehua/model/food_embeddings/siglip'
    elif model_name == "clip":
        embedding_root = '/map-vepfs/dehua/model/food_embeddings/clip'
    elif model_name == "clip-224":
        embedding_root = '/map-vepfs/dehua/model/food_embeddings/clip-224'
        model_name = "clip"
    elif model_name == "clip-base":
        embedding_root = '/map-vepfs/dehua/model/food_embeddings/clip-base'
        model_name = "clip"
    # embedding_root = '/mnt/madehua/food_embeddings/'

    if args.few_shot is not None:
        food_data_path = f'{embedding_root}/{dataset_name}_data.json'
        food_data_path_few_shot = f'{embedding_root}/{dataset_name}_data_fewshot{args.few_shot}.json'
        index_train_path = f'{embedding_root}/{dataset_name}_train_{model_name}.bin'
        index_test_path = f'{embedding_root}/{dataset_name}_test_{model_name}.bin'
        index_train_path_few_shot = f'{embedding_root}/{dataset_name}_train_{model_name}_fewshot{args.few_shot}.bin'

        
    else:
        food_data_path = f'{embedding_root}/{dataset_name}_data.json'
        index_train_path = f'{embedding_root}/{dataset_name}_train_{model_name}.bin'
        index_test_path = f'{embedding_root}/{dataset_name}_test_{model_name}.bin'
        food_data_path_few_shot = food_data_path
        index_train_path_few_shot = index_train_path


    food_data = load_json(food_data_path)
    food_data_few_shot = load_json(food_data_path_few_shot)
    train_num = food_data['train_size']
    test_num = food_data['test_size']

    index_train_fewshot = faiss.read_index(index_train_path_few_shot)
    index_train = faiss.read_index(index_train_path)
    index_test = faiss.read_index(index_test_path)

    Map_function = MapIdx2Category_172 if dataset_name == 'food172' or dataset_name == 'food2k' else MapIdx2Category
    
    category_test = [Map_function(idx + train_num, food_data, dataset_name) for idx in range(test_num)]
    category_train_value = [Map_function(idx, food_data_few_shot, dataset_name) for idx in range(index_train_fewshot.ntotal)]
    category_train_query = [Map_function(idx, food_data, dataset_name) for idx in range(index_train.ntotal)]

    max_k = k + 1
    # D_train, I_train = index_train_fewshot.search(index_train.reconstruct_n(0, index_train.ntotal), max_k)
    D_test, I_test = index_train_fewshot.search(index_test.reconstruct_n(0, index_test.ntotal), max_k)

    # similarity_data_train = get_similarity_data(I_train, D_train, category_train_query, category_train_value, k, food_data, food_data_few_shot, train_num)
    similarity_data_test = get_similarity_data(I_test, D_test, category_test, category_train_value, k, food_data, food_data_few_shot, train_num)


    model_name = 'clip-base' if model_name == 'clip-base' else model_name
    if args.few_shot != None:
        retrieval_train_path = os.path.join(root_dir, f'questions/{dataset_name}/{model_name}_train_{k}_fewshot{args.few_shot}_old.json')
        retrieval_test_path = os.path.join(root_dir, f'questions/{dataset_name}/{model_name}_test_{k}_fewshot{args.few_shot}_old.jsonl')
    else:
        retrieval_train_path = os.path.join(root_dir, f'questions/{dataset_name}/{model_name}_train_{k}_softmax_old.json')
        retrieval_test_path = os.path.join(root_dir, f'questions/{dataset_name}/{model_name}_test_{k}_softmax_old.jsonl')

    # save_retrieval_prompt(similarity_data_train, retrieval_train_path, category_train_query, is_train=True)
    save_retrieval_prompt(similarity_data_test, retrieval_test_path, category_test, is_train=False)

    print(f"Train data saved to: {retrieval_train_path}")
    print(f"Test data saved to: {retrieval_test_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate multi-image prompts for food recognition.")
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--k', type=int, default=5, help='Number of similar images to use')
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory for saving output files')
    parser.add_argument('--few_shot', default=None, help='Number of few shot images to use')
    args = parser.parse_args()
    main(args)
    