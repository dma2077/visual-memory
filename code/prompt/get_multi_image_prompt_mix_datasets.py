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

def get_similarity_data(retrieval_matrix, similarity_matrix, query_image_paths, value_image_paths, value_categories, k_max):
    result_data = {}
    for i in range(len(query_image_paths)):
        retrieved_indices = retrieval_matrix[i, :k_max]
        retrieved_similarities = similarity_matrix[i, :k_max]
        image_path = query_image_paths[i]

        retrieved_data = [
            {
                'category': value_categories[idx],
                'path': value_image_paths[idx],
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

    save_jsonl(retrieval_save_path, conversation_list)

def main(args):
    model_name = args.model_name
    dataset_name = args.dataset_name
    k = args.k
    root_dir = args.root_dir
    few_shot = None if args.few_shot == "None" else args.few_shot
    args.few_shot = few_shot
    if model_name == "dinov2_large":
        embedding_root = '/map-vepfs/dehua/model/food_embeddings/dinov2'
    elif model_name == "siglip":
        embedding_root = '/map-vepfs/dehua/model/food_embeddings/siglip'
    elif model_name == "clip":
        embedding_root = '/map-vepfs/dehua/model/food_embeddings/clip'
    elif model_name == "clip-base":
        embedding_root = '/map-vepfs/dehua/model/food_embeddings/clip-base'
    model_name = 'clip' if model_name == 'clip-base' else model_name
    train_datasets = args.train_datasets.split(',')

    # Prepare lists to hold data from multiple training datasets
    train_embeddings_list = []
    train_image_paths = []
    train_categories = []
    train_indices_offset = 0


    for train_dataset_name in train_datasets:
        if args.few_shot is not None:
            food_data_path = f'{embedding_root}/{train_dataset_name}_data_fewshot{args.few_shot}.json'
        else:
            food_data_path = f'{embedding_root}/{train_dataset_name}_data.json'
        food_data = load_json(food_data_path)
        print(train_dataset_name)
        if args.few_shot is not None:
            index_train_path = f'{embedding_root}/{train_dataset_name}_train_{model_name}_fewshot{args.few_shot}.bin'
        else:
            index_train_path = f'{embedding_root}/{train_dataset_name}_train_{model_name}.bin'
        index_train = faiss.read_index(index_train_path)

        embeddings = index_train.reconstruct_n(0, index_train.ntotal)
        train_embeddings_list.append(embeddings)

        image_paths = food_data['image_path'][:food_data['train_size']]
        train_image_paths.extend(image_paths)
        Map_function = MapIdx2Category_172 if train_dataset_name in ['food172', 'food2k'] else MapIdx2Category
        for idx in tqdm(range(len(image_paths)), total=len(image_paths), desc=f"Processing {train_dataset_name}"):
            category = Map_function(idx, food_data, train_dataset_name)
            train_categories.append(category)

        train_indices_offset += index_train.ntotal
    # Concatenate embeddings
    train_embeddings = np.vstack(train_embeddings_list)

    # Build FAISS index
    index = faiss.IndexFlatL2(train_embeddings.shape[1])
    index.add(train_embeddings)

    # Load test data
    if args.few_shot is not None:
        food_data_test_path = f'{embedding_root}/{dataset_name}_data_fewshot{args.few_shot}.json'
    else:
        food_data_test_path = f'{embedding_root}/{dataset_name}_data.json'
    food_data_test = load_json(food_data_test_path)
    if args.few_shot is not None:
        index_test_path = f'{embedding_root}/{dataset_name}_test_{model_name}_fewshot{args.few_shot}.bin'
    else:
        index_test_path = f'{embedding_root}/{dataset_name}_test_{model_name}.bin'
    index_test = faiss.read_index(index_test_path)


    Map_function = MapIdx2Category_172 if dataset_name in ['food172', 'food2k'] else MapIdx2Category
    test_embeddings = index_test.reconstruct_n(0, index_test.ntotal)
    test_image_paths = food_data_test['image_path'][food_data_test['train_size']:]
    test_categories = [Map_function(idx + food_data_test['train_size'], food_data_test, dataset_name) for idx in range(food_data_test['test_size'])]

    # Perform retrieval
    max_k = k
    D_test, I_test = index.search(test_embeddings, max_k)

    # Get similarity data
    similarity_data_test = get_similarity_data(I_test, D_test, test_image_paths, train_image_paths, train_categories, k)
    # Save retrieval prompts
    if args.few_shot is not None:
        retrieval_test_path = os.path.join(root_dir, f'questions/{dataset_name}/{model_name}_test_{k}_mixedtrain_fewshot{args.few_shot}.jsonl')
    else:
        retrieval_test_path = os.path.join(root_dir, f'questions/{dataset_name}/{model_name}_test_{k}_mixedtrain.jsonl')

    save_retrieval_prompt(similarity_data_test, retrieval_test_path, test_categories, is_train=False)

    print(f"Test data saved to: {retrieval_test_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate multi-image prompts for food recognition.")
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--k', type=int, default=5, help='Number of similar images to use')
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory for saving output files')
    parser.add_argument('--few_shot', default=None, help='Number of few shot images to use')
    parser.add_argument('--train_datasets', type=str, required=True, help='Comma-separated list of training datasets')
    args = parser.parse_args()
    main(args)
