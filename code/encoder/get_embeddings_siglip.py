import os
import argparse
import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import faiss
from code.utils import *
from code.metric import get_metrics
import torch.nn.functional as F
from collections import defaultdict
import random
from torch import nn
import numpy


# Define the maximum k value to test
max_k = 100
k_values = [1, 5, 10, 20, 40, 100]



# 自定义Dataset类
class FoodDataset(Dataset):
    def __init__(self, file_path, root_dir, label, processor, few_shot=None, seed=None):
        self.root_dir = root_dir
        self.processor = processor
        self.label = label
        self.image_paths, self.categories = self.load_image_paths(file_path, few_shot, seed)
        
    def load_image_paths(self, file_path, few_shot, seed):
        if seed is not None:
            random.seed(seed)
        
        image_paths = []
        categories = []
        category_images = defaultdict(list)
        
        with open(file_path, 'r') as file:
            for line in file:
                relative_path = line.strip()
                if ',' in relative_path:
                    relative_path = relative_path.split(',')[0]
                else:
                    relative_path = relative_path.split(' ')[0]
                
                category = relative_path.split('/')[-2]
                if relative_path.startswith('/'):
                    relative_path = relative_path[1:]
                full_path = os.path.join(self.root_dir, relative_path)
                if not full_path.lower().endswith(('.jpg', '.jpeg')):
                    full_path += '.jpg'
                
                category_images[category].append(full_path)
        
        for category, paths in category_images.items():
            if few_shot is not None:
                selected_paths = random.sample(paths, min(few_shot, len(paths)))
            else:
                selected_paths = paths
            
            image_paths.extend(selected_paths)
            categories.extend([category] * len(selected_paths))

        combined = list(zip(image_paths, categories))
        image_paths, categories = zip(*combined)
        
        return list(image_paths), list(categories)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt")
            inputs['label'] = self.label
            inputs['image_path'] = image_path
            inputs['category'] = self.categories[idx]
            return inputs
        except Exception as e:
            print(f"Error processing image {self.image_paths[idx]}: {e}")
            return self.__getitem__((idx + 1) % len(self))

def MapIdx2Category(idx, food_data):
    image_paths = food_data["image_path"]
    image_path = image_paths[idx]
    category = image_path.split('/')[-2]
    return category

def calculate_accuracy_at_k(I, category_test, category_train, k_values):
    accuracy_at_k = {}
    for k in k_values:
        correct_counts = []
        for test_idx in tqdm(range(len(I)), total=len(I), desc=f"Calculating accuracy at k={k}"):
            retrieved_indices = I[test_idx, :k]
            test_category = category_test[test_idx]
            retrieved_categories = [category_train[idx] for idx in retrieved_indices]
            correct_count = sum(1 for category in retrieved_categories if category == test_category)
            correct_counts.append(correct_count / k)
        accuracy_at_k[k] = np.mean(correct_counts)
    return accuracy_at_k

def get_dataset_paths(dataset_name):
    if dataset_name == 'food2k':
        root_dir = '/map-vepfs/dehua/data/data/Food2k_complete_jpg/'
        train_file = '/map-vepfs/dehua/data/data/Food2k_complete_jpg/train.txt'
        test_file = '/map-vepfs/dehua/data/data/Food2k_complete_jpg/test.txt'
    elif dataset_name == 'food101':
        root_dir = '/map-vepfs/dehua/data/data/food-101/images/'
        train_file = '/map-vepfs/dehua/data/data/food-101/meta/train.txt'
        test_file = '/map-vepfs/dehua/data/data/food-101/meta/test.txt'
    elif dataset_name == 'food101-lt':
        root_dir = '/map-vepfs/dehua/data/data/food-101/images/'
        train_file = '/map-vepfs/dehua/data/data/food-101/meta/train_lt.txt'
        test_file = '/map-vepfs/dehua/data/data/food-101/meta/test.txt'
    elif dataset_name == 'food172':
        root_dir = '/map-vepfs/dehua/data/data//VireoFood172/ready_chinese_food/'
        train_file = '/map-vepfs/dehua/data/data/VireoFood172/SplitAndIngreLabel/TR.txt'
        test_file = '/map-vepfs/dehua/data/data/VireoFood172/SplitAndIngreLabel/TE.txt'
    elif dataset_name == 'fru92':
        root_dir = '/map-vepfs/dehua/data/data/vegfru-dataset/fru92_images/'
        train_file = '/map-vepfs/dehua/data/data/vegfru-dataset/fru92_lists/fru_train.txt'
        test_file = '/map-vepfs/dehua/data/data/vegfru-dataset/fru92_lists/fru_test.txt'
    elif dataset_name == 'veg200':
        root_dir = '/map-vepfs/dehua/data/data/vegfru-dataset/veg200_images/'
        train_file = '/map-vepfs/dehua/data/data/vegfru-dataset/veg200_lists/veg_train.txt'
        test_file = '/map-vepfs/dehua/data/data/vegfru-dataset/veg200_lists/veg_test.txt'
    elif dataset_name == 'foodx251':
        root_dir = '/map-vepfs/dehua/data/data/FoodX-251/images/'
        train_file = '/map-vepfs/dehua/data/data/FoodX-251/annot/train.txt'
        test_file = '/map-vepfs/dehua/data/data/FoodX-251/annot/val.txt'
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")
    return root_dir, train_file, test_file


def gem_pooling(x, p=3):
    # 使用 numpy.clip 代替 .clip()，并使用 numpy.power 代替 .power()
    x = np.clip(x, 1e-6, None)
    x = np.power(x, p)
    # 使用 numpy.mean 并设置 keepdims=True 来保持维度
    mean_val = np.mean(x, axis=1, keepdims=True)
    return np.power(mean_val, 1.0 / p).squeeze(1)


def main():
    args = parse_args()
    root_dir, train_file, test_file = get_dataset_paths(args.dataset_name)

    processor = AutoImageProcessor.from_pretrained(args.model_path)

    train_dataset = FoodDataset(train_file, root_dir, "train", processor, few_shot=args.few_shot, seed=args.seed)
    test_dataset = FoodDataset(test_file, root_dir, "test", processor)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = AutoModel.from_pretrained(args.model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    data_dict = {
        'image_path': [],
        'category': [],
        'train_size': len(train_dataset),
        'test_size': len(test_dataset),
    }

    # 获取模型最后一层的特征维度
    d=1152
    index_ip_train = faiss.IndexFlatIP(d)
    index_ip_test = faiss.IndexFlatIP(d)

    # FAISS保存路径
    if args.few_shot:
        train_index_path = f'{args.output_dir}/{args.dataset_name}_train_{args.model_name}_fewshot{args.few_shot}.bin'
        test_index_path = f'{args.output_dir}/{args.dataset_name}_test_{args.model_name}_fewshot{args.few_shot}.bin'
    else:
        train_index_path = f'{args.output_dir}/{args.dataset_name}_train_{args.model_name}.bin'
        test_index_path = f'{args.output_dir}/{args.dataset_name}_test_{args.model_name}.bin'

    def process_and_index(dataloader, index_ip, split_name, index_path):
        if os.path.exists(index_path):
            for batch in tqdm(dataloader, desc=f"Processing {split_name} images"):
                batch_paths = batch['image_path']
                batch_categories = batch['category']
                data_dict['image_path'].extend(batch_paths)
                data_dict['category'].extend(batch_categories)
            print(f"Skipping {split_name} as index already exists at {index_path}.")
            return faiss.read_index(index_path)
        else:
            for batch in tqdm(dataloader, desc=f"Processing {split_name} images"):
                batch_paths = batch['image_path']
                batch_categories = batch['category']
                batch_pixel_values = batch['pixel_values'].to(device).squeeze(1)
                #outputs = image_encoder(pixel_values=batch_pixel_values)
                # outputs = image_encoder(batch_pixel_values)
                # embeddings = gem_pooling(outputs.last_hidden_state.detach().cpu().numpy(), p=3)
                outputs = model.get_image_features(batch_pixel_values)
                embeddings = outputs.detach().cpu().numpy()
                norms = np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)  # 调整为 (n, d)
                embeddings = embeddings / norms
                index_ip.add(embeddings)
                data_dict['image_path'].extend(batch_paths)
                data_dict['category'].extend(batch_categories)

            faiss.write_index(index_ip, index_path)
            print(f"FAISS index saved to {index_path}.")
            return index_ip


    # 保存数据字典
    if args.few_shot:
        data_save_path = f'{args.output_dir}/{args.dataset_name}_data_fewshot{args.few_shot}.json'
    else:
        data_save_path = f'{args.output_dir}/{args.dataset_name}_data.json'
    # 处理训练集和测试集
    if not os.path.exists(train_index_path) or not os.path.exists(test_index_path) :
        index_train = process_and_index(train_dataloader, index_ip_train, "train", train_index_path)
        index_test = process_and_index(test_dataloader, index_ip_test, "test", test_index_path)
        save_json(data_save_path, data_dict)
    else:
        with open(data_save_path, 'r') as file:
            data_dict = json.load(file)
        index_train=faiss.read_index(train_index_path)
        index_test=faiss.read_index(test_index_path)
    # del model
    # index_train = process_and_index(train_dataloader, index_ip_train, "train", train_index_path)
    # index_test = process_and_index(test_dataloader, index_ip_test, "test", test_index_path)
    # save_json(data_save_path, data_dict)
    # 计算检索准确率
    category_test = data_dict['category'][data_dict['train_size']:]
    category_train = data_dict['category'][:data_dict['train_size']]

    D, I = index_train.search(index_test.reconstruct_n(0, index_test.ntotal), max_k)
    accuracies = calculate_accuracy_at_k(I, category_test, category_train, k_values)

    for k, acc in accuracies.items():
        print(f"Accuracy at {k}: {acc:.4f}")




import argparse

def none_or_int(value):
    if value.lower() == 'none':
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid value for --few_shot: {value}. Must be an integer or 'None'.")

def parse_args():
    parser = argparse.ArgumentParser(description="Food Embedding Extraction")
    parser.add_argument('--dataset_name', type=str, required=True, help="Dataset name, e.g., food101, food2k, etc.")
    parser.add_argument('--model_name', type=str, required=True, help="Model name, e.g., dinov2_large")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the pre-trained model")
    parser.add_argument('--distance_method', type=str, default="l2", help="Distance method for FAISS index")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for DataLoader")
    parser.add_argument('--num_workers', type=int, default=64, help="Number of workers for DataLoader")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save FAISS index and JSON")
    parser.add_argument('--few_shot', type=none_or_int, default=5, help="Number of images per class for few-shot learning or 'None'")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


if __name__ == '__main__':
    main()