import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import faiss
from utils import load_jsonl, load_json, save_json, save_jsonl
import random

def MapIdx2Category(idx, food_data):
    image_paths = food_data["image_path"]
    image_path = image_paths[idx]
    category = image_path.split('/')[-2]
    return category

def MapIdx2Category_172(idx, food_data):
    label_file = '/mnt/madehua/fooddata/VireoFood172/SplitAndIngreLabel/FoodList.txt'
    with open(label_file, 'r') as file:
        lines = file.readlines()
    image_paths = food_data["image_path"]
    image_path = image_paths[idx]
    category_id = int(image_path.split('/')[-2])
    category = lines[category_id - 1].strip()
    return category



# Define the maximum k value to test
# max_k = 100

model_name = 'dinov2_large'
dataset_name = 'food101'

food_data_path = f'/mnt/madehua/food_embeddings/{dataset_name}_data.json'
index_train_path = f'/mnt/madehua/food_embeddings/{dataset_name}_train_{model_name}_cos.bin'
index_test_path = f'/mnt/madehua/food_embeddings/{dataset_name}_test_{model_name}_cos.bin'


k_values = [1, 5, 10, 20, 40, 100]

food_data = load_json(food_data_path)
train_num = food_data['train_size']
test_num = food_data['test_size']

index_train = faiss.read_index(index_train_path)
index_test = faiss.read_index(index_test_path)
print(index_train.ntotal)
print(index_test.ntotal)


Map_function = MapIdx2Category_172
if dataset_name == 'food172':
    Map_function = MapIdx2Category_172
elif dataset_name == 'food101':
    Map_function = MapIdx2Category
    
category_test = [Map_function(idx + train_num, food_data) for idx in range(test_num)]
category_train = [Map_function(idx, food_data) for idx in range(index_train.ntotal)]


category_list = list(set(category_train))

max_k=20
# Perform a single search for the maximum k value
D, I = index_train.search(index_test.reconstruct_n(0, index_test.ntotal), max_k)


def get_similarity_categories(retrieval_matrix, query_category, value_category, k_max):
    category_matrix = {}
    for i in range(len(query_category)):
        retrieved_indices = retrieval_matrix[i, :k_max]
        if len(query_category) == train_num:
            retrieved_indices = retrieval_matrix[i, 1: k_max + 1]
        retrieved_category = [value_category[idx] for idx in retrieved_indices]

        if len(query_category) == train_num:
            image_path = food_data['image_path'][i]
        else:
            image_path = food_data['image_path'][i + train_num]
        category_matrix[image_path] = retrieved_category
    return category_matrix
    
def get_similarities(similarity_matrix, query_category, value_category, k_max):
    retrieval_similatirity_matrix = {}
    for i in range(len(query_category)):
        retrieved_similarities = similarity_matrix[i, :k_max]
        if len(query_category) == train_num:
            retrieved_similarities = similarity_matrix[i, 1: k_max + 1]

        if len(query_category) == train_num:
            image_path = food_data['image_path'][i]
        else:
            image_path = food_data['image_path'][i + train_num]
        retrieval_similatirity_matrix[image_path] = retrieved_similarities
    return retrieval_similatirity_matrix
    
retrieval_k=5
max_k = retrieval_k + 1
D_train, I_train = index_train.search(index_train.reconstruct_n(0, index_train.ntotal), max_k)

D_test, I_test = index_train.search(index_test.reconstruct_n(0, index_test.ntotal), max_k)


category_matrix_train = get_similarity_categories(I_train, category_train, category_train, retrieval_k)
category_matrix_test = get_similarity_categories(I_test, category_test, category_train, retrieval_k)
similarity_matrix_train = get_similarities(D_train, category_train, category_train, retrieval_k)
similarity_matrix_test = get_similarities(D_test, category_test, category_train, retrieval_k)

food_recognition_prompts = [
    "What is the name of this dish?",
    "What is the name of the dish shown in the image",
    "Can you tell me the dish's name?",
    "What dish is this?",
    "Can you tell me the name of this dish?",
    "What is the culinary name of this dish?",
    "Can you provide the name of the dish?",
    "What is the category of the dish presented in the image?",
    "Can you identify the dish displayed in the photo?",
    "Which dish is depicted in the picture?"
]

# prompt_template = 'The categories and similarities of the k images most similar to this image are: {{ {category1}: {similarity1} }}, {{ {category2}: {similarity2} }}, {{ {category3}: {similarity3} }}, {{ {category4}: {similarity4} }} and {{ {category5}: {similarity5} }}. Based on the information above, please answer the following questions. What dish is this? Just provide its category.'

prompt_template = 'The categories of the {k} images most similar to this image are: {category1}, {category2}, {category3}, {category4}, {category5}. Based on the information above, please answer the following questions. What dish is this? Just provide its category.'

text_prompt_template = (
    "In this food recognition task, you are given similarity scores for the 5 most similar images to the current food image. "
    "Each image is associated with a category and a similarity score: "
    "{{ {category1}: {similarity1} }}, "
    "{{ {category2}: {similarity2} }}, "
    "{{ {category3}: {similarity3} }}, "
    "{{ {category4}: {similarity4} }}, "
    "and {{ {category5}: {similarity5} }}. "
    "Your task is to determine and provide only the category name of the current food image."
)

    
def save_test_retrieval_prompt(category_matrix, similarity_matrix, retrieval_save_path):
    conversation_list = []
    for idx, (k, v) in enumerate(category_matrix.items()):
        similarity_value = list(similarity_matrix.values())[idx]
        category_template = [v[0].replace('_', ' '), v[1].replace('_', ' '), v[2].replace('_', ' '), v[3].replace('_', ' '), v[4].replace('_', ' ')]
        similarity_template = [similarity_value[0], similarity_value[1], similarity_value[2], similarity_value[3], similarity_value[4]]
        conversation = {
        "question_id": str(idx),
        "image": k,
        "text": f"{prompt_template.format(category1=category_template[0], category2=category_template[1], category3=category_template[2],  category4=category_template[3], category5=category_template[4])}",
        #### similarity1=similarity_template[0], similarity2=similarity_template[1], similarity3=similarity_template[2], similarity4=similarity_template[3], similarity5=similarity_template[4]
        "category": "default"
        }
        conversation_list.append(conversation)
    save_jsonl(retrieval_save_path, conversation_list)

def save_train_text_retrieval_prompt(category_matrix, similarity_matrix, retrieval_save_path):
    conversation_list = []
    for idx, (k, v) in enumerate(category_matrix.items()):
        similarity_value = list(similarity_matrix.values())[idx]
        
        # 处理10个元素
        category_template = [v[i].replace('_', ' ') for i in range(retrieval_k)]
        similarity_template = [similarity_value[i] for i in range(retrieval_k)]
        
        # 动态生成instruction内容
        instruction_text = text_prompt_template.format(
            category1=category_template[0], similarity1=similarity_template[0],
            category2=category_template[1], similarity2=similarity_template[1],
            category3=category_template[2], similarity3=similarity_template[2],
            category4=category_template[3], similarity4=similarity_template[3],
            category5=category_template[4], similarity5=similarity_template[4],
            # category6=category_template[5], similarity6=similarity_template[5],
            # category7=category_template[6], similarity7=similarity_template[6],
            # category8=category_template[7], similarity8=similarity_template[7],
            # category9=category_template[8], similarity9=similarity_template[8],
            # category10=category_template[9], similarity10=similarity_template[9]
        )
        
        conversation = {
            "instruction": instruction_text,
            "input": "",
            "output": k.split('/')[-2].replace('_', ' ')
        }
        
        conversation_list.append(conversation)
    
    save_json(retrieval_save_path, conversation_list)

def save_test_text_retrieval_prompt(category_matrix, similarity_matrix, retrieval_save_path):
    conversation_list = []
    for idx, (k, v) in enumerate(category_matrix.items()):
        similarity_value = list(similarity_matrix.values())[idx]
        
        # 处理10个元素
        category_template = [v[i].replace('_', ' ') for i in range(retrieval_k)]
        similarity_template = [similarity_value[i] for i in range(retrieval_k)]
        
        # 动态生成instruction内容
        instruction_text = text_prompt_template.format(
            category1=category_template[0], similarity1=similarity_template[0],
            category2=category_template[1], similarity2=similarity_template[1],
            category3=category_template[2], similarity3=similarity_template[2],
            category4=category_template[3], similarity4=similarity_template[3],
            category5=category_template[4], similarity5=similarity_template[4],
            # category6=category_template[5], similarity6=similarity_template[5],
            # category7=category_template[6], similarity7=similarity_template[6],
            # category8=category_template[7], similarity8=similarity_template[7],
            # category9=category_template[8], similarity9=similarity_template[8],
            # category10=category_template[9], similarity10=similarity_template[9]
        )
        
        conversation = {
        "question_id": str(idx),
        "image": k,
        "text": instruction_text,
        "category": "default"
        }
        
        conversation_list.append(conversation)
    
    save_jsonl(retrieval_save_path, conversation_list)


    
retrieval_trian_path = f'/mnt/madehua/fooddata/json_file/alpaca/{dataset_name}_train_{max_k-1}.json'
retrieval_test_path = f'/mnt/madehua/fooddata/json_file/alpaca/{dataset_name}_test_{max_k-1}.jsonl'
#retrieval_test_path = '/mnt/madehua/fooddata/json_file/172_test_retrieval_category_and_similarity.jsonl'
#save_train_retrieval_prompt(category_matrix_train, similarity_matrix_train, retrieval_trian_path)
save_train_text_retrieval_prompt(category_matrix_train, similarity_matrix_train, retrieval_trian_path)
save_test_text_retrieval_prompt(category_matrix_test, similarity_matrix_train, retrieval_test_path)
print(retrieval_trian_path)
