import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from metric import get_metrics  # Assuming get_metrics is in code/metric.py
import os
from tqdm import tqdm
from metric_calculator import main as metric_main

def rank_voting(neighbors_1, neighbors_2=None, neighbors_3=None, alpha=2.0, k1=3):
    """
    Perform rank-based voting on one, two, or three lists of neighbors.

    :param neighbors_1: List of tuples (category, similarity) from the first model.
    :param neighbors_2: List of tuples (category, similarity) from the second model (optional).
    :param neighbors_3: List of tuples (category, similarity) from the third model (optional).
    :param alpha: Offset to avoid division by zero, default is 2.0.
    :return: The category with the highest total weight.
    """
    # Sort neighbors based on the similarity (descending order)
    # sorted_neighbors_1 = sorted(neighbors_1, key=lambda x: x[1], reverse=True)
    # Initialize a dictionary to accumulate weights for each category
    category_weights = {}

    # Function to accumulate weights from sorted neighbors
    def accumulate_weights(sorted_neighbors):
        for rank, (category, _) in enumerate(sorted_neighbors):
            if rank < k1:
                weight = 1
                category_weights[category] = category_weights.get(category, 0) + weight

    # Accumulate weights from the first model
    accumulate_weights(neighbors_1)

    # If a second set of neighbors is provided, accumulate weights from it as well
    if neighbors_2 is not None:
        sorted_neighbors_2 = sorted(neighbors_2, key=lambda x: x[1], reverse=True)
        accumulate_weights(sorted_neighbors_2)

    # If a third set of neighbors is provided, accumulate weights from it as well
    if neighbors_3 is not None:
        sorted_neighbors_3 = sorted(neighbors_3, key=lambda x: x[1], reverse=True)
        accumulate_weights(sorted_neighbors_3)

    # Find the category with the maximum total weight
    best_category = max(category_weights, key=category_weights.get)

    return best_category

def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def add_jsonl(file_path, data):
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')

def save_jsonl(filename, data):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main(input_file_1, input_file_2=None, input_file_3=None, output_file=None, gold_answer_file_path=None, k1=3):
    data_1 = read_jsonl(input_file_1)
    data_2 = read_jsonl(input_file_2) if input_file_2 else None
    data_3 = read_jsonl(input_file_3) if input_file_3 else None

    if data_2 and len(data_1) != len(data_2):
        raise ValueError("Input files 1 and 2 have different lengths.")
    if data_3 and len(data_1) != len(data_3):
        raise ValueError("Input files 1 and 3 have different lengths.")

    results = []

    for idx, item in tqdm(enumerate(data_1), total=len(data_1), desc="Processing items"):
        neighbors = list(zip(item['categories'], item['similarities']))
        neighbors_2 = list(zip(data_2[idx]['categories'], data_2[idx]['similarities'])) if data_2 else None
        neighbors_3 = list(zip(data_3[idx]['categories'], data_3[idx]['similarities'])) if data_3 else None

        result = rank_voting(neighbors, neighbors_2, neighbors_3, k1=k1)

        results.append({
            "question_id": item['question_id'],
            "image": item['image'],
            "text": result,
            "categories": "default"
        })
    save_jsonl(output_file, results)

    # Calculate metrics
    metric_main(output_file)

if __name__ == "__main__":
    os.environ['ROOT_DIR'] = "/map-vepfs/dehua/code/visual-memory"
    k1 = 3
    dataset_names = ["food101", "food172", "food2k", "fru92", "veg200", "foodx251"]
    for dataset_name in dataset_names:
        input_file_1 = f"/map-vepfs/dehua/code/visual-memory/questions/{dataset_name}/clip_test_5_mixedtrain_fewshot8.jsonl"
        input_file_1 = f"/map-vepfs/dehua/code/visual-memory/questions/{dataset_name}/dinov2_large_test_5_mixedtrain_fewshot8.jsonl"
        input_file_2 = f"/map-vepfs/dehua/code/visual-memory/questions/{dataset_name}/siglip_test_5_mixedtrain_fewshot8.jsonl"
        output_file = f"/map-vepfs/dehua/code/visual-memory/answers/rank_voting/{dataset_name}/{dataset_name}_dinov2_mixedtrain_fewshot8.jsonl"
        gold_answer_file_path = f"/map-vepfs/dehua/code/visual-memory/answers/groundtruth/{dataset_name}_answers.jsonl"

        # Call the main function with the specified parameters
        main(input_file_1=input_file_1, input_file_2=input_file_2, input_file_3=None, output_file=output_file, gold_answer_file_path=gold_answer_file_path, k1=k1)
