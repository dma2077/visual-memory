import json
import os
import re
from nltk.metrics.distance import edit_distance
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from code.utils import load_jsonl


NUMBER = 0

def get_dataset_name(prediction_file):
    return prediction_file.split("/")[-2]


def get_label_file(dataset_name):
    label_files = {
        "food2k": "/llm_reco/dehua/data/food_data/Food2k_complete_jpg/food2k_label2name_en.txt",
        "food101": "/llm_reco/dehua/data/food_data/food-101/meta/classes.txt",
        "food172": "/llm_reco/dehua/data/food_data/VireoFood172/SplitAndIngreLabel/FoodList.txt",
        "fru92": "/llm_reco/dehua/data/food_data/vegfru-dataset/fru92_lists/fru_subclasses.txt",
        "veg200": "/llm_reco/dehua/data/food_data/vegfru-dataset/veg200_lists/veg_subclasses.txt",
        "food251": "/llm_reco/dehua/data/food_data/FoodX-251/annot/class_list.txt",
        "foodx251": "/llm_reco/dehua/data/food_data/FoodX-251/annot/class_list.txt",
    }
    return label_files.get(dataset_name)


def load_data(prediction_file):
    return load_jsonl(prediction_file)


def build_id2category_dict(label_file, dataset_name):
    id2category_dict = {}
    with open(label_file, "r") as file:
        for idx, line in enumerate(file):
            line = line.strip()
            if dataset_name == "food2k":
                category = line.split("--")[1]
            elif dataset_name in ["food251", "foodx251"]:
                category = line.split(" ")[1].replace("_", " ")
            elif dataset_name == "food172":
                category = line
            elif dataset_name in ["fru92", "food101", "veg200"]:
                category = line.replace("_", " ")
            id2category_dict[idx] = category.lower()
    return id2category_dict


def find_most_similar_word(target_word, word_list):
    min_distance = float("inf")
    most_similar_word = None
    for word in word_list:
        distance = edit_distance(target_word, word)
        if distance < min_distance:
            min_distance = distance
            most_similar_word = word
    return most_similar_word, min_distance


def process_data(data, id2category_dict, dataset_name):
    text = data["text"].lower()
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)

    if match:
        prediction = match.group(1).strip()
    else:
        NUMBER += 1
        print(NUMBER)
        return None  # 不匹配时返回 None

    try:
        if dataset_name == "food2k":
            truth_category_id = data["image"].split("/")[-2]
            truth_category = id2category_dict[int(truth_category_id)]
        elif dataset_name == "food172":
            truth_category_id = data["image"].split("/")[-2]
            truth_category = id2category_dict[int(truth_category_id) - 1]
        else:
            truth_category = data["image"].split("/")[-2].replace("_", " ")

        truth_category = truth_category.lower()
        predicted_category = prediction.lower()

        if truth_category == predicted_category:
            return (1, 1, None)  # 正确匹配
        else:
            most_similar_category, _ = find_most_similar_word(
                predicted_category, list(id2category_dict.values())
            )
            if most_similar_category == truth_category:
                return (1, 1, None)  # 近似匹配成功
            else:
                # 不匹配时返回预测和真实类别
                return (0, 1, {"prediction": predicted_category, "truth": truth_category})

    except Exception as e:
        print(f"Error processing data: {data}, Error: {e}")
        return (0, 1, {"prediction": "error", "truth": str(e)})


def main(prediction_file, output_mismatches_file):
    dataset_name = get_dataset_name(prediction_file)
    label_file = get_label_file(dataset_name)
    datas = load_data(prediction_file)
    id2category_dict = build_id2category_dict(label_file, dataset_name)

    true_count, total_count = 0, 0
    mismatches = []

    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(
            tqdm(
                executor.map(
                    process_data,
                    datas,
                    repeat(id2category_dict),
                    repeat(dataset_name),
                ),
                total=len(datas),
                desc=f"Processing {dataset_name}",
            )
        )

    for result in results:
        if result:
            correct, total, mismatch = result
            true_count += correct
            total_count += total
            if mismatch:
                mismatches.append(mismatch)

    accuracy = true_count / total_count if total_count else 0
    print(f"{dataset_name} accuracy: {accuracy:.4f}")

    # 保存不匹配的结果到文件
    with open(output_mismatches_file, 'w', encoding='utf-8') as f:
        json.dump(mismatches, f, indent=2, ensure_ascii=False)

    print(f"Mismatched predictions saved to {output_mismatches_file}")


if __name__ == "__main__":
    os.environ["PYTHONPATH"] = "/llm_reco/dehua/code/visual-memory"

    dataset_names = ["food172"]  # 根据需要修改数据集列表
    base_dir = "/llm_reco/dehua/code/visual-memory/answers"

    for dataset_name in dataset_names:
        prediction_file = f"answers/food172/Qwen2.5-VL-7B-Instruct_GRPO_food172_all_shot_food172_None_cot_k5.jsonl"
        output_mismatches_file = f"{base_dir}/{dataset_name}/mismatches_{dataset_name}.json"
        
        print(f"Processing file: {prediction_file}")
        main(prediction_file, output_mismatches_file)
