import json
import re
from nltk.metrics.distance import edit_distance
from tqdm import tqdm


def build_food2k_id2category(label_file):
    """
    构建 FoodX-251 的 ID→类别 映射。
    class_list.txt 中每行格式 "index class_name"，索引从0开始。
    """
    id2cat = {}
    with open(label_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            parts = line.strip().split('--')
            # 若行中包含索引和类别名，则取类别名；否则整体作为类别
            if len(parts) >= 2:
                category = parts[1].replace('_', ' ')
            else:
                category = parts[0].replace('_', ' ')
            id2cat[idx] = category.lower()
    return id2cat

def find_most_similar_word(target, candidates):
    """在候选列表中找到与 target 编辑距离最小的词及其距离。"""
    best_word, best_dist = None, float('inf')
    for w in candidates:
        d = edit_distance(target, w)
        if d < best_dist:
            best_dist, best_word = d, w
    return best_word, best_dist


def evaluate_food2k(pred_file, label_file, save_mismatch_json=None, extract_answer=False):
    """
    评估 FoodX-251 数据集预测准确率。
    - pred_file: JSONL 文件，每行包含 "text" 和 "image"。
    - label_file: FoodX-251 的类别列表文件。
    - save_mismatch_json: 若指定，则把所有不匹配项写入该文件。
    - extract_answer: 是否抽取 <answer> 标签内的内容作为预测。
    返回 (accuracy, total, correct)。
    """
    # 构建类别映射
    id2cat = build_food2k_id2category(label_file)
    all_categories = list(id2cat.values())

    total, correct = 0, 0
    mismatches = []

    # 逐行评估
    with open(pred_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Evaluating FoodX-251"):
            item = json.loads(line)
            text = item.get("text", "").lower().strip()

            # 根据 extract_answer 决定是否抽取 <answer> 标签
            if extract_answer:
                m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
                if not m:
                    continue
                pred = m.group(1).strip().lower()
            else:
                pred = text

            # 提取真实类别：路径倒数第二级目录名
            img_path = item.get("image", "")
            try:
                true_id = int(img_path.split("/")[-2])
                true_cat = id2cat[true_id]
            except Exception:
                continue
            total += 1
            if pred.lower() == true_cat.lower():
                correct += 1
            else:
                # 近似匹配检查
                closest, dist = find_most_similar_word(pred, all_categories)
                if closest.lower() == true_cat.lower():
                    correct += 1
                else:
                    mismatches.append({
                        "prediction": pred,
                        "truth": true_cat,
                        "closest": closest,
                        "distance": dist
                    })

    accuracy = correct / total if total else 0.0
    print(f"Food2k accuracy: {accuracy:.4f} ({correct}/{total})")

    # 保存不匹配记录（可选）
    if save_mismatch_json:
        with open(save_mismatch_json, 'w', encoding='utf-8') as out_f:
            json.dump(mismatches, out_f, indent=2, ensure_ascii=False)
        print(f"Saved mismatches to {save_mismatch_json}")

    return accuracy, total, correct


if __name__ == "__main__":
    # ====== 用户配置区 ======
    prediction_file = "/llm_reco/dehua/code/visual-memory/answers/food2k/Qwen2.5-VL-food2k_cold_sft_nofreeze_food2k_None_att_k5.jsonl"
    label_file = "/llm_reco/dehua/data/food_data/Food2k_complete_jpg/food2k_label2name_en.txt"
    mismatch_output = "/llm_reco/dehua/code/visual-memory/answers/food2k/mismatches_food2k.json"
    extract_ans = True  # 是否从 <answer> 标签中抽取预测
    # =======================

    evaluate_food2k(
        prediction_file,
        label_file,
        save_mismatch_json=mismatch_output,
        extract_answer=extract_ans
    )