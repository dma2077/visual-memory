import json
import re
from nltk.metrics.distance import edit_distance
from tqdm import tqdm
import sys

def build_food172_id2category(label_file):
    """
    构建 Food172 数据集的 ID→类别 映射。
    文件每行一个类别名称，对应的索引从 0 开始；
    真实类别索引存储在图片路径中，但需要减 1。
    """
    id2cat = {}
    with open(label_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            id2cat[idx] = line.strip().lower()
    return id2cat

def find_most_similar_word(target, candidates):
    """在候选列表中找到与 target 编辑距离最小的词及其距离。"""
    best_word, best_dist = None, float('inf')
    for w in candidates:
        d = edit_distance(target, w)
        if d < best_dist:
            best_dist, best_word = d, w
    return best_word, best_dist

def extract_category(text):
    match = re.search(r'Category is\s*(.+?)\s*\.', text, re.I)
    if match:
        return match.group(1)
    return text

def evaluate_food172(pred_file, label_file, save_mismatch_json=None, extract_answer=False, attribute=False):
    """
    评估 Food172 预测：
      - pred_file: JSONL，内部每行 {"text": "...", "image": ".../<id>/..."} 
      - label_file: Food172 的类别列表文件
      - save_mismatch_json: 若指定，则把所有不匹配项写入该文件
      - extract_answer: 是否只抽取 <answer>...</answer> 标签内的内容作为预测，默认为 False
    返回 (accuracy, total, correct)
    """
    # 1. 加载类别映射
    id2cat = build_food172_id2category(label_file)
    all_cats = list(id2cat.values())
    
    total, correct = 0, 0
    mismatches = []

    # 2. 逐行读取预测并计算
    with open(pred_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Evaluating Food172"):
            item = json.loads(line)
            text = item.get("text", "").lower().strip()
            img_path = item.get("image", "")

            # 根据 extract_answer 决定是否抽取 answer 标签
            if extract_answer:
                m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
                if not m:
                    pred = text
                else:
                    pred = m.group(1).strip()
            else:
                # 不抽取标签，直接把整个输出作为预测
                pred = text
            if attribute:
                pred = extract_category(text)
            # 提取真实类别：路径倒数第二级为原始 ID，需要 -1
            try:
                true_id = int(img_path.split("/")[-2]) - 1
                true_cat = id2cat[true_id]
            except Exception:
                continue

            total += 1
            if pred.lower() == true_cat.lower():
                correct += 1
            else:
                # 近似匹配
                closest, dist = find_most_similar_word(pred, all_cats)
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
    print(f"Food172 accuracy: {accuracy:.4f} ({correct}/{total})")

    # 3. 保存不匹配记录（可选）
    if save_mismatch_json:
        with open(save_mismatch_json, 'w', encoding='utf-8') as out_f:
            json.dump(mismatches, out_f, indent=2, ensure_ascii=False)
        print(f"Saved mismatches to {save_mismatch_json}")

    return accuracy, total, correct

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate_food172.py <prediction_file>")
        sys.exit(1)
    
    prediction_file = sys.argv[1]
    label_file = "/llm_reco/dehua/data/food_data/VireoFood172/SplitAndIngreLabel/FoodList.txt"
    mismatch_output = "/llm_reco/dehua/code/visual-memory/answers/food172/mismatches_food172.json"
    extract_ans = True
    attribute = False

    evaluate_food172(
        prediction_file,
        label_file,
        save_mismatch_json=mismatch_output,
        extract_answer=extract_ans,
        attribute=attribute
    )