from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.metrics.distance import edit_distance
import json
from tqdm import tqdm

def sort_jsonl_by_question_id(input_file, output_file):
    # 读取 JSONL 文件中的所有行
    with open(input_file, 'r', encoding='utf-8') as f:
        json_lines = [json.loads(line) for line in f]
    
    # 按照 question_id 对 JSON 对象进行排序
    sorted_json_lines = sorted(json_lines, key=lambda x: int(x['question_id']))
    
    # 将排序后的 JSON 对象写回到新的 JSONL 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for json_obj in sorted_json_lines:
            f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')

def read_jsonl(file):
    data_list = []
    with open(file, mode='r', encoding='utf-8') as fr:
        for line in fr.readlines():
            data = json.loads(line)
            data_list.append(data)
    return data_list

def find_most_similar_word(target_word, word_list):
    min_distance = float('inf')
    most_similar_word = None
    for word in word_list:
        distance = edit_distance(target_word, word)
        if distance < min_distance:
            min_distance = distance
            most_similar_word = word
    return most_similar_word, min_distance

def get_metrics(answer_file_path, gold_answer_file_path, retrieval_file_path=None, error_output_file=None):
    answer_file = read_jsonl(answer_file_path)
    gold_answer_file = read_jsonl(gold_answer_file_path)
    if retrieval_file_path:
        retrieval_file = read_jsonl(retrieval_file_path)
    
    answer_dict = {}
    # Build answer_dict with all keys in lowercase
    idx = 0
    for data in gold_answer_file:
        key = data['text'].lower()
        if key not in answer_dict:
            answer_dict[key] = idx
            idx += 1
    print(answer_dict)
    y_true = []
    for i in range(len(gold_answer_file)):
        # Ensure we access the correct lowercased key
        y = answer_dict[gold_answer_file[i]['text'].lower()]
        y_true.append(y)

    y_pred = []
    question_ids = []
    errors = []  # List to store errors
    for i in tqdm(range(len(answer_file))):
        if answer_file[i]['question_id'] not in question_ids:
            question_ids.append(answer_file[i]['question_id'])
            # Check and get the lowercased key from answer_dict
            text_lower = answer_file[i]['text'].lower()
            text_lower = text_lower.split('\n')[0]
            # Check if text_lower contains any answer_dict key
            found_match = False
            for key in answer_dict:
                if key == text_lower:
                    y = answer_dict[key]
                    y_pred.append(y)
                    found_match = True
                    break

            # If no match is found, find the most similar word
            if not found_match:
                answer, _ = find_most_similar_word(text_lower, list(answer_dict.keys()))
                y = answer_dict[answer]
                y_pred.append(y)
            if error_output_file:
            # Record errors
                if y_pred[-1] != y_true[i]:
                    errors.append({
                        "question_id": answer_file[i]['question_id'],
                        "image_path": answer_file[i]['image_path'],
                        "predicted": answer_file[i]['text'],  # Use the original predicted text
                        "actual": gold_answer_file[i]['text'],  # Use the original actual text
                        "text": answer_file[i]['text'],
                    })

    if error_output_file:
    # Save errors to a file
        with open(error_output_file, 'w', encoding='utf-8') as f:
            for error in errors:
                f.write(json.dumps(error, ensure_ascii=False) + '\n')

    # Evaluate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    return accuracy, precision, recall, f1

if __name__ == "__main__":
    gold_answer_file_path = "/map-vepfs/dehua/code/visual-memory/answers/groundtruth/food172_answers.jsonl"
    sorted_answer_file_path = '/map-vepfs/dehua/code/visual-memory/answers/food172/qwen2-vl-7b_rank.jsonl'
    #retrieval_file_path = '/home/madehua/code/visual-memory/questions/multi_image/food172/test_5.jsonl'  # Specify the path to the retrieval file
    retrieval_file_path = None
    #error_output_file = '/home/madehua/code/visual-memory/answers/multi_turn/food101/qwen2-vl-7b_k5_error.jsonl'  # Specify the path to save errors
    error_output_file = None
    accuracy, precision, recall, f1 = get_metrics(sorted_answer_file_path, gold_answer_file_path, retrieval_file_path, error_output_file)
    print(accuracy, precision, recall, f1)
    print(sorted_answer_file_path)