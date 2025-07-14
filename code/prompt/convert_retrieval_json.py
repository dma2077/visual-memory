from utils import *

food101_question_path = '/mnt/madehua/fooddata/json_file/101_questions.jsonl'


questions = load_jsonl(food101_question_path)


conversations = []
for idx, question in enumerate(questions):
    tamplate = "The categories of the k images most similar to this image are:"
    question_tamplate = "Based on the information above, please answer the following questions. What dish is this? Just provide its category."
    category = question["image"].split('/')[-2].replace('_', ' ')
    tamplate += ' ' + category + ', '
    for i in range(3):
        tamplate += category + ', '
    tamplate += category + '. '
    tamplate += question_tamplate
    conversation = {
        "question_id": idx,
        "image":  question["image"],
        "text": tamplate,
        "categrory": "default"
    }
    conversations.append(conversation)

food101_gold_retrieval_path = '/mnt/madehua/fooddata/json_file/101_gold_retrieval_questions.jsonl'
save_jsonl(food101_gold_retrieval_path, conversations)
