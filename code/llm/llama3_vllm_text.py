import torch
import PIL
from vllm import LLM, SamplingParams
from tqdm import tqdm
import os
import argparse
import wandb
from code.utils import *
from code.metric import get_metrics
from code.metric_calculator import main as get_metrics_main
from code.metric_calculator import get_dataset_name, get_label_file
from code.metric_calculator import build_id2category_dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from qwen_vl_utils import process_vision_info
import random

question_template = """
We use two retrievers to identify the category of the current image. The food categories and similarity scores returned by the first retriever are {content1}, and those returned by the second retriever are {content2}. Based on the current image and the information provided by both retrievers, please make a comprehensive judgment on the category of the current image. Only provide the category name; no need to include similarity scores or other details.
"""

template = """
   <|begin_of_text|><|start_header_id|>user<|end_header_id|>
   {content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
   """
   
def get_food_prompts(filename, data_proportion):
    test_data = load_jsonl(filename)
    num_samples = int(len(test_data) * data_proportion)
    test_data = test_data[:num_samples]
    messages = []
    image_path_list = []
    for data in tqdm(test_data, total=len(test_data), desc="Loading test data"):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": data["image"]},
                    {"type": "text", "text": data["text"]},
                ],
            }
        ]
        messages.append(conversation)
        image_path_list.append(data["image"])
    return messages, image_path_list

def get_siglip_food_prompts(filename, data_proportion):
    test_data = load_jsonl(filename)
    messages = []
    image_path_list = []

    for idx, data in tqdm(enumerate(test_data), total=len(test_data), desc="Loading test data"):
        # Add final user question
        categories = data["categories"]
        similarities = data["similarities"]
        prompt = "The categories of the 5 images most similar to this image are: "
        categories = [img['category'].replace('_', ' ') for img in categories]
        prompt += ", ".join(categories)
        prompt += ". Based on the information above, please answer the following questions. What dish is this? Just provide its category."
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": data["image"]},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        messages.append(conversation)
        
        image_path_list.append(data["image"])

def get_food_prompts_multi_model(filename1, filename2, data_proportion, k1, category2id_dict):

    test_data1 = load_jsonl(filename1)
    test_data2 = load_jsonl(filename2)
    num_samples = int(len(test_data1) * data_proportion)
    test_data1 = test_data1[:num_samples]
    test_data2 = test_data2[:num_samples]
    messages = []
    image_path_list = []
    prompts = []
    for idx, data in tqdm(enumerate(test_data1), total=len(test_data1), desc="Loading test data"):
        # Add final user question
        categories1 = data["categories"][:k1]
        similarities1 = data["similarities"][:k1]
        categories2 = test_data2[idx]["categories"][:k1]
        similarities2 = test_data2[idx]["similarities"][:k1]

        
        content1 = ", ".join([f"{category2id_dict[category.lower()]}" for category in categories1])
        content2 = ", ".join([f"{category2id_dict[category.lower()]}" for category in categories2])

        question = question_template.format(content1=content1, content2=content2)
        prompt = template.format(content=question)
        prompts.append(prompt)
        image_path_list.append(data["image"])
    return prompts, image_path_list


def main(args):
    os.environ['NCCL_SHM_DISABLE'] = '1'
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['NCCL_IB_TIMEOUT'] = '22'
    # Initialize wandb
    wandb.init(project="vlm-inference", config=args)

        # Print all the arguments
    print("Running with the following parameters:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    # Construct absolute paths using ROOT_DIR
    root_dir = os.environ.get('ROOT_DIR', '')
    save_filename = os.path.join(root_dir, args.save_filename)
    question_file = os.path.join(root_dir, args.question_file)
    siglip_file = os.path.join(root_dir, args.siglip_file)
    label_file = get_label_file(args.dataset_name)
    id2category_dict = build_id2category_dict(label_file, args.dataset_name)
    category2id_dict = {v: f"category{k}" for k, v in id2category_dict.items()}
    id2category_dict = {v: k for k, v in category2id_dict.items()}
    model = LLM(
        model=args.model_path, 
        max_model_len=2048, 
        trust_remote_code=True, 
        gpu_memory_utilization=0.95, 
        tensor_parallel_size=1
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if args.mode == "mix":     
        prompts, image_path_list = get_food_prompts_multi_model(question_file, siglip_file, args.data_proportion, args.k1, category2id_dict)
    elif args.mode == "dinov2":
        prompts, image_path_list = get_food_prompts(question_file, args.data_proportion)
    elif args.mode == "siglip":
        prompts, image_path_list = get_siglip_food_prompts(question_file, args.data_proportion)

    # Set sampling parameters
    sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=2048)
    outputs = model.generate(prompts, sampling_params=sampling_params)

    # Save the results
    formmatted_datas = []
    for idx, o in enumerate(outputs):
        generated_text = o.outputs[0].text.strip()
        if generated_text in id2category_dict:
            generated_text = id2category_dict[generated_text]
        else:
            generated_text = "unknown"
        formmatted_data = {
            "question_id": idx,
            "image": image_path_list[idx],
            # "prompt": prompt,
            "text": generated_text,
            "category": "default",
        }
        formmatted_datas.append(formmatted_data)
    save_jsonl(save_filename, formmatted_datas)
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VLM inference with specified parameters.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint.')
    parser.add_argument('--save_filename', type=str, required=True, help='Filename to save the results.')
    parser.add_argument('--gold_answer_file_path', type=str, required=True, help='Path to the gold answer file.')
    parser.add_argument('--label_file', type=str, required=True, help='Path to the label file.')
    parser.add_argument('--question_file', type=str, required=True, help='Path to the question file.')
    parser.add_argument('--siglip_file', type=str, required=True, help='Path to the siglip file.')
    parser.add_argument('--data_proportion', type=float, default=1.0, help='Proportion of data to use (0.0 to 1.0).')
    parser.add_argument('--mode', type=str, default='mix', help='Proportion of data to use (0.0 to 1.0).')
    parser.add_argument('--k1', type=int, default='3', help='retrieval item number')
    parser.add_argument('--dataset_name', type=str, default='food2k', help='dataset name')
    args = parser.parse_args()
    main(args)