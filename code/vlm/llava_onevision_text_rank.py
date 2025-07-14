import torch
import PIL
from vllm import LLM, SamplingParams
from tqdm import tqdm
import os
import argparse
import wandb
from code.utils import *
from code.metric import get_metrics
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
from PIL import Image 

prompt_template = """
We use two retrievers to identify the category of the current image. The food categories returned by the first retriever, listed in descending order of similarity, are {content1}; similarly, the categories returned by the second retriever, also sorted by descending similarity, are {content2}. Please make a comprehensive judgment on the final category of the image based on the current image and the results from both retrievers. Only provide the category name; no need to include other details.
"""

prompt_template1 = """

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

def get_food_prompts_multi_model(filename1, filename2, data_proportion, k1):

    test_data1 = load_jsonl(filename1)
    test_data2 = load_jsonl(filename2)
    num_samples = int(len(test_data1) * data_proportion)
    test_data1 = test_data1[:num_samples]
    test_data2 = test_data2[:num_samples]
    messages = []
    image_path_list = []

    for idx, data in tqdm(enumerate(test_data1), total=len(test_data1), desc="Loading test data"):
        # Add final user question
        categories1 = data["categories"][:k1]
        categories2 = test_data2[idx]["categories"][:k1]

        content1 = ", ".join([f"{category}" for category in categories1])
        content2 = ", ".join([f"{category}" for category in categories2])
            
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": data["image"]},
                    {"type": "text", "text": prompt_template.format(content1=content1, content2=content2)},
                ],
            }
        ]
        messages.append(conversation)
        
        image_path_list.append(data["image"])
    return messages, image_path_list


def get_llm_input(messages, tokenizer):
    llm_inputs = []
    for message in tqdm(messages, total=len(messages), desc="Processing messages"):
        question = message[0]["content"][1]["text"]
        prompt = f"<|im_start|>user <image>\n{question}<|im_end|> \
        <|im_start|>assistant\n"
        image_inputs = Image.open(message[0]["content"][0]["image"])
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        
        llm_input = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
        }

        llm_inputs.append(llm_input)
    
    return llm_inputs


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
    
    model = LLM(
        model=args.model_path, 
        max_model_len=2048*4, 
        limit_mm_per_prompt={"image": 1},
        trust_remote_code=True, 
        gpu_memory_utilization=0.95, 
        tensor_parallel_size=1
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if args.mode == "mix":     
        messages, image_path_list = get_food_prompts_multi_model(question_file, siglip_file, args.data_proportion, args.k1)
    elif args.mode == "dinov2":
        messages, image_path_list = get_food_prompts(question_file, args.data_proportion)
    elif args.mode == "siglip":
        messages, image_path_list = get_siglip_food_prompts(question_file, args.data_proportion)

    llm_inputs = get_llm_input(messages, tokenizer)

    # Set sampling parameters
    sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=2048)

    outputs = model.generate(llm_inputs, sampling_params=sampling_params)

    # Save the results
    formmatted_datas = []
    for idx, o in enumerate(outputs):
        generated_text = o.outputs[0].text

        formmatted_data = {
            "question_id": idx,
            "image": image_path_list[idx],
            "text": generated_text,
            "category": "default",
        }
        formmatted_datas.append(formmatted_data)
    save_jsonl(save_filename, formmatted_datas)


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
    args = parser.parse_args()
    main(args)
