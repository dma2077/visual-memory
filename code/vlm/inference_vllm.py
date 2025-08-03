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
from qwen_vl_utils import process_vision_info
from code.models.qwen2vl import Qwen2VL
from code.models.qwen2_5vl import Qwen2_5VL
from code.models.internvl import InternVL
from transformers import AutoConfig
import io
import base64
from PIL import Image
import random

prompt_template = """
We use two retrievers to identify the category of the current image. The food categories returned by the first retriever, listed in descending order of similarity, are {content1}; similarly, the categories returned by the second retriever, also sorted by descending similarity, are {content2}. Please make a comprehensive judgment on the final category of the image based on the current image and the results from both retrievers. Only provide the category name; no need to include other details.
"""
cot_prompt_template = """
Please help me identify the category of the food in the image. Before providing the final category, you should first analyze the attributes of the food and then determine its type.
"""
RAW_template = """
Please help me identify the category of the food in the image. 
"""

GRPO_PROMPT = """
"This is an image containing a food. Please identify the categories of the food based on the image.\nOutput the thinking process in <think> </think> and final answer in <answer> </answer> tags. The output answer format should be as follows:\n <think> ... </think> <answer>category name</answer>\n. Please strictly follow the format.
"""

prompt_template1 = """

"""

COT_PROMPT= """
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant \"\n    \"first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning \"\n    \"process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., \"\n    \"<think> reasoning process here </think><answer> answer here </answer>
"""


def image_to_base64(image_path):
    """
    Convert image to base64 string
    """
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # # Resize to 224x224 if needed
            # if img.size != (224, 224):
            #     img = img.resize((224, 224), Image.LANCZOS)
            
            # Convert to base64
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            return img_str
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


def get_food_prompts_unified(mode, filename1, data_proportion=1.0, filename2=None, k1=None):
    """
    Unified function to generate food prompts.

    Args:
        mode (str): Type of prompt generation. Options: 'basic', 'siglip', 'multi_model'.
        filename1 (str): Path to the first dataset (used in all modes).
        data_proportion (float, optional): Proportion of data to use. Default is 1.0.
        filename2 (str, optional): Path to the second dataset (only for 'multi_model' mode).
        k1 (int, optional): Number of categories to use from each dataset (only for 'multi_model' mode).

    Returns:
        messages (list): List of formatted conversation prompts.
        image_path_list (list): List of image paths.
    """
    test_data1 = load_jsonl(filename1)
    num_samples = int(len(test_data1) * data_proportion)
    test_data1 = test_data1[:num_samples]

    messages = []
    image_path_list = []

    if mode == "multi_model":
        if not filename2 or k1 is None:
            raise ValueError("For 'multi_model' mode, filename2 and k1 must be provided.")
    if mode == "mix":
        test_data2 = load_jsonl(filename2)[:num_samples]  # Trim second dataset to match size


    for idx, data in tqdm(enumerate(test_data1), total=len(test_data1), desc="Loading test data"):
        if mode == "dinov2":
            target_width = 114
            target_height = 114
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": data["image"], "resized_width": target_width, "resized_height": target_height,},
                        {"type": "text", "text": data["text"]},
                    ],
                }
            ]

        elif mode == "raw":
            prompt = "What is the category of the food? Just only give the category"
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": data["image"]},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        # elif mode == "cot":
        #     prompt = GRPO_PROMPT
        #     conversation = [
        #         {
        #             "role": "system",
        #             "content": [
        #                 {"type": "text", "text": prompt}
        #             ]
        #         },
        #         {
        #             "role": "user",
        #             "content": [
        #                 {"type": "image", "image": data["image"]},
        #                 {"type": "text", "text": "<image>\nWhat dish is this?"},
        #             ],
        #         }
        #     ]
        elif mode == "nocot":
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": data["image"].replace("/map-vepfs/dehua/data/data", "/llm_reco/dehua/data/food_data").replace("vegfru-dataset/", "")},
                        {"type": "text", "text": "<image>\nWhat is the category of the food?"},
                    ],
                }
            ]
        elif mode == "att":
            image_path = data["image"].replace("/map-vepfs/dehua/data/data", "/llm_reco/dehua/data/food_data").replace("vegfru-dataset/", "")
            base64_image = image_to_base64(image_path)
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {'url': f"data:image/jpeg;base64,{base64_image}"}},
                        {"type": "text", "text": "Please analyze these food attributes in the image: shape, texture, composition, color, and cooking style. Then identify the food category."},
                    ],
                }
            ]
        elif mode == "mix":
            categories1 = data["categories"][:k1]
            categories2 = test_data2[idx]["categories"][:k1]
            content1 = ", ".join(categories1)
            content2 = ", ".join(categories2)
            image_path = data["image"].replace("/map-vepfs/dehua/data/data", "/llm_reco/dehua/data/food_data").replace("vegfru-dataset/", "")
            base64_image = image_to_base64(image_path)
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {'url': f"data:image/jpeg;base64,{base64_image}"}},
                        {"type": "text", "text": prompt_template.format(content1=content1, content2=content2)},
                    ],
                }
            ]

        else:
            raise ValueError("Invalid mode. Choose from 'basic', 'siglip', or 'multi_model'.")

        messages.append(conversation)
        image_path_list.append(data["image"])

    return messages, image_path_list



def get_model(model_path, max_model_len=4096):
    import re
    def extract_number(text):
        match = re.search(r'-(\d+)B', text)
        return int(match.group(1)) if match else None
    model_size = extract_number(model_path)
    if model_size:
        if model_size <= 10:
            tensor_parallel_size = 1
        elif model_size > 10 and model_size < 30:
            tensor_parallel_size = 2
        elif model_size >=30:
            tensor_parallel_size = 4
    else:
        tensor_parallel_size = 1
    print(f"tensor_parallel_size is {tensor_parallel_size}")
    # if "qwen2-vl" in model_path.lower():
    #     model = Qwen2VL(model_path=model_path, max_model_len=max_model_len, tensor_parallel_size=tensor_parallel_size)
    # elif "qwen2.5" in model_path.lower() or "qwen2vl" in model_path.lower():
    #     model = Qwen2_5VL(model_path=model_path, max_model_len=max_model_len, tensor_parallel_size=tensor_parallel_size)
    if "internvl" in model_path.lower():
        model = InternVL(model_path=model_path, max_model_len=max_model_len, tensor_parallel_size=tensor_parallel_size)
    else:
        model = Qwen2_5VL(model_path=model_path, max_model_len=max_model_len, tensor_parallel_size=tensor_parallel_size)
    return model

def main(args):
    os.environ['NCCL_SHM_DISABLE'] = '1'
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['NCCL_IB_TIMEOUT'] = '22'

    # Initialize wandb
    # wandb.init(project="vlm-inference", config=args)

        # Print all the arguments
    print("Running with the following parameters:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    # Construct absolute paths using ROOT_DIR
    root_dir = os.environ.get('ROOT_DIR', '')
    save_filename = os.path.join(root_dir, args.save_filename)
    question_file = os.path.join(root_dir, args.question_file)
    siglip_file = os.path.join(root_dir, args.siglip_file)
    
    model = get_model(args.model_path)
    messages, image_path_list = get_food_prompts_unified(mode=args.mode, data_proportion=args.data_proportion, filename1=question_file, filename2=siglip_file, k1=args.k1)
    outputs = model.generate_until(messages)

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
