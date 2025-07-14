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

    
    def get_food_prompts(filename, data_proportion):
        test_data = load_jsonl(filename)
        num_samples = int(len(test_data) * data_proportion)
        test_data = test_data[:num_samples]
        messages = []
        image_path_list = []

        for data in tqdm(test_data, total=len(test_data), desc="Loading test data"):
            # Create multi-round dialogue
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please help me determine the category of this target food image. We will an example of an AI assistant's judgment on the image with the highest similarity to the target image will be shown Additionally, we provide the top-5 categories with the highest similarity to the target image. Based on all the information above, please determine the category of the target food image."},
                    ],
                }
            ]
            # Add retrieval images and their categories
            for idx, (retrieval_image, category) in enumerate(zip(data["retrieval_images"], data["categories"])):
                if idx == 0:
                    conversation.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": retrieval_image},
                                {"type": "text", "text": "what dish is this?"},
                            ],
                        }
                    )
                    conversation.append(
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": category},
                            ],
                        }
                    )
                conversation.append(
                    {
                        "role": "user",
                    "content": [
                        {"type": "text", "text": f"The categories of the top-5 images with the highest similarity to the current food image are: {data['categories'][0]}, {data['categories'][1]}, {data['categories'][2]}, {data['categories'][3]}, {data['categories'][4]}"},
                    ],
                }
            )
            
            # Add final user question
            conversation.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": data["image"]},
                        {"type": "text", "text": "Based on all the information above, please directly provide the category of the target food image. what dish is this?"},
                    ],
                }
            )
            messages.append(conversation)
            
            image_path_list.append(data["image"])
        return messages, image_path_list

    # Construct absolute paths using ROOT_DIR
    root_dir = os.environ.get('ROOT_DIR', '')
    save_filename = os.path.join(root_dir, args.save_filename)
    question_file = os.path.join(root_dir, args.question_file)
    
    os.environ['QWEN_LIMIT_MM_PER_PROMPT'] = '4'
    model = LLM(
        model=args.model_path, 
        max_model_len=4096, 
        limit_mm_per_prompt={"image": 2},
        trust_remote_code=True, 
        gpu_memory_utilization=0.95, 
        tensor_parallel_size=1
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    messages, image_path_list = get_food_prompts(question_file, args.data_proportion)
    llm_inputs = []
    for message in tqdm(messages, total=len(messages), desc="Processing messages"):
        prompt=tokenizer.apply_chat_template(message,tokenize=False,add_generation_prompt=True) 
        image_inputs, video_inputs = process_vision_info(message)
        
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs
        
        
        llm_input = {
                "prompt": prompt,
                "multi_modal_data": mm_data,
            }

        llm_inputs.append(llm_input)

    # Set sampling parameters
    sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=2048)

    outputs = model.generate(llm_inputs, sampling_params=sampling_params)

    # Save the results
    for idx, o in enumerate(outputs):
        generated_text = o.outputs[0].text

        formmatted_data = {
            "question_id": idx,
            "image": image_path_list[idx],
            "text": generated_text,
            "category": "default",
        }
        add_jsonl(save_filename, formmatted_data)

    # Calculate metrics
    accuracy, precision, recall, f1 = get_metrics(save_filename, args.gold_answer_file_path, label_file=args.label_file)
    print(accuracy, precision, recall, f1)
    print(save_filename)

    # Log metrics to wandb
    wandb.log({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

    # Finish the wandb run
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VLM inference with specified parameters.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint.')
    parser.add_argument('--save_filename', type=str, required=True, help='Filename to save the results.')
    parser.add_argument('--gold_answer_file_path', type=str, required=True, help='Path to the gold answer file.')
    parser.add_argument('--label_file', type=str, required=True, help='Path to the label file.')
    parser.add_argument('--question_file', type=str, required=True, help='Path to the question file.')
    parser.add_argument('--data_proportion', type=float, default=1.0, help='Proportion of data to use (0.0 to 1.0).')

    args = parser.parse_args()
    main(args)