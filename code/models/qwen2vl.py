import torch
import PIL
from vllm import LLM, SamplingParams
from tqdm import tqdm
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from qwen_vl_utils import process_vision_info
import random

class Qwen2VL():
    def __init__(
            self,
            model_path: str = "Qwen/Qwen2-VL-7B-Instruct",
            max_model_len: int = 2048,
            tensor_parallel_size: int = 1,
            temperature: float = 0.0,
        ):
            self._model = LLM(
                model=model_path, 
                max_model_len=max_model_len, 
                trust_remote_code=True, 
                gpu_memory_utilization=0.95, 
                tensor_parallel_size=tensor_parallel_size,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                # Set sampling parameters
            self._sampling_params = SamplingParams(temperature=temperature, top_p=0.95, max_tokens=2048)

        
    def _process_inputs(self, messages):
        llm_inputs = []
        for message in tqdm(messages, total=len(messages), desc="Processing messages"):
                prompt=self._tokenizer.apply_chat_template(message,tokenize=False,add_generation_prompt=True) 
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
        print("processing message ending...")
        return llm_inputs
    def generate_until(self, messages):
        llm_inputs = self._process_inputs(messages)
        outputs = self._model.generate(llm_inputs, sampling_params=self._sampling_params)
        return outputs

