from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

# 加载新的模型和预处理器
device = "cuda" if torch.cuda.is_available() else "cpu"  # 使用 GPU 加速
model = AutoModelForCausalLM.from_pretrained(
    "/map-vepfs/models/Qwen2.5-14B-Instruct"
).to(device)
tokenizer = AutoTokenizer.from_pretrained("/map-vepfs/models/Qwen2.5-14B-Instruct")


# 自定义 Dataset 类
class TextDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]


# 计算 token 数量的函数
def calculate_token_counts(sentences):
    # 使用新的 tokenizer 对句子进行处理
    inputs = tokenizer(
        sentences, return_tensors="pt", padding=True, truncation=True
    ).to(device)

    # 获取每个句子的 token 数量
    token_counts = inputs["input_ids"].shape[
        1
    ]  # 获取每个句子的 token 数量（包括 padding）

    return token_counts  # 返回批次中所有句子的 token 数量


# 加载 JSON 数据并提取句子
file_path = (
    "/map-vepfs/dehua/data/data/ShareGPT4V/share-captioner_coco_lcs_sam_1246k_1107.json"
)
sentences = []

# 从 JSON 文件中提取所有句子，并显示进度条
with open(file_path, "r") as f:
    data = json.load(f)

    # 使用 tqdm 显示进度条
    for item in tqdm(data, desc="Processing items", unit="item"):
        try:
            # 假设 "conversations" 中的第二项是我们需要的句子
            sentence = item["conversations"][1]["value"]
            print(len(sentence))  # 打印每个句子的长度（字符数）
            sentences.append(sentence)
        except (KeyError, IndexError):
            # 如果遇到格式不符合预期的条目，跳过
            continue

# 使用 DataLoader 进行批量处理
batch_size = 32
dataset = TextDataset(sentences)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# 计算所有句子的平均 token 数量
total_token_count = 0
total_sentences = 0

# 使用 tqdm 显示 DataLoader 的进度
for batch in tqdm(dataloader, desc="Processing batches", unit="batch"):
    per_sentence_token_counts = calculate_token_counts(
        batch
    )  # 获取每个句子的 token 数量

    # 对批次中的每个句子 token 数量进行累加
    total_token_count += per_sentence_token_counts * len(batch)
    total_sentences += len(batch)

average_token_count = total_token_count / total_sentences if total_sentences else 0

print(f"Average number of tokens: {average_token_count}")
