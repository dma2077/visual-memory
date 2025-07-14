import argparse
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import Dataset, DataLoader
import torch
import json
from tqdm import tqdm
import torch.nn.functional as F

# 定义FoodDataset类
class FoodDataset(Dataset):
    def __init__(self, file_path, root_dir, label_file):
        self.root_dir = root_dir
        self.category_names = self.load_category_names(label_file)
        self.image_paths, self.labels = self.load_image_paths_and_labels(file_path)
        
    def load_category_names(self, label_file):
        with open(label_file, 'r') as file:
            lines = []
            for line in file:
                line = line.strip()
                if '--' in line:
                    line = line.split('--')[1].replace('-', ' ')
                lines.append(line)
            return lines

    def load_image_paths_and_labels(self, file_path):
        image_paths = []
        labels = []
        with open(file_path, 'r') as file:
            for line in file:
                relative_path = line.strip()
                parts = relative_path.split('/')
                # Determine if the path uses a category ID or a name
                if parts[-2].isdigit():
                    # Format: /1/16_1.jpg
                    category_id = int(parts[-2])  # Extract category ID
                    category_name = self.category_names[category_id - 1]  # Map ID to name
                else:
                    # Format: apple_pie/103801
                    category_name = parts[0].replace("_", " ")  # Use the category name directly

                full_path = f"{self.root_dir}/{relative_path}"
                if not full_path.endswith(('.jpg', '.jpeg')):
                    full_path += '.jpg'
                image_paths.append(full_path)
                labels.append(category_name)
        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image_path = self.image_paths[idx]
            label = self.labels[idx]
            return image_path, label  # Return both the image path and the category name
        except Exception as e:
            print(f"Error accessing image path {self.image_paths[idx]}: {e}")
            return self.__getitem__((idx + 1) % len(self))  # Handle error by trying the next index

def get_texts(filename, template):
    with open(filename, 'r') as f:
        lines = f.readlines()
    return [template.format(line.strip().replace("_", " ")) for line in lines]

# 定义处理批次的函数
def process_batch(batch, model, processor, k=2, texts=None):
    from torch.cuda.amp import autocast
    image_paths, labels = zip(*batch)  # Unpack image paths and labels
    images = [Image.open(image_path) for image_path in image_paths]
    with torch.no_grad():
        with autocast():
            # 模型推理
            inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to('cuda') for k, v in inputs.items() if isinstance(v, torch.Tensor)}  # Move inputs to GPU
            outputs = model(**inputs)
            
            # 获取logit分数并提取前k个logits及其对应的索引
            logits_per_image = outputs.logits_per_image
            probs_per_image = F.softmax(logits_per_image, dim=1)  # Convert logits to probabilities
            topk_probs, topk_ids = torch.topk(probs_per_image, k, dim=1)
            predicted_ids = topk_ids[:, 0]  # 每行第一个是logit最大值，即预测的结果
    return topk_ids, topk_probs, predicted_ids, image_paths, labels

# 保存推理结果到文件
def save_results(image_paths, labels, topk_ids, topk_probs, results, category_names):
    for i in range(len(image_paths)):
        top_categories = [category_names[idx] for idx in topk_ids[i].tolist()]
        result = {
            "image_path": image_paths[i],  # Use the image path directly from the batch
            "label": labels[i],
            "top_c": top_categories,
            "top_logit": topk_probs[i].tolist()
        }
        results.append(result)


def compute_accuracy_from_file(output_file):
    with open(output_file, 'r') as f:
        results = json.load(f)  # Load the entire JSON file

    total_count = len(results)
    correct_count = 0

    for data in results:
        label = data.get("label", "")
        top_c = data.get("top_c", [])
        if top_c[0].lower() == label.lower():
            correct_count += 1

    if total_count == 0:
        print("No valid entries found.")
        return 0.0

    accuracy = correct_count / total_count
    print(f"Number of correct predictions: {correct_count}")
    print(f"Total predictions: {total_count}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy


def main(args):
    # 加载CLIP模型和处理器
    model = CLIPModel.from_pretrained(args.model_path).to('cuda')  # Move model to GPU
    processor = CLIPProcessor.from_pretrained(args.model_path)

    # 创建数据集和DataLoader
    dataset = FoodDataset(args.file_path, args.root_dir, args.label_file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: x)

    # # 初始化保存的结果
    all_results = []
    texts = get_texts(args.label_file, args.template)
    
    # 遍历数据集并进行批次推理
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Processing Batches"):
        topk_ids, topk_probs, predicted_ids, image_paths, labels = process_batch(batch, model, processor, k=args.k, texts=texts)

        # 保存每个批次的推理结果
        save_results(image_paths, labels, topk_ids, topk_probs, all_results, dataset.category_names)

    # 将所有批次结果保存到同一个文件中
    with open(args.output_file, 'w') as f:
        json.dump(all_results, f, indent=4)  # Save the results as a JSON 

    # 计算并输出准确率
    compute_accuracy_from_file(args.output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP Batch Inference with FoodDataset")
    # 添加参数
    parser.add_argument('--model_path', type=str, default="openai/clip-vit-large-patch14", help="Path to the pretrained CLIP model")
    parser.add_argument('--file_path', type=str, required=True, help="Path to the image list file")
    parser.add_argument('--root_dir', type=str, required=True, help="Root directory of images")
    parser.add_argument('--label_file', type=str, required=True, help="Path to the label file")
    parser.add_argument('--batch_size', type=int, default=5, help="Batch size for inference")
    parser.add_argument('--k', type=int, default=2, help="Number of top logits to save")
    parser.add_argument('--output_file', type=str, default="inference_results.json", help="File to save inference results")
    parser.add_argument('--template', type=str, default="a photo of a {}", help="Template for text descriptions")

    # 解析参数
    args = parser.parse_args()
    # 执行主函数
    main(args)