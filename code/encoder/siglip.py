import argparse
from PIL import Image
from transformers import AutoProcessor, AutoModel
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
                    line = line.split('--')[1].replace('_', ' ')
                if line.split(' ')[0].isdigit():
                    line = line.split(' ')[1]
                else: 
                    line = line.replace('_', ' ')
                lines.append(line)
            print(lines)
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
                    if args.dataset == "food172":
                        category_name = self.category_names[category_id - 1]  # Map ID to name
                    else:
                        category_name = self.category_names[category_id]
                else:
                    # Format: apple_pie/103801
                    category_name = parts[0].replace("_", " ")  # Use the category name directly
                if len(relative_path.split(' ')) == 2:
                    relative_path = relative_path.split(' ')[0]
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
    image_paths, labels = zip(*batch)
    images = [Image.open(image_path).convert('RGB') for image_path in image_paths]  # Convert images to RGB
    with torch.no_grad():
        with autocast():
            inputs = processor(text=texts, images=images, padding="max_length", return_tensors="pt")
            inputs = {k: v.to('cuda') for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            outputs = model(**inputs)
            
            logits_per_image = outputs.logits_per_image
            probs_per_image = torch.softmax(logits_per_image, dim=1)  # 使用 sigmoid 而不是 softmax
            topk_probs, topk_ids = torch.topk(probs_per_image, k, dim=1)
            predicted_ids = topk_ids[:, 0]
    return topk_ids, topk_probs, predicted_ids, image_paths, labels

# 保存推理结果到文件
def save_results(image_paths, labels, topk_ids, topk_probs, results, category_names):
    for i in range(len(image_paths)):
        top_categories = [category_names[idx] for idx in topk_ids[i].tolist()]
        result = {
            "image_path": image_paths[i],  # Use the image path directly from the batch
            "label": labels[i],
            "categories": top_categories,
            "similarities": topk_probs[i].tolist()
        }
        results.append(result)


def compute_accuracy_from_file(output_file):
    results = []
    with open(output_file, 'r') as f:
        for line in f:
            result = json.loads(line.strip())
            results.append(result)
    total_count = len(results)
    correct_count = 0

    for data in results:
        label = data.get("label", "")
        top_c = data.get("categories", [])
        if top_c[0].lower().replace('_', ' ') == label.lower():

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
    # 加载 SigLIP 模型和处理器
    model = AutoModel.from_pretrained(args.model_path).to('cuda')
    processor = AutoProcessor.from_pretrained(args.model_path)

    # 创建数据集和DataLoader
    dataset = FoodDataset(args.file_path, args.root_dir, args.label_file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: x)

    all_results = []
    texts = get_texts(args.label_file, args.template)
    
    # 遍历数据集并进行批次推理
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Processing Batches"):
        topk_ids, topk_probs, predicted_ids, image_paths, labels = process_batch(batch, model, processor, k=args.k, texts=texts)

        # 保存每个批次的推理结果
        save_results(image_paths, labels, topk_ids, topk_probs, all_results, dataset.category_names)

    # 将所有批次结果保存到同一个文件中
    with open(args.output_file, 'w') as f:
        for result in all_results:
            json_line = json.dumps(result)
            f.write(json_line + '\n')

    # 计算并输出准确率
    compute_accuracy_from_file(args.output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SigLIP Batch Inference with FoodDataset")
    # 添加参数
    parser.add_argument('--model_path', type=str, default="google/siglip-so400m-patch14-384", help="Path to the pretrained SigLIP model")
    parser.add_argument('--file_path', type=str, required=True, help="Path to the image list file")
    parser.add_argument('--root_dir', type=str, required=True, help="Root directory of images")
    parser.add_argument('--label_file', type=str, required=True, help="Path to the label file")
    parser.add_argument('--batch_size', type=int, default=5, help="Batch size for inference")
    parser.add_argument('--k', type=int, default=2, help="Number of top logits to save")
    parser.add_argument('--output_file', type=str, default="inference_results.json", help="File to save inference results")
    parser.add_argument('--template', type=str, default="a photo of a {}", help="Template for text descriptions")
    parser.add_argument('--dataset', type=str, default="food172", help="Dataset name")
    args = parser.parse_args()
    main(args)