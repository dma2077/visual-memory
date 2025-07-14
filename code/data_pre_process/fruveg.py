import os
from PIL import Image
from tqdm import tqdm  # 导入 tqdm

def resize_image(image_path, output_path, target_size=384):
    with Image.open(image_path) as img:
        # 计算新的尺寸
        if img.width > img.height:
            new_width = target_size
            new_height = int((target_size / img.width) * img.height)
        else:
            new_height = target_size
            new_width = int((target_size / img.height) * img.width)

        # 调整图像大小
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)  # 使用 LANCZOS 代替 ANTIALIAS
        
        # 检查图像模式并转换为 RGB
        if resized_img.mode in ['RGBA', 'P']:
            resized_img = resized_img.convert('RGB')

        # 保存图像
        resized_img.save(output_path)

def process_directory(source_dir, target_dir):
    # 创建目标目录（如果不存在）
    os.makedirs(target_dir, exist_ok=True)

    # 获取所有图像文件
    image_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # 支持的图像格式
                image_files.append(os.path.join(root, file))

    # 使用 tqdm 显示进度条
    for image_path in tqdm(image_files, desc="Processing images"):
        # 生成输出路径，保持子目录结构
        relative_path = os.path.relpath(os.path.dirname(image_path), source_dir)
        output_dir = os.path.join(target_dir, relative_path)
        os.makedirs(output_dir, exist_ok=True)  # 创建输出子目录

        output_path = os.path.join(output_dir, os.path.basename(image_path))  # 保存为新的文件名
        resize_image(image_path, output_path)

if __name__ == "__main__":
    source_dir = "/map-vepfs/dehua/data/data/vegfru-dataset/fru92_images"
    target_dir = "/map-vepfs/dehua/data/data/vegfru-dataset/fru92_images_resized"
    process_directory(source_dir, target_dir)