import os
import argparse
import json
from tqdm import tqdm
import torch
from PIL import Image, UnidentifiedImageError # 明确导入 UnidentifiedImageError
import numpy as np
# 导入 Hugging Face 的 CLIP 相关库
from transformers import CLIPProcessor, CLIPModel

# 不再需要 OpenAI 的 clip 包

def load_json(file):
    # (保持不变)
    try:
        with open(file, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"错误: 文件未找到 {file}")
        return None
    except json.JSONDecodeError:
        print(f"错误: 解析 JSON 文件失败 {file}")
        return None

def get_id2item_dict(item2id_file):
    # (保持不变)
    if not os.path.exists(item2id_file):
        print(f"错误: item2id 文件未找到 {item2id_file}")
        return {}
    with open(item2id_file, 'r') as fp:
        all_item2id = fp.readlines()
        
    id2item = {}
    for line in all_item2id:
        try:
            item, item_id = line.strip().split('\t')
            id2item[item_id] = item 
        except ValueError:
            continue 
    return id2item

# --- BACKBONE_MAP 不再需要 ---

def get_feature(args):
    # --- 修改：打印信息使用 model_name_or_path ---
    print(f"开始处理: Dataset={args.dataset}, Version={args.data_version}, Model={args.model_name_or_path}")

    # --- 路径构建 (保持不变) ---
    image_base_path = os.path.join(args.image_root, f'amazon{args.data_version}', 'Images')
    image_file_path = os.path.join(image_base_path, args.dataset)
    images_info_path = os.path.join(image_base_path, f'{args.dataset}_images_info.json')
    processed_data_path = os.path.join(args.save_root, args.dataset)
    item2id_file = os.path.join(processed_data_path, f'{args.dataset}.item2id')

    # --- 2. 加载所需文件 (保持不变) ---
    print(f"从 {item2id_file} 加载 item2id 映射...")
    id2item = get_id2item_dict(item2id_file)
    if not id2item:
        print(f"错误: 未能从 {item2id_file} 加载任何物品ID。")
        return

    print(f"从 {images_info_path} 加载图片信息...")
    images_info = load_json(images_info_path)
    if images_info is None:
        print(f"错误: 加载图片信息失败 {images_info_path}")
        return
        
    # --- 3. 加载 Hugging Face CLIP 模型 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'使用设备: {device}')
    
    # --- 关键修改：直接使用 args.model_name_or_path ---
    print(f'加载 Hugging Face CLIP 模型: {args.model_name_or_path} ...')
    try:
        processor = CLIPProcessor.from_pretrained(args.model_name_or_path, cache_dir=args.model_cache_dir)
        model = CLIPModel.from_pretrained(args.model_name_or_path, cache_dir=args.model_cache_dir).to(device)
        model.eval() 
        embedding_dim = model.config.projection_dim 
        print(f"模型加载成功，嵌入维度: {embedding_dim}")
    except Exception as e:
        print(f"加载 Hugging Face 模型或预处理器失败: {e}")
        return

    # --- 4. 提取特征 (逻辑基本不变) ---
    embeddings = []
    sorted_ids = sorted(id2item.keys(), key=int) 

    with torch.no_grad():
        for mapped_id_str in tqdm(sorted_ids, desc="提取图像特征"):
            original_item_id = id2item.get(mapped_id_str)
            if not original_item_id:
                embeddings.append(torch.zeros(embedding_dim)) 
                continue

            image_name_list = images_info.get(original_item_id, [])
            if not isinstance(image_name_list, list): image_name_list = [] 

            image_feature = torch.zeros(embedding_dim) 
            image_processed = False 

            for image_name in image_name_list: 
                if not isinstance(image_name, str) or not image_name: continue 
                
                image_file = os.path.join(image_file_path, image_name)
                
                if not os.path.exists(image_file):
                     continue

                try:
                    image = Image.open(image_file).convert("RGB")
                    inputs = processor(images=image, return_tensors="pt").to(device)
                    image_features = model.get_image_features(**inputs)
                    image_feature = image_features[0].cpu() 
                    image_processed = True
                    break 

                except UnidentifiedImageError:
                     print(f"\n无法识别的图片文件格式: {image_file}")
                     continue 
                except Exception as e:
                    print(f"\n处理图片时发生错误: {e}")
                    print(f"文件路径: {image_file}")
                    image_feature = torch.zeros(embedding_dim) 
                    continue 
            
            embeddings.append(image_feature)
            
    # --- 5. 保存特征文件 ---
    if not embeddings:
         print("错误: 未能提取任何嵌入向量。")
         return
         
    embeddings = torch.stack(embeddings, dim=0).numpy()
    print('Embeddings shape: ', embeddings.shape)

    # --- 关键修改：文件名使用 model_name_or_path ---
    # 将路径中的 / 替换为 - 以创建有效的文件名
    hf_model_tag = args.model_name_or_path.replace('/', '-')
    
    save_dir = os.path.join(processed_data_path, "embeddings")
    os.makedirs(save_dir, exist_ok=True)
    
    save_file = os.path.join(save_dir, f'{args.dataset}.emb-image-{hf_model_tag}.npy')
    
    print(f"正在保存特征到: {save_file}")
    try:
        np.save(save_file, embeddings)
        print("特征提取并保存成功！")
    except Exception as e:
         print(f"保存特征文件失败: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description="使用CLIP模型为Amazon数据集提取图像特征。")
    parser.add_argument('--data_version', type=str, default='14', choices=['14', '18'], help='数据集年份版本 (14 或 18)')
    parser.add_argument('--dataset', type=str, default='Home', help='例如 Baby, All_Beauty, Home 等')
    parser.add_argument('--image_root', type=str, default="../datasets", help='包含 amazon14/ 和 amazon18/ 的根目录')
    parser.add_argument('--save_root', type=str, default="../datasets", help='预处理后数据的根目录')
    # --- 关键修改：用 model_name_or_path 替换 backbone ---
    # 移除了 --backbone 参数
    parser.add_argument('--model_name_or_path', type=str, default='openai/clip-vit-base-patch32', help='Hugging Face Hub 上的 CLIP 模型 ID 或本地路径')
    parser.add_argument('--model_cache_dir', type=str, default='./hf_clip_cache', help='Hugging Face 模型下载缓存目录')
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()
    get_feature(args)