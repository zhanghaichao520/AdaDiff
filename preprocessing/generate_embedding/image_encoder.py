# preprocessing/generate_embeddings/image_encoder.py

import os
import torch
import torch.nn.functional as F # 导入 F 用于 normalize
from PIL import Image, UnidentifiedImageError
import numpy as np
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from typing import Dict, List, Any
import sys # 导入 sys

# ✅ (核心修改) 从父目录导入共享函数
try:
    # 添加父目录 (preprocessing/) 到 Python 路径
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # 从 utils 导入需要的函数
    from utils import load_json, find_first_image_path, load_pil_image # 使用 utils 中的版本
    print("[INFO] image_encoder: 成功从父目录 utils.py 导入共享函数。")
except ImportError as e:
    print(f"导入错误: {e}")
    print("错误: 无法从父目录 (preprocessing/) 导入 utils.py。请检查文件结构。")


def generate_clip_image(args, id2item: Dict[str, str], images_info: Dict[str, List[str]], image_dir: str) -> np.ndarray:
    """
    使用指定的 CLIP 模型提取图像嵌入。
    (函数体保持不变，因为它已经依赖于从 utils 导入的函数)
    """
    print(f"🔹 使用 CLIP 模型生成图像嵌入: {args.clip_model_name}")
    device = args.device # device 已在 main_generate.py 中设置

    # --- 1. 加载 CLIP 模型和处理器 ---
    print(f'加载 Hugging Face CLIP 模型: {args.clip_model_name} ...')
    try:
        # 使用 model_cache_dir 参数
        processor = CLIPProcessor.from_pretrained(args.clip_model_name, cache_dir=args.model_cache_dir)
        model = CLIPModel.from_pretrained(args.clip_model_name, cache_dir=args.model_cache_dir).to(device)
        model.eval()
        # 尝试更安全地获取维度
        embedding_dim = getattr(getattr(model, 'config', None), 'projection_dim', 512) # 提供默认值
        print(f"模型加载成功，嵌入维度: {embedding_dim}")
    except Exception as e:
        print(f"加载 CLIP 模型或预处理器失败: {e}")
        raise # 重新抛出异常

    # --- 2. 准备图像数据 (按新ID顺序) ---
    sorted_new_ids = sorted(id2item.keys(), key=int)
    all_pil_images = []
    print("准备图像数据 (加载 PIL 对象)...")
    load_errors = 0
    for mapped_id_str in tqdm(sorted_new_ids, desc="查找并加载图像"):
        original_item_id = id2item.get(mapped_id_str)
        img_path = None
        if original_item_id and image_dir: # 确保 image_dir 有效
            # 使用 utils.find_first_image_path
            img_path = find_first_image_path(original_item_id, images_info, image_dir)
        
        # 使用 utils.load_pil_image
        pil_img = load_pil_image(img_path)
        if img_path and pil_img is None:
             load_errors += 1
        all_pil_images.append(pil_img)
        
    if load_errors > 0:
         print(f"[警告] {load_errors} 个图像文件无法加载。")

    # --- 3. 分批次提取特征 ---
    embeddings = []
    total_items = len(sorted_new_ids)
    print(f"开始使用批次大小 {args.batch_size} 提取图像特征...")

    with torch.no_grad():
        for i in tqdm(range(0, total_items, args.batch_size), desc="CLIP Image Encoding"):
            batch_images = all_pil_images[i : i + args.batch_size]
            
            processed_batch = []
            for img in batch_images:
                if img is None:
                    try: # 尝试获取 processor 定义的尺寸
                        if hasattr(processor, 'image_processor') and hasattr(processor.image_processor, 'size'):
                            size_info = processor.image_processor.size
                            if isinstance(size_info, dict): # 处理 ViT-L/14@336px 等情况
                                img_size = size_info.get('shortest_edge', size_info.get('height', 224))
                            else: # 处理整数情况
                                img_size = int(size_info)
                        else: # 回退到默认
                            img_size = 224
                    except:
                        img_size = 224 # 最终回退
                        
                    processed_batch.append(Image.new("RGB", (img_size, img_size), color=(0, 0, 0)))
                else:
                    processed_batch.append(img)
            
            try:
                # 添加 error handling for processor call
                inputs = processor(images=processed_batch, return_tensors="pt", padding=True).to(device) # padding=True maybe needed
                image_features = model.get_image_features(**inputs)
                
                # L2 归一化
                image_features = F.normalize(image_features, p=2, dim=-1)
                
                embeddings.append(image_features.cpu())
            except Exception as e:
                print(f"\n[警告] CLIP 图像编码批次 {i//args.batch_size} 失败: {e}")
                embeddings.append(torch.zeros((len(batch_images), embedding_dim)))

    if not embeddings: # 处理完全失败的情况
        raise RuntimeError("未能生成任何 CLIP 图像嵌入。")

    final_embeddings = torch.cat(embeddings, dim=0).numpy().astype(np.float32)

    # 验证数量 (与 text_encoder 保持一致)
    if final_embeddings.shape[0] != total_items:
         print(f"[警告] 输出嵌入数量 ({final_embeddings.shape[0]}) 与物品数量 ({total_items}) 不符！")
         target_len = total_items
         current_len = final_embeddings.shape[0]
         emb_dim = final_embeddings.shape[1]
         if current_len < target_len:
              print(" -> 将用零向量填充。")
              padding = np.zeros((target_len - current_len, emb_dim), dtype=np.float32)
              final_embeddings = np.concatenate([final_embeddings, padding], axis=0)
         else:
              print(" -> 将截断多余部分。")
              final_embeddings = final_embeddings[:target_len]
         
    print(f"CLIP 图像嵌入维度: {final_embeddings.shape}")
    return final_embeddings

# 🚨 (移除) 不再需要 main 函数或 argparse，因为这个文件现在是模块
# if __name__ == "__main__":
#     args = ...
#     generate_clip_image(...)