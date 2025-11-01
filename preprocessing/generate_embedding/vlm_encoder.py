# preprocessing/generate_embeddings/vlm_encoder.py

import os
import sys
import torch
from PIL import Image, UnidentifiedImageError
import numpy as np
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM 
from typing import List, Dict, Any

# ✅ (核心修改) 从父目录导入共享函数
try:
    # 添加父目录 (preprocessing/) 到 Python 路径
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import load_json, find_first_image_path, load_pil_image # 使用 utils 中的版本
    print("[INFO] vlm_encoder: 成功从父目录 utils.py 导入共享函数。")
except ImportError as e:
    print(f"导入错误: {e}")
    print("错误: 无法从父目录 (preprocessing/) 导入 utils.py。请检查文件结构。")
    sys.exit(1)

# 🚨 (移除) 不再需要导入或定义 common_utils 中的函数
# try:
#     from .common_utils import load_json
# except ImportError: ...

# 🚨 (移除) 不再需要在这里定义 load_pil_image 和 find_first_image_path

# =================================================================
# ================== VLM 特徵提取核心函數 (保持不变) ==================
# =================================================================

def vlm_encode_batch(
    processor, 
    model, 
    texts: List[str], 
    images: List[Any], 
    device: torch.device, 
    prompt_template: str = "Represent this item for recommendation: {}"
) -> np.ndarray:
    """
    使用 VLM 處理一批文本和圖像，提取融合後的 Embedding (最後 token)。
    (函数体保持不变)
    """
    if len(texts) != len(images):
        raise ValueError(f"文本 ({len(texts)}) 和圖像 ({len(images)}) 的數量不匹配")
        
    batch_size = len(texts)
    # 尝试更安全地获取 hidden_dim
    hidden_dim = getattr(getattr(model, 'config', None), 'hidden_size', 4096) # 提供默认值
    
    processed_texts = [prompt_template.format(t if t and t.strip() else "N/A") for t in texts]
    pil_images = images 
    
    try:
        # 添加 padding=True, truncation=True
        inputs = processor(
            text=processed_texts, 
            images=pil_images, 
            return_tensors="pt", 
            padding=True, # 确保批处理长度一致
            truncation=True # 确保不超过模型最大长度
        ).to(device)
    except Exception as e:
        print(f"\n[错误] VLM Processor 失败: {e}")
        return np.zeros((batch_size, hidden_dim), dtype=np.float32)

    with torch.no_grad():
        try:
            # 确保模型在 eval 模式
            model.eval() 
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden_states = outputs.hidden_states[-1]
            # 检查 hidden_states 是否有效
            if last_hidden_states is None or last_hidden_states.numel() == 0:
                 raise ValueError("模型输出了空的 hidden_states。")
                 
            fused_embeddings = last_hidden_states[:, -1, :]
            # L2 归一化 (可选但推荐)
            fused_embeddings = F.normalize(fused_embeddings, p=2, dim=-1) 
            
            return fused_embeddings.cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"\n[错误] VLM Forward pass 或隐藏状态提取失败: {e}")
            return np.zeros((batch_size, hidden_dim), dtype=np.float32)

# =================================================================
# ================== 主提取函數 (保持不变) ==================
# =================================================================

def generate_vlm_fused(args, id2item: Dict[str, str], text_map: Dict[str, str], images_info: Dict[str, List[str]], image_dir: str) -> np.ndarray:
    """
    使用 VLM 提取深度融合的多模態 Embedding。
    (函数体保持不变，因为它已经依赖于从 utils 导入的函数)
    """
    print(f"🔹 使用 VLM 模型生成融合嵌入: {args.vlm_model_name_or_path}")
    device = args.device # device 已由 main_generate 设置

    # --- 1. 加載 VLM 模型和處理器 ---
    print(f'加载 VLM 模型: {args.vlm_model_name_or_path} ...')
    try:
        # 增加 torch_dtype="auto" 以便自动使用半精度 (如果支持)
        model = AutoModelForCausalLM.from_pretrained(
            args.vlm_model_name_or_path, 
            torch_dtype="auto", # 使用 bfloat16 或 float16
            device_map="auto", # 自动 GPU 分配
            trust_remote_code=True,
            cache_dir=args.model_cache_dir,
            # (可选) 如果显存不足，尝试 4-bit 量化加载
            # load_in_4bit=True, 
            # bnb_4bit_compute_dtype=torch.bfloat16 
        )
        processor = AutoProcessor.from_pretrained(
            args.vlm_model_name_or_path,
            trust_remote_code=True,
            cache_dir=args.model_cache_dir
        )
        model.eval()
        print("VLM 模型加载成功。")
    except Exception as e:
        print(f"加载 VLM 模型或处理器失败: {e}")
        raise

    # --- 2. 准备所有物品的文本和圖像 ---
    sorted_new_ids = sorted(id2item.keys(), key=int)
    texts_all: List[str] = []
    images_all: List[Any] = [] # PIL Image or None
    
    print("准备文本和圖像数据...")
    load_errors = 0
    for mapped_id_str in tqdm(sorted_new_ids, desc="准备数据"):
        original_item_id = id2item.get(mapped_id_str)
        img_path = None # 初始化
        text = "N/A" # 默认值
        if original_item_id:
            text = text_map.get(original_item_id, "N/A")
            if image_dir and images_info: # 只有在需要图像时才查找
                # 使用 utils.find_first_image_path
                img_path = find_first_image_path(original_item_id, images_info, image_dir)
        else:
            print(f"[警告] 找不到新 ID {mapped_id_str} 对应的原始 ID！")

        texts_all.append(text)
        # 使用 utils.load_pil_image
        pil_img = load_pil_image(img_path) 
        if img_path and pil_img is None: load_errors += 1
        images_all.append(pil_img)
        
    if load_errors > 0: print(f"[警告] {load_errors} 个图像文件无法加载。")

    # --- 3. 分批次提取 Embedding ---
    all_embeddings = []
    total_items = len(sorted_new_ids)
    print(f"开始使用批次大小 {args.batch_size} 提取 VLM 嵌入...")

    # (减小 batch size 以防 OOM)
    effective_batch_size = min(args.batch_size, 16) # VLM 通常需要更小的 batch size
    if effective_batch_size != args.batch_size:
        print(f"[INFO] VLM batch size 调整为 {effective_batch_size} 以适应显存。")

    for i in tqdm(range(0, total_items, effective_batch_size), desc="VLM Encoding"):
        batch_texts = texts_all[i : i + effective_batch_size]
        batch_images = images_all[i : i + effective_batch_size]
        
        batch_embeddings = vlm_encode_batch(
            processor, model, batch_texts, batch_images, device, args.vlm_prompt_template
        )
        all_embeddings.append(batch_embeddings)
        
        # (可选) 清理 GPU 缓存
        # if device.type == 'cuda':
        #     torch.cuda.empty_cache()

    # --- 4. 汇总 ---
    if not all_embeddings:
        raise RuntimeError("未能生成任何 VLM Embedding。")

    final_embeddings_np = np.concatenate(all_embeddings, axis=0)
    
    # 验证数量 (与 text_encoder 保持一致)
    if final_embeddings_np.shape[0] != total_items:
         print(f"[警告] 输出 VLM 嵌入数量 ({final_embeddings_np.shape[0]}) 与物品数量 ({total_items}) 不符！")
         target_len = total_items
         current_len = final_embeddings_np.shape[0]
         emb_dim = final_embeddings_np.shape[1]
         if current_len < target_len:
              print(" -> 将用零向量填充。")
              padding = np.zeros((target_len - current_len, emb_dim), dtype=np.float32)
              final_embeddings_np = np.concatenate([final_embeddings_np, padding], axis=0)
         else:
              print(" -> 将截断多余部分。")
              final_embeddings_np = final_embeddings_np[:target_len]
         
    print(f"VLM 融合嵌入维度: {final_embeddings_np.shape}")
    return final_embeddings_np

# 🚨 (移除) 不再需要 main 或 argparse
# if __name__ == "__main__":
#     args = ...
#     generate_vlm_fused(...)