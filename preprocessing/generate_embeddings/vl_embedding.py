# preprocessing/generate_embeddings/vl_embedding.py
"""
使用 VLM (視覺語言模型) 提取深度融合的多模態 Embedding。
無需訓練，直接利用預訓練 VLM 的能力。
"""

import os
import sys
import json
import argparse
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from typing import List, Dict, Any

import torch
# 🚨 移除 nn, functional, DataLoader, Dataset 等訓練相關導入

from sklearn.decomposition import PCA
# ✅ 導入 VLM 相關庫
from transformers import AutoProcessor, AutoModelForCausalLM # 使用 AutoModel 更通用
# from transformers import Qwen3VLForConditionalGeneration # 或者直接指定 Qwen


# ----------------- utils (保持不變) -----------------
def load_json(p):
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERR] load_json({p}): {e}")
        return None

def get_id2item_dict(item2id_file):
    if not os.path.exists(item2id_file):
        raise FileNotFoundError(f"item2id not found: {item2id_file}")
    id2item = {}
    with open(item2id_file, "r") as fp:
        for line in fp:
            try:
                item, item_id = line.strip().split("\t")
                id2item[item_id] = item
            except ValueError:
                continue
    if not id2item:
        raise RuntimeError("id2item is empty.")
    return id2item

def build_text_map(args, id2item):
    item_json = os.path.join(args.save_root, args.dataset, f"{args.dataset}.item.json")
    data = load_json(item_json)
    if not isinstance(data, dict):
        raise FileNotFoundError(f"Invalid item.json: {item_json}")

    if args.dataset_type == "amazon":
        fields = ["title", "description", "brand", "categories"]
    elif args.dataset_type == "movielens":
        fields = ["title", "description", "genres"]
    else:
        raise ValueError("--dataset_type must be amazon or movielens")

    text_map = {}
    for _, orig_id in id2item.items():
        v = data.get(orig_id, {})
        parts = []
        for f in fields:
            if f in v:
                val = v[f]
                if isinstance(val, list):
                    val = " ".join(str(x) for x in val)
                parts.append(str(val))
        text = " ".join(p.strip() for p in parts if str(p).strip())
        text_map[orig_id] = text if text.strip() else "N/A"
    return text_map

def find_first_image_path(original_item_id, images_info, image_dir):
    """(保持不變) 為給定 item 查找第一個存在的圖像文件路徑"""
    names = images_info.get(original_item_id, [])
    if not isinstance(names, list):
        names = []
    for name in names:
        if not isinstance(name, str) or not name:
            continue
        fp = os.path.join(image_dir, name)
        if os.path.exists(fp):
            return fp
    return None

def load_pil_image(img_path):
    """安全地加載 PIL 圖像，失敗時返回 None"""
    if img_path is None:
        return None
    try:
        # 轉換為 RGB 以確保通道數一致
        return Image.open(img_path).convert("RGB") 
    except (UnidentifiedImageError, FileNotFoundError, Exception) as e:
        # print(f"[警告] 無法加載圖像 {img_path}: {e}") # 可能打印過多
        return None

# ----------------- VLM 特徵提取核心函數 -----------------
def vlm_encode_batch(
    processor, 
    model, 
    texts: List[str], 
    images: List[Any], # List[PIL.Image or None]
    device: torch.device, 
    prompt_template: str = "Describe this item for recommendation: {}"
) -> np.ndarray:
    """
    使用 VLM 處理一批文本和圖像，提取融合後的 Embedding。
    
    Args:
        processor: VLM 的處理器 (AutoProcessor)
        model: VLM 模型 (AutoModelForCausalLM)
        texts: 批次的文本列表
        images: 批次的 PIL.Image 對象列表 (缺失圖像用 None 表示)
        device: 模型所在的設備
        prompt_template: 用於包裝文本的模板 (可選)

    Returns:
        np.ndarray: 融合後的 Embedding 數組 (batch_size, hidden_dim)
    """
    
    # 檢查輸入長度是否一致
    if len(texts) != len(images):
        raise ValueError(f"文本 ({len(texts)}) 和圖像 ({len(images)}) 的數量不匹配")
        
    batch_size = len(texts)
    
    # --- 準備 VLM 輸入 ---
    # 大多數 VLM 處理器接受文本列表和 PIL 圖像列表
    # 我們需要為每個樣本構建輸入，可能包含圖像或不包含
    # 為了簡化批處理，我們創建兩個列表：一個包含所有文本（帶 Prompt），一個包含所有圖像（None 表示缺失）
    
    processed_texts = [prompt_template.format(t if t and t.strip() else "N/A") for t in texts]
    # 對於沒有圖像的樣本，傳遞 None 給 processor 通常可行
    pil_images = images # images 列表已經是 PIL 或 None
    
    try:
        # 使用 processor 進行預處理
        # padding=True, return_tensors="pt" 是標準操作
        # truncation=True 也是必要的，但 max_length 取決於模型，處理器通常有默認值
        inputs = processor(
            text=processed_texts, 
            images=pil_images, 
            return_tensors="pt", 
            padding=True, 
            truncation=True 
            # max_length=... # 可以考慮設置一個最大長度
        ).to(device)
    except Exception as e:
        print(f"\n[錯誤] VLM Processor 處理失敗: {e}")
        # 返回零向量作為錯誤處理
        hidden_dim = model.config.hidden_size # 嘗試獲取隱藏層維度
        return np.zeros((batch_size, hidden_dim if hidden_dim else 4096), dtype=np.float32)

    # --- 執行前向傳播並提取隱藏狀態 ---
    with torch.no_grad():
        try:
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden_states = outputs.hidden_states[-1] # (batch_size, seq_len, hidden_dim)
            
            # 策略：取最後一個 token 的隱藏狀態
            # 注意：這裡假設最後一個 token 的 embedding 最能代表整體
            fused_embeddings = last_hidden_states[:, -1, :] # (batch_size, hidden_dim)
            
            return fused_embeddings.cpu().numpy().astype(np.float32)
            
        except Exception as e:
            print(f"\n[錯誤] VLM Forward pass 或隱藏狀態提取失敗: {e}")
            hidden_dim = model.config.hidden_size
            return np.zeros((batch_size, hidden_dim if hidden_dim else 4096), dtype=np.float32)

# ----------------- VLM 特徵提取核心函數 -----------------
def vlm_encode_one(processor, model, text, image, device, prompt_template):
    """
    单样本编码：用 chat_template 生成字符串，再由 processor 打包成 mapping，避免 **Tensor 错传。
    """
    # 1) 组装 messages —— 有图就放 image，没有图就只放 text；不要手写 "<image>"。
    msg_content = []
    if image is not None:
        msg_content.append({"type": "image", "image": image})
    msg_content.append({"type": "text", "text": prompt_template.format(text if text and text.strip() else "N/A")})

    messages = [{"role": "user", "content": msg_content}]

    # 2) 先用模板生成“字符串”，绝对不要 tokenize=True / return_tensors="pt"
    chat_str = processor.apply_chat_template(
        messages,
        tokenize=False,              # 关键：返回字符串而不是 tensor
        add_generation_prompt=False  # 我们只取 hidden states，不需要生成提示
    )

    # 3) 再用 processor 打包成“字典”（mapping）：text + image -> BatchFeature(dict)
    inputs = processor(
        text=[chat_str],             # 列表形式，保持 batch 维
        images=[image] if image is not None else None,
        padding=True,
        return_tensors="pt"
    ).to(device)

    # 4) 前向并取隐藏态
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # (1) 先做池化
        last_hidden = outputs.hidden_states[-1].mean(dim=1)   # [1, H]，可能是 bf16
        # (2) 显式转到 float32 再搬到 CPU、再转 numpy
        return last_hidden.to(dtype=torch.float32).squeeze(0).cpu().numpy()



def vlm_encode_batch(processor, model, texts, images, device, prompt_template):
    """
    小批量封装：内部逐样本调用，屏蔽“有图/无图混批”带来的 tokens/features 不匹配与 **Tensor 错传。
    """
    assert len(texts) == len(images), "文本与图像数量不一致"
    embs = []
    for t, img in zip(texts, images):
        emb = vlm_encode_one(processor, model, t, img, device, prompt_template)
        embs.append(emb)
    return np.stack(embs, axis=0)  # (batch, hidden_dim)


# ----------------- 主提取與導出邏輯 -----------------
def extract_vlm_embeddings(args):
    """主函數：加載數據、模型，提取並導出 VLM Embedding"""
    
    # --- 1. 加載數據 ---
    processed_data_path = os.path.join(args.save_root, args.dataset)
    item2id_file = os.path.join(processed_data_path, f"{args.dataset}.item2id")
    id2item = get_id2item_dict(item2id_file)

    image_base_path = os.path.join(args.image_root, f"amazon{args.data_version}", "Images")
    image_dir = os.path.join(image_base_path, args.dataset)
    images_info_path = os.path.join(image_base_path, f"{args.dataset}_images_info.json")
    images_info = load_json(images_info_path) or {}

    text_map = build_text_map(args, id2item)

    # --- 2. 加載 VLM 模型和處理器 ---
    device = torch.device(args.device)
    print(f"[INFO] 正在加載 VLM 模型: {args.vlm_model_name_or_path} 到 {device}")
    try:
        # 嘗試使用 AutoModelForCausalLM，它適用於很多 Decoder-only 的 VLM
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.vlm_model_name_or_path,
            dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(args.vlm_model_name_or_path, trust_remote_code=True)

        model.eval() # 設置為評估模式
        print("[INFO] VLM 模型加載成功。")
    except Exception as e:
        print(f"[錯誤] 加載 VLM 模型或處理器失敗: {e}")
        print("請確保模型名稱正確，且已安裝所有必要的依賴庫 (可能需要 'pip install accelerate bitsandbytes')。")
        sys.exit(1)

    # --- 3. 準備所有物品的文本和圖像 ---
    sorted_ids = sorted(id2item.keys(), key=int)
    texts_all: List[str] = []
    images_all: List[Any] = [] # 存儲 PIL Image 或 None
    
    print("[INFO] 準備所有物品的文本和圖像...")
    for mapped_id in tqdm(sorted_ids, desc="準備數據"):
        orig_id = id2item[mapped_id]
        texts_all.append(text_map.get(orig_id, "N/A"))
        img_path = find_first_image_path(orig_id, images_info, image_dir)
        images_all.append(load_pil_image(img_path)) # 加載 PIL 圖像

    # --- 4. 分批次提取 Embedding ---
    all_embeddings = []
    total_items = len(sorted_ids)
    
    print(f"[INFO] 開始使用批次大小 {args.batch_size} 提取 Embedding...")
    for i in tqdm(range(0, total_items, args.batch_size), desc="VLM Encoding"):
        batch_texts = texts_all[i : i + args.batch_size]
        batch_images = images_all[i : i + args.batch_size]
        
        batch_embeddings = vlm_encode_batch(
            processor, model, batch_texts, batch_images, device, args.prompt_template
        )

        all_embeddings.append(batch_embeddings)

    # --- 5. 匯總並保存 ---
    if not all_embeddings:
        print("[錯誤] 未能生成任何 Embedding。")
        return

    final_embeddings_np = np.concatenate(all_embeddings, axis=0)
    
    # 驗證數量是否匹配
    if final_embeddings_np.shape[0] != total_items:
         print(f"[警告] 最終 Embedding 數量 ({final_embeddings_np.shape[0]}) 與物品總數 ({total_items}) 不匹配！")
         # 可以考慮填充或截斷，但這裡先打印警告
         final_embeddings_np = final_embeddings_np[:total_items] # 嘗試截斷以匹配

    out_dir = os.path.join(args.save_root, args.dataset, "embeddings")
    os.makedirs(out_dir, exist_ok=True)
    
    # 文件名包含 VLM 模型名稱
    vlm_tag = args.vlm_model_name_or_path.split("/")[-1].replace("/", "-")
    out_base_name = f"{args.dataset}.emb-{args.export_tag}-{vlm_tag}.npy"
    out_base_path = os.path.join(out_dir, out_base_name)
    
    np.save(out_base_path, final_embeddings_np)
    print(f"\n[OK] saved fused VLM embeddings: {out_base_path}  shape={final_embeddings_np.shape}")

    # --- 6. (可選) PCA ---
    if args.pca_dim > 0 and args.pca_dim < final_embeddings_np.shape[1]:
        print(f"[INFO] Performing PCA -> {args.pca_dim} (whiten={args.whiten})")
        try:
            pca = PCA(n_components=args.pca_dim, whiten=args.whiten, svd_solver="auto", random_state=42)
            Zp = pca.fit_transform(final_embeddings_np).astype(np.float32)
            
            out_pca_name = f"{args.dataset}.emb-{args.export_tag}-{vlm_tag}-pca{args.pca_dim}.npy"
            out_pca_path = os.path.join(out_dir, out_pca_name)
            np.save(out_pca_path, Zp)
            print(f"[OK] saved PCA embeddings: {out_pca_path}  shape={Zp.shape}  explained_variance={pca.explained_variance_ratio_.sum():.4f}")
        except Exception as e:
            print(f"[錯誤] PCA 失敗: {e}")

# ----------------- argparser -----------------
def build_parser():
    ap = argparse.ArgumentParser("Extract Deep Fused Embeddings using VLM (No Training)")
    
    # --- 數據路徑參數 (與之前類似) ---
    ap.add_argument("--data_version", type=str, default="14", choices=["14","18"])
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--dataset_type", type=str, default="amazon", choices=["amazon","movielens"])
    ap.add_argument("--image_root", type=str, default="../datasets")
    ap.add_argument("--save_root", type=str, default="../datasets")

    # --- VLM 模型參數 ---
    ap.add_argument("--vlm_model_name_or_path", type=str, required=True, 
                        default="Qwen/Qwen3-VL-32B-Instruct",
                        help="要使用的 VLM 模型 Hugging Face ID 或本地路徑 (e.g., 'Qwen/Qwen3-VL-7B-Instruct', 'llava-hf/llava-1.5-7b-hf')")
    ap.add_argument("--model_cache_dir", type=str, default=None, help="Hugging Face 模型緩存目錄 (可選)")
    ap.add_argument("--prompt_template", type=str, default="Represent this item for recommendation: {}", 
                        help="用於包裝物品文本的 Prompt 模板")

    # --- 提取與導出參數 ---
    ap.add_argument("--batch_size", type=int, default=16, help="VLM 推理的批次大小 (根據顯存調整)")
    ap.add_argument("--export_tag", type=str, default="vlm-fused", help="輸出文件名中的標籤")
    ap.add_argument("--export_bs", type=int, default=1024, help="(此參數在此腳本中未使用，保留以兼容)") # 保留以防混淆
    ap.add_argument("--pca_dim", type=int, default=0, help="PCA 降維維度 (0表示不降維)")
    ap.add_argument("--whiten", action="store_true", help="PCA 時是否白化")

    # --- 設備 ---
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # 🚨 移除所有訓練相關參數 (epochs, lr, temperature, dropout 等)
    
    return ap


def main():
    args = build_parser().parse_args()
    print(f"[CFG] VLM Model: {args.vlm_model_name_or_path}, Device: {args.device}")
    extract_vlm_embeddings(args)


if __name__ == "__main__":
    """
    使用示例:
    
    python vl_embedding.py \
        --dataset Baby \
        --vlm_model_name_or_path Qwen/Qwen3-VL-7B-Instruct \
        --batch_size 32 \
        --export_tag qwen7b \
        --pca_dim 512 
        # --model_cache_dir /path/to/cache # (可選)
        # --device cuda:0 # (可選)
    """
    main()