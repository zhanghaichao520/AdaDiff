import argparse
import os
import random
import torch
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel
import sys, os
import time

# 假设 utils.py 在上一级目录
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import load_json, clean_text, set_device
except ImportError:
    print("错误: 找不到 utils.py。请确保它在上一级目录或 Python 路径中。")
    sys.exit(1)


# =============== (共享) 数据预处理 ===============
def load_data(args):
    """(共享) 加载 .item.json 文件"""
    item2feature_path = os.path.join(args.root, f'{args.dataset}.item.json')
    return load_json(item2feature_path)


def generate_text(item2feature, features):
    """(共享) 从指定的特征列表中拼接文本"""
    item_text_list = []
    for item, data in item2feature.items():
        text = []
        for meta_key in features:
            if meta_key in data:
                meta_value = data[meta_key]
                if isinstance(meta_value, list):
                    meta_value = " ".join(meta_value)
                
                meta_value = clean_text(meta_value)
                if meta_value.strip():
                    text.append(meta_value.strip())
        
        item_text_list.append([int(item), text])
    return item_text_list


def preprocess_text(args):
    """
    (调度器) 根据 dataset_type 选择要提取的文本特征
    """
    print(f"处理文本数据: {args.dataset} (类型: {args.dataset_type})")
    item2feature = load_data(args)
    
    features = []
    if args.dataset_type == 'movielens':
        features = ['title', 'description', 'genres']
    elif args.dataset_type == 'amazon':
        features = ['title', 'description', 'brand', 'categories']
    else:
        raise ValueError(f"未知的 dataset_type: {args.dataset_type}")
        
    print(f"将使用以下元数据字段: {features}")
    return generate_text(item2feature, features)


# =============== (共享) 本地模型嵌入生成 ===============
def generate_local_embeddings(args, item_text_list):
    print(f"🔹 使用本地模型生成嵌入: {args.model_name_or_path}")
    from transformers import AutoTokenizer, AutoModel
    
    # 自动信任远程代码（适用于 Qwen 等模型）
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True).to(args.device)
    model.eval()

    items, texts = zip(*item_text_list)
    order_texts = [[0]] * len(items)
    for item, text in zip(items, texts):
        order_texts[item] = text
    for i, text in enumerate(order_texts):
        if text == [0]:
            print(f"[警告] Item {i} 缺少文本数据，将使用空字符串。")
            order_texts[i] = [""] 

    embeddings = []
    start = 0
    with torch.no_grad():
        pbar = tqdm(total=len(order_texts), desc="生成嵌入", ncols=100)
        while start < len(order_texts):
            batch_texts = order_texts[start: start + args.batch_size]
            batch_texts = [" ".join(t) if t else "" for t in batch_texts]

            encoded = tokenizer(batch_texts, padding=True, truncation=True,
                                return_tensors="pt", max_length=args.max_sent_len).to(args.device)
            outputs = model(**encoded)
            
            attn = encoded['attention_mask'].unsqueeze(-1)
            masked = outputs.last_hidden_state * attn
            mean_output = masked.sum(dim=1) / attn.sum(dim=1)
            
            embeddings.append(mean_output.cpu())
            start += args.batch_size
            pbar.update(mean_output.size(0))
        pbar.close()

    embeddings = torch.cat(embeddings, dim=0).numpy()
    return embeddings


# =============== (共享) API 模型嵌入生成 ===============
def generate_api_embeddings(args, item_text_list):
    print(f"🔹 使用 API 模型生成嵌入: {args.sent_emb_model}")
    try:
        from openai import OpenAI
    except ImportError:
        print("错误: 'openai' 库未找到。请运行: pip install openai")
        sys.exit(1)
        
    client = OpenAI(api_key=args.openai_api_key, base_url=args.openai_base_url)

    items, texts = zip(*item_text_list)
    order_texts = [[""]] * len(items)
    for item, text in zip(items, texts):
        order_texts[item] = text if text else [""]

    final_texts = [" ".join(t) for t in order_texts]

    sent_embs = []
    for i in tqdm(range(0, len(final_texts), args.batch_size), desc="API Encoding"):
        batch = final_texts[i: i + args.batch_size]
        batch = [t if t.strip() else "N/A" for t in batch]
        
        try:
            response = client.embeddings.create(
                model=args.sent_emb_model,
                input=batch
            )
            sent_embs.extend([d.embedding for d in response.data])
        except Exception as e:
            print(f"[警告] 第 {i} 批请求失败 (items {i*args.batch_size} - {(i+1)*args.batch_size})，错误：{e}")
            # 确定 API 维度 (从参数读取，如果未设置则尝试从第一个成功的 batch 推断)
            api_emb_dim = args.api_emb_dim
            if api_emb_dim <= 0 and len(sent_embs) > 0:
                 api_emb_dim = len(sent_embs[0])
            if api_emb_dim <= 0:
                 print("错误：API 维度未知且未在 --api_emb_dim 中指定。无法创建零向量。")
                 api_emb_dim = 3072 # 默认回退
                 
            sent_embs.extend([np.zeros(api_emb_dim, dtype=np.float32) for _ in batch])
            time.sleep(1)

    sent_embs = np.array(sent_embs, dtype=np.float32)
    
    # 动态设置 api_emb_dim (如果之前不知道)
    if args.api_emb_dim <= 0 and sent_embs.shape[0] > 0:
        args.api_emb_dim = sent_embs.shape[1]
        
    print(f"API 嵌入维度: {sent_embs.shape}")
    return sent_embs


# =============== (共享) PCA 降维 (恢复你原来的逻辑) ===============
def apply_pca_and_save(original_embeddings, args, save_path):
    """
    应用 PCA 并 *覆盖* 保存到原始路径 (save_path)。
    """
    if args.pca_dim <= 0:
        print("跳过 PCA 降维。")
        return

    print(f"\n应用 PCA 降维，目标维度: {args.pca_dim}")
    if original_embeddings.shape[1] < args.pca_dim:
        print(f"原始维度 ({original_embeddings.shape[1]}) 小于目标维度 ({args.pca_dim})，跳过降维。")
        return

    pca = PCA(n_components=args.pca_dim)
    reduced = pca.fit_transform(original_embeddings)
    print(f"降维后维度: {reduced.shape}，保留方差: {sum(pca.explained_variance_ratio_):.4f}")

    np.save(save_path, reduced) # <--- 覆盖原始文件
    print(f"✅ PCA 降维后嵌入已保存到: {save_path}")


# =============== (统一) 主程序入口 ===============
def parse_args():
    parser = argparse.ArgumentParser(description="从 .item.json 为 Amazon 或 MovieLens 生成文本嵌入")
    
    # --- 调度参数 (必需) ---
    parser.add_argument('--dataset_type', type=str, required=True, choices=['amazon', 'movielens'],
                        help='要处理的数据集类型 (amazon 或 movielens)')
    parser.add_argument('--dataset', type=str, required=True, 
                        help='数据集名称 (e.g., Home, Baby, ml-1m, ml-20m)')
    
    # --- 通用参数 (保留了你的所有参数) ---
    parser.add_argument('--root', type=str, default="../datasets")
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--max_sent_len', type=int, default=1024)
    parser.add_argument('--pca_dim', type=int, default=512, 
                        help="PCA 降维的目标维度。<= 0 表示不进行 PCA。")
    parser.add_argument('--mode', type=str, choices=['local', 'api'], default='local',
                        help="使用 'local' (transformers) 还是 'api' (OpenAI)")

    # --- 本地模型参数 ---
    parser.add_argument('--model_name_or_path', type=str, default='sentence-transformers/sentence-t5-base')

    # --- API 参数 ---
    parser.add_argument('--sent_emb_model', type=str, default='text-embedding-3-large',
                        help="OpenAI 模型 (e.g., text-embedding-3-large, text-embedding-3-small)")
    parser.add_argument('--api_emb_dim', type=int, default=0,
                        help="API 模型的维度 (3-large=3072, 3-small=1536)。如果为0，将自动检测，但在API请求失败时可能出错。")
    parser.add_argument('--openai_api_key', type=str, default=os.environ.get('OPENAI_API_KEY', 'sk-xxx'),
                        help="OpenAI API 密钥。默认从环境变量 OPENAI_API_KEY 读取。")
    parser.add_argument('--openai_base_url', type=str, default=os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
                        help="OpenAI API Base URL。默认从环境变量 OPENAI_BASE_URL 读取。")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # 路径设置
    args.root = os.path.join(args.root, args.dataset)
    os.makedirs(args.root, exist_ok=True)
    args.device = set_device(args.gpu_id)

    # 1. 预处理 (根据 dataset_type 调度)
    item_text_list = preprocess_text(args)
    
    # 动态设置 API 维度 (针对 text-embedding-3-large)
    if args.mode == 'api' and args.sent_emb_model == 'text-embedding-3-large' and args.api_emb_dim == 0:
        args.api_emb_dim = 3072
    if args.mode == 'api' and args.sent_emb_model == 'text-embedding-3-small' and args.api_emb_dim == 0:
        args.api_emb_dim = 1536

    # 2. 生成嵌入 (根据 mode 调度)
    emb = None
    if args.mode == "local":
        emb = generate_local_embeddings(args, item_text_list)
    elif args.mode == "api":
        emb = generate_api_embeddings(args, item_text_list)
    else:
        raise ValueError("未知模式，请选择 local 或 api")

    # --- 3. 保存 (恢复你原来的逻辑) ---
    
    # 创建独立 embedding 目录
    emb_dir = os.path.join(args.root, "embeddings")
    os.makedirs(emb_dir, exist_ok=True)

    # 定义模型标识
    model_tag = args.model_name_or_path.split('/')[-1] if args.mode == "local" else args.sent_emb_model
    model_tag = model_tag.replace('/', '-') # 移除路径斜杠

    # 3a. 定义 *最终* 路径
    save_path = os.path.join(emb_dir, f"{args.dataset}.emb-text-{model_tag}.npy")
    
    # 3b. 保存完整嵌入
    np.save(save_path, emb)
    print(f"✅ 文本嵌入已保存至: {save_path} (维度: {emb.shape})")

    # 3c. (可选) 使用 PCA 覆盖
    if args.pca_dim > 0:
        # 调用函数，传入 *相同* 的 save_path 来实现覆盖
        apply_pca_and_save(emb, args, save_path)
    else:
        print("pca_dim <= 0，跳过 PCA 降维。")
        
    print("\n🎉 所有任务完成。")