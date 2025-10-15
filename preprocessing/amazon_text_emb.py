import argparse
import os
import random
import torch
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel
from utils import load_json, clean_text, set_device


# =============== 数据预处理 ===============
def load_data(args):
    item2feature_path = os.path.join(args.root, f'{args.dataset}.item.json')
    return load_json(item2feature_path)


def generate_text(item2feature, features):
    item_text_list = []
    for item, data in item2feature.items():
        text = []
        for meta_key in features:
            if meta_key in data:
                meta_value = clean_text(data[meta_key])
                text.append(meta_value.strip())
        item_text_list.append([int(item), text])
    return item_text_list


def preprocess_text(args):
    print(f"处理文本数据: {args.dataset}")
    item2feature = load_data(args)
    return generate_text(item2feature, ['title', 'description'])


# =============== 本地模型嵌入生成 ===============
def generate_local_embeddings(args, item_text_list):
    print(f"🔹 使用本地模型生成嵌入: {args.model_name_or_path}")
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModel.from_pretrained(args.model_name_or_path).to(args.device)
    model.eval()

    items, texts = zip(*item_text_list)
    order_texts = [[0]] * len(items)
    for item, text in zip(items, texts):
        order_texts[item] = text
    for text in order_texts:
        assert text != [0]

    embeddings = []
    start = 0
    with torch.no_grad():
        pbar = tqdm(total=len(order_texts), desc="生成嵌入", ncols=100)
        while start < len(order_texts):
            batch_texts = order_texts[start: start + args.batch_size]
            batch_texts = [" ".join(t) for t in batch_texts]

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


# =============== API 模型嵌入生成 ===============
def generate_api_embeddings(args, item_text_list):
    print(f"🔹 使用 API 模型生成嵌入: {args.sent_emb_model}")
    from openai import OpenAI
    client = OpenAI(api_key=args.openai_api_key, base_url=args.openai_base_url)

    items, texts = zip(*item_text_list)
    order_texts = [" ".join(t) for _, t in item_text_list]

    sent_embs = []
    for i in tqdm(range(0, len(order_texts), args.batch_size), desc="API Encoding"):
        batch = order_texts[i: i + args.batch_size]
        try:
            response = client.embeddings.create(
                model=args.sent_emb_model,
                input=batch
            )
            sent_embs.extend([d.embedding for d in response.data])
        except Exception as e:
            print(f"[警告] 第{i}批请求失败，错误：{e}")

    sent_embs = np.array(sent_embs, dtype=np.float32)
    print(f"API 嵌入维度: {sent_embs.shape}")
    return sent_embs


# =============== PCA 降维 ===============
def apply_pca_and_save(original_embeddings, args, save_path):
    if args.pca_dim <= 0:
        print("跳过 PCA 降维。")
        return

    print(f"\n应用 PCA 降维，目标维度: {args.pca_dim}")
    if original_embeddings.shape[1] < args.pca_dim:
        print("原始维度小于目标维度，跳过降维。")
        return

    pca = PCA(n_components=args.pca_dim)
    reduced = pca.fit_transform(original_embeddings)
    print(f"降维后维度: {reduced.shape}，保留方差: {sum(pca.explained_variance_ratio_):.4f}")

    np.save(save_path, reduced)
    print(f"✅ PCA 降维后嵌入已保存到: {save_path}")



# =============== 主程序入口 ===============
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Beauty')
    parser.add_argument('--root', type=str, default="../datasets")
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--max_sent_len', type=int, default=1024)
    parser.add_argument('--pca_dim', type=int, default=512)
    parser.add_argument('--mode', type=str, choices=['local', 'api'], default='local')

    # 本地模型参数
    parser.add_argument('--model_name_or_path', type=str, default='sentence-transformers/sentence-t5-base')

    # API 参数
    parser.add_argument('--sent_emb_model', type=str, default='text-embedding-3-large')
    parser.add_argument('--openai_api_key', type=str, default='sk-xxx')
    parser.add_argument('--openai_base_url', type=str, default='https://api.openai.com/v1')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.root = os.path.join(args.root, args.dataset)
    os.makedirs(args.root, exist_ok=True)
    args.device = set_device(args.gpu_id)

    item_text_list = preprocess_text(args)

    if args.mode == "local":
        emb = generate_local_embeddings(args, item_text_list)
    elif args.mode == "api":
        emb = generate_api_embeddings(args, item_text_list)
    else:
        raise ValueError("未知模式，请选择 local 或 api")

    # 创建独立 embedding 目录
    emb_dir = os.path.join(args.root, "embeddings")
    os.makedirs(emb_dir, exist_ok=True)

    # 加上模型名标识（去掉斜杠，防止路径错误）
    model_tag = args.model_name_or_path.split('/')[-1] if args.mode == "local" else args.sent_emb_model
    model_tag = model_tag.replace('/', '-')

    save_path = os.path.join(emb_dir, f"{args.dataset}.emb-text-{model_tag}.npy")
    np.save(save_path, emb)
    print(f"✅ 文本嵌入已保存至: {save_path}")

    apply_pca_and_save(emb, args, save_path)
    print("\n🎉 所有任务完成。")
