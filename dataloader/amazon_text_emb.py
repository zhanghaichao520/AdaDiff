import argparse
import collections
import gzip
import html
import json
import os
import random
import re
import torch
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA # <-- 新增导入
from utils import *
from transformers import AutoTokenizer, AutoModel

# load_data, generate_text, preprocess_text 函数保持不变
def load_data(args):
    item2feature_path = os.path.join(args.root, f'{args.dataset}.item.json')
    item2feature = load_json(item2feature_path)
    return item2feature

def generate_text(item2feature, features):
    item_text_list = []
    for item in item2feature:
        data = item2feature[item]
        text = []
        for meta_key in features:
            if meta_key in data:
                meta_value = clean_text(data[meta_key])
                text.append(meta_value.strip())
        item_text_list.append([int(item), text])
    return item_text_list

def preprocess_text(args):
    print('处理文本数据: ')
    print(' 数据集: ', args.dataset)
    item2feature = load_data(args)
    item_text_list = generate_text(item2feature, ['title', 'description'])
    return item_text_list

# generate_item_embedding 函数稍作修改，现在会返回生成的嵌入
def generate_item_embedding(args, item_text_list, tokenizer, model, word_drop_ratio=-1):
    """
    生成原始的高维文本嵌入。
    """
    print(f'生成文本嵌入: ')
    print(' 数据集: ', args.dataset)

    items, texts = zip(*item_text_list)
    order_texts = [[0]] * len(items)
    for item, text in zip(items, texts):
        order_texts[item] = text
    for text in order_texts:
        assert text != [0]

    embeddings = []
    start, batch_size = 0, 1
    with torch.no_grad():
        # 使用 tqdm 替代手动打印进度
        pbar = tqdm(total=len(order_texts), desc="生成嵌入", ncols=100)
        while start < len(order_texts):
            field_texts = order_texts[start: start + batch_size]
            field_texts = zip(*field_texts)
    
            field_embeddings = []
            for sentences in field_texts:
                sentences = list(sentences)
                if word_drop_ratio > 0:
                    new_sentences = []
                    for sent in sentences:
                        new_sent = []
                        sent = sent.split(' ')
                        for wd in sent:
                            if random.random() > word_drop_ratio:
                                new_sent.append(wd)
                        new_sent = ' '.join(new_sent)
                        new_sentences.append(new_sent)
                    sentences = new_sentences
                
                encoded_sentences = tokenizer(sentences, max_length=args.max_sent_len,
                                              truncation=True, return_tensors='pt', padding="longest").to(args.device)
                outputs = model(input_ids=encoded_sentences.input_ids,
                                attention_mask=encoded_sentences.attention_mask)
    
                masked_output = outputs.last_hidden_state * encoded_sentences['attention_mask'].unsqueeze(-1)
                mean_output = masked_output.sum(dim=1) / encoded_sentences['attention_mask'].sum(dim=-1, keepdim=True)
                mean_output = mean_output.detach().cpu()
                field_embeddings.append(mean_output)
    
            field_mean_embedding = torch.stack(field_embeddings, dim=0).mean(dim=0)
            embeddings.append(field_mean_embedding)
            start += batch_size
            pbar.update(batch_size)
        pbar.close()

    embeddings = torch.cat(embeddings, dim=0).numpy()
    print('原始嵌入维度: ', embeddings.shape)

    # 保存原始高维嵌入
    file = os.path.join(args.root, args.dataset + '.emb-' + args.plm_name + "-td" + ".npy")
    np.save(file, embeddings)
    print(f'成功保存原始嵌入到: {file}')
    
    # 返回生成的嵌入，以便后续处理
    return embeddings

# --- 新增函数: 应用PCA并保存 ---
def apply_pca_and_save(original_embeddings, args):
    """
    对给定的嵌入应用PCA降维并保存结果。
    """
    if args.pca_dim <= 0:
        print("pca_dim <= 0, 跳过PCA降维。")
        return

    target_dim = args.pca_dim
    print(f"\n应用PCA降维，目标维度: {target_dim}...")
    
    if original_embeddings.shape[1] < target_dim:
        print(f"警告: 原始维度 ({original_embeddings.shape[1]}) 小于目标维度 ({target_dim})。跳过PCA。")
        return

    # 初始化并执行PCA
    pca = PCA(n_components=target_dim)
    reduced_embeddings = pca.fit_transform(original_embeddings)
    
    print(f"降维前维度: {original_embeddings.shape}")
    print(f"降维后维度: {reduced_embeddings.shape}")
    print(f"保留的方差比例: {sum(pca.explained_variance_ratio_):.4f}")

    # 构建新的文件名并保存
    pca_filename = f"{args.dataset}.emb-td.npy"
    save_path = os.path.join(args.root, pca_filename)
    np.save(save_path, reduced_embeddings)
    print(f'成功保存PCA降维后的嵌入到: {save_path}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Instruments', help='Instruments / Arts / Games')
    parser.add_argument('--root', type=str, default="../datasets")
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of running GPU')
    parser.add_argument('--plm_name', type=str, default='llama')
    parser.add_argument('--model_name_or_path', type=str, default='huggyllama/llama-7b')
    parser.add_argument('--model_cache_dir', type=str, default='/userhome/cache_models')
    parser.add_argument('--max_sent_len', type=int, default=2048)
    parser.add_argument('--word_drop_ratio', type=float, default=-1, help='word drop ratio, do not drop by default')
    
    # --- 新增PCA参数 ---
    parser.add_argument('--pca_dim', type=int, default=512, 
                        help='PCA降维的目标维度。默认512。设置为0或负数则不进行PCA。')
                        
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    args.root = os.path.join(args.root, args.dataset)
    os.makedirs(args.root, exist_ok=True) # 确保目录存在

    device = set_device(args.gpu_id)
    args.device = device
    
    # --- 智能加载/生成逻辑 ---
    original_emb_path = os.path.join(args.root, f"{args.dataset}.emb-{args.plm_name}-td.npy")
    
    if os.path.exists(original_emb_path):
        print(f"发现已存在的原始嵌入文件，直接加载: {original_emb_path}")
        full_embeddings = np.load(original_emb_path)
        print(f"加载完成，维度: {full_embeddings.shape}")
    else:
        print(f"未找到原始嵌入文件，开始生成...")
        item_text_list = preprocess_text(args)

        kwargs = {"cache_dir": args.model_cache_dir, "local_files_only": os.path.exists(args.model_cache_dir)}
        
        # 注意: LlamaForCausalLM 可能不是最适合提取嵌入的，通常用 AutoModel
        # 为了与您原代码保持一致，这里暂时保留。如果遇到问题，可以切换到 AutoModel.from_pretrained
        print(f"加载PLM: {args.model_name_or_path}")
        plm_tokenizer, plm_model = load_plm(args.model_name_or_path, kwargs)
        if plm_tokenizer.pad_token_id is None:
            plm_tokenizer.pad_token_id = 0
        plm_model = plm_model.to(device)

        full_embeddings = generate_item_embedding(args, item_text_list, plm_tokenizer,
                                                  plm_model, word_drop_ratio=args.word_drop_ratio)

    # --- 对获取到的高维嵌入执行PCA ---
    if full_embeddings is not None:
        apply_pca_and_save(full_embeddings, args)

    print("\n所有任务完成。")