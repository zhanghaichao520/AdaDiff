#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-pass fusion extractor:
1) 用同一 CLIP 模型抽取文本/图像嵌入（对齐空间）
2) 分别 L2 归一化（可选）后拼接
3) PCA 压到 512 维
4) 保存融合模态 .npy

目录假设：
- item2id:        <save_root>/<dataset>/<dataset>.item2id
- item.json:      <save_root>/<dataset>/<dataset>.item.json
- images_info:    <image_root>/amazon<data_version>/Images/<dataset>_images_info.json
- image folder:   <image_root>/amazon<data_version>/Images/<dataset>/
- 输出：          <save_root>/<dataset>/embeddings/<dataset>.emb-fused-<model>-pca512.npy
"""

import os
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError
from sklearn.decomposition import PCA
from transformers import CLIPProcessor, CLIPModel

# ----------------- utils -----------------
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

def l2_normalize_np(x: np.ndarray, eps=1e-12):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)

def l2_normalize_t(x: torch.Tensor, eps=1e-12):
    return x / (x.norm(dim=-1, keepdim=True) + eps)

# ----------------- extraction -----------------
def extract_image_embeddings(args, processor, model, id2item, emb_dim):
    image_base_path = os.path.join(args.image_root, f"amazon{args.data_version}", "Images")
    image_dir = os.path.join(image_base_path, args.dataset)
    images_info_path = os.path.join(image_base_path, f"{args.dataset}_images_info.json")
    images_info = load_json(images_info_path) or {}

    embeddings = []
    sorted_ids = sorted(id2item.keys(), key=int)

    model.eval()
    with torch.no_grad():
        for mapped_id_str in tqdm(sorted_ids, desc="Image feats"):
            original_item_id = id2item.get(mapped_id_str, None)
            names = images_info.get(original_item_id, []) if original_item_id else []
            if not isinstance(names, list):
                names = []

            feat = torch.zeros(emb_dim)
            for name in names:
                if not isinstance(name, str) or not name:
                    continue
                fp = os.path.join(image_dir, name)
                if not os.path.exists(fp):
                    continue
                try:
                    img = Image.open(fp).convert("RGB")
                    inputs = processor(images=img, return_tensors="pt").to(args.device)
                    f = model.get_image_features(**inputs)  # (1, D)
                    f = l2_normalize_t(f) if args.norm_each else f
                    feat = f[0].cpu()
                    break  # 取第一张可用图
                except UnidentifiedImageError:
                    continue
                except Exception:
                    continue
            embeddings.append(feat)

    img_emb = torch.stack(embeddings, dim=0).numpy().astype(np.float32)
    if args.norm_each:
        # 已在 torch 端归一化，这里可省；保守再归一次无害
        img_emb = l2_normalize_np(img_emb)
    return img_emb

def extract_text_embeddings(args, processor, model, id2item):
    item_json = os.path.join(args.save_root, args.dataset, f"{args.dataset}.item.json")
    data = load_json(item_json)
    if not isinstance(data, dict):
        raise FileNotFoundError(f"Invalid item.json: {item_json}")

    # 字段选择
    if args.dataset_type == "amazon":
        fields = ["title", "description", "brand", "categories"]
    elif args.dataset_type == "movielens":
        fields = ["title", "description", "genres"]
    else:
        raise ValueError("--dataset_type must be amazon or movielens")

    # 保证与 id2item 顺序一致
    sorted_ids = sorted(id2item.keys(), key=int)
    texts = []
    for mapped_id_str in sorted_ids:
        original_item_id = id2item[mapped_id_str]
        v = data.get(original_item_id, {})
        parts = []
        for f in fields:
            if f in v:
                val = v[f]
                if isinstance(val, list):
                    val = " ".join(str(x) for x in val)
                parts.append(str(val))
        txt = " ".join(p.strip() for p in parts if str(p).strip())
        texts.append(txt if txt.strip() else "N/A")

    # 批处理
    all_emb = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), args.batch_size), desc="Text feats"):
            batch = texts[i : i + args.batch_size]
            inputs = processor(text=batch, return_tensors="pt", padding=True,
                               truncation=True, max_length=args.max_sent_len).to(args.device)
            f = model.get_text_features(**inputs)  # (B, D)
            f = l2_normalize_t(f) if args.norm_each else f
            all_emb.append(f.cpu())
    txt_emb = torch.cat(all_emb, dim=0).numpy().astype(np.float32)
    if args.norm_each:
        txt_emb = l2_normalize_np(txt_emb)
    return txt_emb

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser("CLIP dual-modality fusion (concat + PCA to 512)")
    ap.add_argument("--data_version", type=str, default="14", choices=["14","18"])
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--dataset_type", type=str, default="amazon", choices=["amazon","movielens"])
    ap.add_argument("--image_root", type=str, default="../datasets")
    ap.add_argument("--save_root", type=str, default="../datasets")
    ap.add_argument("--model_name_or_path", type=str, default="openai/clip-vit-base-patch32")
    ap.add_argument("--model_cache_dir", type=str, default=None)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--max_sent_len", type=int, default=1024)
    ap.add_argument("--pca_dim", type=int, default=512)
    ap.add_argument("--whiten", action="store_true")
    ap.add_argument("--norm_each", action="store_true", help="拼接前分别对 text/image L2 归一化（推荐）")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # 路径与模型
    processed_data_path = os.path.join(args.save_root, args.dataset)
    item2id_file = os.path.join(processed_data_path, f"{args.dataset}.item2id")
    id2item = get_id2item_dict(item2id_file)

    processor = CLIPProcessor.from_pretrained(args.model_name_or_path, cache_dir=args.model_cache_dir)
    model = CLIPModel.from_pretrained(args.model_name_or_path, cache_dir=args.model_cache_dir).to(args.device)
    emb_dim = model.config.projection_dim
    print(f"[CLIP] model={args.model_name_or_path}  proj_dim={emb_dim}  device={args.device}")

    # 提取两路
    txt = extract_text_embeddings(args, processor, model, id2item)   # (N, D)
    img = extract_image_embeddings(args, processor, model, id2item, emb_dim)  # (N, D)

    if txt.shape[0] != img.shape[0]:
        raise ValueError(f"N mismatch: text={txt.shape}, image={img.shape}")

    # 拼接
    X = np.concatenate([txt, img], axis=1).astype(np.float32)
    print(f"[INFO] concat shape = {X.shape} (= (N, Dt+Di))")

    # PCA
    if args.pca_dim > 0 and args.pca_dim < X.shape[1]:
        print(f"[INFO] PCA -> {args.pca_dim} (whiten={args.whiten})")
        pca = PCA(n_components=args.pca_dim, whiten=args.whiten, svd_solver="auto", random_state=42)
        Z = pca.fit_transform(X).astype(np.float32)
        print(f"[INFO] explained variance ratio = {pca.explained_variance_ratio_.sum():.4f}")
    else:
        print("[WARN] pca_dim 无效或 >= 输入维度，跳过 PCA")
        Z = X

    # 保存
    hf_tag = args.model_name_or_path.split("/")[-1].replace("/","-")
    save_dir = os.path.join(args.save_root, args.dataset, "embeddings")
    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, f"{args.dataset}.emb-fused-{hf_tag}.npy")
    np.save(out, Z.astype(np.float32))
    print(f"[OK] saved: {out}  shape={Z.shape}  dtype=float32")

if __name__ == "__main__":
    main()
