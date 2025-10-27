#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pure text-image contrastive multimodal fusion (no collaborative/i2i).

做的事：
1) 用同一 CLIP 模型抽取文本/图像嵌入到同一空间；
2) 融合头：norm 后 concat -> 门控+残差 MLP -> 512d，并整体 L2；
3) 训练损失（仅模态相关，不含协同）：
   - text ↔ image   （双向 InfoNCE）
   - fusion ↔ text  （单向 InfoNCE）
   - fusion ↔ image （单向 InfoNCE）
4) 训练后导出融合 512d 到 .npy；可选 PCA/whiten；
5) 目录结构与原先保持一致（不再需要 .i2i.json）。

输出示例：
<save_root>/<dataset>/embeddings/<dataset>.emb-fused-<model>-forge512.npy
"""

import os
import json
import argparse
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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


# ----------------- dataset -----------------
class ItemDataset(Dataset):
    """
    提供按 id 顺序访问的 item：
    - text 字符串
    - 第一张可用图片路径（找不到则用占位黑图）
    """
    def __init__(self, id2item, text_map, image_dir, images_info):
        self.idx_list = sorted(id2item.keys(), key=int)  # mapped ids (string)
        self.id2item = id2item
        self.text_map = text_map
        self.image_dir = image_dir
        self.images_info = images_info or {}

    def __len__(self):
        return len(self.idx_list)

    def _pick_one_image(self, original_item_id):
        names = self.images_info.get(original_item_id, [])
        if not isinstance(names, list):
            names = []
        for name in names:
            if not isinstance(name, str) or not name:
                continue
            fp = os.path.join(self.image_dir, name)
            if os.path.exists(fp):
                return fp
        return None

    def __getitem__(self, idx):
        mapped_id_str = self.idx_list[idx]
        original_item_id = self.id2item[mapped_id_str]
        text = self.text_map.get(original_item_id, "N/A")
        img_path = self._pick_one_image(original_item_id)
        return {
            "mapped_id": mapped_id_str,
            "orig_id": original_item_id,
            "text": text,
            "img_path": img_path,
        }


# ----------------- fusion head -----------------
class GatedFusion(nn.Module):
    """
    稳定融合头：
    [norm(t); norm(i)] -> LN -> Linear(2D->D) -> GELU -> Linear(D->512) (残差)
    + 门控（sigmoid）在 t/i 间做可学习加权
    输出 L2 归一化的 512 维
    """
    def __init__(self, in_dim, mid_dim, out_dim=512, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

        self.gate = nn.Linear(in_dim, 2)
        self.proj_text = nn.Linear(in_dim // 2, out_dim, bias=False)
        self.proj_image = nn.Linear(in_dim // 2, out_dim, bias=False)

        self.res_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, txt, img):
        x = torch.cat([txt, img], dim=-1)    # (B, 2D)
        x_n = self.ln(x)
        h = self.fc2(F.gelu(self.fc1(x_n)))
        h = self.dropout(h)

        g = torch.sigmoid(self.gate(x_n))    # (B, 2)
        t_part, i_part = torch.split(x_n, x_n.size(-1)//2, dim=-1)
        t_proj = self.proj_text(t_part)
        i_proj = self.proj_image(i_part)
        gated = g[:, :1]*t_proj + g[:, 1:2]*i_proj

        out = gated + self.res_scale * h
        out = F.normalize(out, dim=-1)
        return out


# ----------------- encode & losses -----------------
def clip_encode_text_batch(processor, model, texts, device, max_len):
    inputs = processor(text=texts, return_tensors="pt", padding=True,
                       truncation=True, max_length=max_len).to(device)
    with torch.no_grad():
        f = model.get_text_features(**inputs)
    return F.normalize(f, dim=-1)

def clip_encode_image_batch(processor, model, img_paths, device):
    images = []
    for p in img_paths:
        if p is None:
            images.append(Image.new("RGB", (224, 224), color=(0, 0, 0)))
        else:
            try:
                images.append(Image.open(p).convert("RGB"))
            except (UnidentifiedImageError, FileNotFoundError):
                images.append(Image.new("RGB", (224, 224), color=(0, 0, 0)))
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        f = model.get_image_features(**inputs)
    return F.normalize(f, dim=-1)

def info_nce_from_pairs(anchor, positive, temperature=0.07):
    """
    in-batch negatives：anchor (B,d), positive (B,d)
    logits: diag 为正，其余为负
    """
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    logits = anchor @ positive.t() / temperature
    labels = torch.arange(anchor.size(0), device=anchor.device)
    return F.cross_entropy(logits, labels)


# ----------------- builders -----------------
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


# ----------------- train & export -----------------
def train_pure_ti(args):
    # 路径
    processed_data_path = os.path.join(args.save_root, args.dataset)
    item2id_file = os.path.join(processed_data_path, f"{args.dataset}.item2id")
    id2item = get_id2item_dict(item2id_file)

    image_base_path = os.path.join(args.image_root, f"amazon{args.data_version}", "Images")
    image_dir = os.path.join(image_base_path, args.dataset)
    images_info_path = os.path.join(image_base_path, f"{args.dataset}_images_info.json")
    images_info = load_json(images_info_path) or {}

    # 文本
    text_map = build_text_map(args, id2item)

    # 模型
    device = torch.device(args.device)
    processor = CLIPProcessor.from_pretrained(args.model_name_or_path, cache_dir=args.model_cache_dir, use_fast=True)
    clip = CLIPModel.from_pretrained(args.model_name_or_path, cache_dir=args.model_cache_dir).to(device)
    clip.eval()
    if args.freeze_clip:
        for p in clip.parameters():
            p.requires_grad = False

    proj_dim = clip.config.projection_dim
    fusion = GatedFusion(in_dim=proj_dim*2, mid_dim=max(512, proj_dim),
                         out_dim=512, dropout=args.dropout).to(device)

    # 优化器
    params = list(fusion.parameters())
    if not args.freeze_clip and args.tune_clip_last:
        for n, p in clip.named_parameters():
            if any(k in n for k in ["text_projection", "visual_projection"]):
                p.requires_grad = True
                params.append(p)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type=="cuda" and args.amp))

    # 数据
    ds = ItemDataset(id2item, text_map, image_dir, images_info)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # 训练（仅模态损失）
    best_loss = float("inf")
    fusion.train()
    for epoch in range(args.epochs):
        pbar = tqdm(dl, desc=f"Train epoch {epoch+1}/{args.epochs}")
        epoch_loss = 0.0
        for batch in pbar:
            texts = batch["text"]
            img_paths = batch["img_path"]

            # 编码
            with torch.no_grad():
                T = clip_encode_text_batch(processor, clip, texts, device, args.max_sent_len)  # (B,D)
                I = clip_encode_image_batch(processor, clip, img_paths, device)                # (B,D)

            # 融合
            with torch.amp.autocast('cuda', enabled=(device.type=="cuda" and args.amp)):
                H = fusion(T, I)  # (B,512)

                # 纯模态对齐：text↔image（双向） + fusion↔text + fusion↔image
                L_ti = info_nce_from_pairs(T, I, temperature=args.temperature) \
                       + info_nce_from_pairs(I, T, temperature=args.temperature)
                L_ht = info_nce_from_pairs(H, T, temperature=args.temperature_fusion)
                L_hi = info_nce_from_pairs(H, I, temperature=args.temperature_fusion)

                loss = args.w_ti * L_ti + args.w_ht * L_ht + args.w_hi * L_hi

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg = epoch_loss / len(dl)
        print(f"[E{epoch+1}] loss={avg:.4f}")
        if avg < best_loss:
            best_loss = avg
            ckpt_dir = os.path.join(args.save_root, args.dataset, "embeddings")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({"fusion": fusion.state_dict()},
                       os.path.join(ckpt_dir, "fusion_pure_ti_best.pt"))

    # -------- 导出最终 512 维融合向量 --------
    fusion.eval()
    sorted_ids = sorted(id2item.keys(), key=int)

    # 准备文本与首图
    texts_all = []
    imgs_all = []
    for mapped_id in sorted_ids:
        orig = id2item[mapped_id]
        texts_all.append(text_map.get(orig, "N/A"))
        # 选第一张可用图
        img = None
        names = images_info.get(orig, [])
        if isinstance(names, list):
            for name in names:
                fp = os.path.join(image_dir, name)
                if os.path.exists(fp):
                    img = fp
                    break
        imgs_all.append(img)

    all_feats = []
    with torch.no_grad():
        for i in tqdm(range(0, len(sorted_ids), args.export_bs), desc="Export fused 512d"):
            t_batch = texts_all[i:i+args.export_bs]
            p_batch = imgs_all[i:i+args.export_bs]
            T = clip_encode_text_batch(processor, clip, t_batch, device, args.max_sent_len)
            I = clip_encode_image_batch(processor, clip, p_batch, device)
            H = fusion(T, I)   # (b,512)
            all_feats.append(H.cpu())
    Z = torch.cat(all_feats, dim=0).numpy().astype(np.float32)

    # 可选 PCA/whiten
    out_dir = os.path.join(args.save_root, args.dataset, "embeddings")
    os.makedirs(out_dir, exist_ok=True)
    hf_tag = args.model_name_or_path.split("/")[-1].replace("/", "-")
    out_base = os.path.join(out_dir, f"{args.dataset}.emb-fused-{hf_tag}-forge512.npy")
    np.save(out_base, Z)
    print(f"[OK] saved fused 512d: {out_base}  shape={Z.shape}")

    if args.pca_dim > 0 and args.pca_dim < Z.shape[1]:
        print(f"[INFO] PCA -> {args.pca_dim} (whiten={args.whiten})")
        pca = PCA(n_components=args.pca_dim, whiten=args.whiten, svd_solver="auto", random_state=42)
        Zp = pca.fit_transform(Z).astype(np.float32)
        out_pca = os.path.join(out_dir, f"{args.dataset}.emb-fused-{hf_tag}-forge512-pca{args.pca_dim}.npy")
        np.save(out_pca, Zp)
        print(f"[OK] saved PCA: {out_pca}  shape={Zp.shape}  var={pca.explained_variance_ratio_.sum():.4f}")

    return out_base


# ----------------- argparser -----------------
def build_parser():
    ap = argparse.ArgumentParser("Pure text-image contrastive fusion (no collaborative positives)")
    # data
    ap.add_argument("--data_version", type=str, default="14", choices=["14","18"])
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--dataset_type", type=str, default="amazon", choices=["amazon","movielens"])
    ap.add_argument("--image_root", type=str, default="../datasets")
    ap.add_argument("--save_root", type=str, default="../datasets")

    # model
    ap.add_argument("--model_name_or_path", type=str, default="openai/clip-vit-base-patch32")
    ap.add_argument("--model_cache_dir", type=str, default=None)
    ap.add_argument("--freeze_clip", action="store_true", help="默认冻结 CLIP，稳定省算力")
    ap.add_argument("--tune_clip_last", action="store_true", help="不冻结时：仅解冻投影头等末端层")
    ap.add_argument("--dropout", type=float, default=0.1)

    # train
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--max_sent_len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.07, help="text-image 温度")
    ap.add_argument("--temperature_fusion", type=float, default=0.07, help="fusion 对齐温度")

    # loss weights
    ap.add_argument("--w_ti", type=float, default=1.0, help="text<->image (双向)")
    ap.add_argument("--w_ht", type=float, default=0.5, help="fusion->text")
    ap.add_argument("--w_hi", type=float, default=0.5, help="fusion->image")

    # export
    ap.add_argument("--export_bs", type=int, default=1024)
    ap.add_argument("--pca_dim", type=int, default=0)
    ap.add_argument("--whiten", action="store_true")

    # device
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return ap


def main():
    args = build_parser().parse_args()
    print(f"[CFG] device={args.device}  model={args.model_name_or_path}")
    train_pure_ti(args)


if __name__ == "__main__":
    """
    例子：
    python fuse_contrastive_pure_ti.py \
      --dataset Baby --dataset_type amazon \
      --image_root ../datasets --save_root ../datasets \
      --model_name_or_path /home/wj/peiyu/LLM_Models/openai-mirror/clip-vit-base-patch32 \
      --freeze_clip --epochs 4 --batch_size 512 --amp \
      --w_ti 1.0 --w_ht 0.5 --w_hi 0.5 \
      --temperature 0.07 --temperature_fusion 0.07 \
      --export_bs 1024 --pca_dim 0
    """
    main()
