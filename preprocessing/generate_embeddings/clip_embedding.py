#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FORGE-style multimodal fusion with contrastive learning (text/image/fusion) + i2i positives.

相对你的原版增强点：
1) 仍用同一 CLIP 模型抽取文本/图像嵌入（同空间）；
2) 新增三路 InfoNCE（text、image、fusion），正样本含：
   - 同一 item 的 (text, image) 跨模态对齐（CLIP 风格）
   - i2i 最强共现邻居 (text-text, image-image, fusion-fusion)
3) 融合：norm 后 concat -> 小型 MLP 投影到 512 维（可学习门控+残差）
4) 训练完成后，导出融合后的 512 维 emb 到 .npy（可选再做 PCA/whiten）
5) 仍兼容你原始目录结构；新增可选 i2i 文件支持（若 item.json 里含 related_item 也可自动读）

目录假设（兼容你原版）：
- item2id:        <save_root>/<dataset>/<dataset>.item2id
- item.json:      <save_root>/<dataset>/<dataset>.item.json   # 文本/sideinfo
- images_info:    <image_root>/amazon<data_version>/Images/<dataset>_images_info.json
- image folder:   <image_root>/amazon<data_version>/Images/<dataset>/
- 可选 i2i:       <save_root>/<dataset>/<dataset>.i2i.json    # { item_id: related_item_id }
输出：
- 训练日志与最终 emb: <save_root>/<dataset>/embeddings/<dataset>.emb-fused-<model>-forge512.npy
- 可选 PCA:       <save_root>/<dataset>/embeddings/<dataset>.emb-fused-<model>-forge512-pca<n>.npy
"""

import os
import json
import math
import argparse
import random
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

def l2_normalize_np(x: np.ndarray, eps=1e-12):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)

def l2_normalize_t(x: torch.Tensor, eps=1e-12):
    return x / (x.norm(dim=-1, keepdim=True) + eps)


# ----------------- dataset -----------------
class ItemDataset(Dataset):
    """
    提供按 id 顺序访问的 item 列表，并可取到：
    - text 字符串
    - 一个可用图片文件路径（如果没有，返回 None）
    - i2i 正样本的 index（没有则回退为自身）
    """
    def __init__(self, args, id2item, text_map, image_dir, images_info, i2i_map):
        self.args = args
        self.idx_list = sorted(id2item.keys(), key=int)  # mapped ids (string)
        self.id2item = id2item
        self.text_map = text_map
        self.image_dir = image_dir
        self.images_info = images_info or {}
        self.i2i_map = i2i_map or {}
        # 反向索引：original_item_id -> mapped_id(str)
        self.item2mapped = {v: k for k, v in id2item.items()}

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

    def _text_of(self, original_item_id):
        return self.text_map.get(original_item_id, "N/A")

    def _get_pos_index(self, original_item_id):
        pos_item = self.i2i_map.get(original_item_id, None)
        if pos_item and pos_item in self.item2mapped:
            return self.item2mapped[pos_item]   # mapped id(str)
        # fallback: self
        return self.item2mapped.get(original_item_id)

    def __getitem__(self, idx):
        mapped_id_str = self.idx_list[idx]
        original_item_id = self.id2item[mapped_id_str]

        # text
        text = self._text_of(original_item_id)

        # image
        img_path = self._pick_one_image(original_item_id)

        # i2i positive index (mapped string id)
        pos_mapped_id_str = self._get_pos_index(original_item_id)
        return {
            "mapped_id": mapped_id_str,
            "orig_id": original_item_id,
            "text": text,
            "img_path": img_path,
            "pos_mapped_id": pos_mapped_id_str,
        }


# ----------------- fusion head -----------------
class GatedFusion(nn.Module):
    """
    简洁稳定的融合头：
    [norm(t); norm(i)] -> LN -> Linear(2D->D) -> GELU -> Linear(D->D)  (residual)
    + 门控（sigmoid）在 t/i 间做可学习加权
    最终投影到 out_dim=512
    """
    def __init__(self, in_dim, mid_dim, out_dim=512, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

        # 门控：从 concat 输入预测两个权重
        self.gate = nn.Linear(in_dim, 2)
        self.proj_text = nn.Linear(in_dim // 2, out_dim, bias=False)
        self.proj_image = nn.Linear(in_dim // 2, out_dim, bias=False)

        self.res_scale = nn.Parameter(torch.tensor(0.5))  # 残差缩放，稳住训练

    def forward(self, txt, img):
        # txt/img 已经是 L2 norm 的同维向量 (B, D)
        x = torch.cat([txt, img], dim=-1)    # (B, 2D)
        x_n = self.ln(x)
        h = self.fc2(F.gelu(self.fc1(x_n)))
        h = self.dropout(h)

        # 门控加权（学习每模态重要性）
        g = torch.sigmoid(self.gate(x_n))    # (B,2)
        t_part, i_part = torch.split(x_n, x_n.size(-1)//2, dim=-1)
        t_proj = self.proj_text(t_part)
        i_proj = self.proj_image(i_part)
        gated = g[:, :1]*t_proj + g[:, 1:2]*i_proj

        # 残差融合
        out = gated + self.res_scale * h
        out = F.normalize(out, dim=-1)       # 最终再 L2 归一化
        return out


# ----------------- training utils -----------------
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

def batch_index_of(mapped_ids, id2row):
    # 把 batch 内的 pos_mapped_id（字符串）映射成本批行号，映射不到的回退自身
    idxs = []
    B = len(mapped_ids)
    for i, mid in enumerate(mapped_ids):
        j = id2row.get(mid, i)
        idxs.append(j)
    return torch.tensor(idxs, dtype=torch.long)

def gather_by_index(x, idx):
    # x: (B,d), idx: (B,) -> (B,d)
    return x[idx]


# ----------------- main training & export -----------------
def build_text_map(args, id2item):
    # 读 item.json -> 拼 sideinfo 文本
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

    # 允许把 NER/额外 sideinfo 拼接到文本里
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

def build_i2i_map(args, id2item):
    """
    i2i 优先级：
    1) <save_root>/<dataset>/<dataset>.i2i.json  若存在：{ item_id: related_item_id }
    2) 若 item.json 的条目里有 'related_item' 字段，也会读取
    3) 不存在则返回空（训练时会回退到自配对）
    """
    i2i_path = os.path.join(args.save_root, args.dataset, f"{args.dataset}.i2i.json")
    i2i = {}
    if os.path.exists(i2i_path):
        data = load_json(i2i_path) or {}
        if isinstance(data, dict):
            i2i.update({str(k): str(v) for k, v in data.items()})

    # try from item.json if provided
    item_json = os.path.join(args.save_root, args.dataset, f"{args.dataset}.item.json")
    meta = load_json(item_json) or {}
    if isinstance(meta, dict):
        for orig_id in meta.keys():
            v = meta.get(orig_id, {})
            rid = v.get("related_item", None)
            if isinstance(rid, (str, int)):
                i2i[str(orig_id)] = str(rid)
    print(f"[I2I] loaded {len(i2i)} relations.")
    return i2i

def train_forge_contrastive(args):
    # 基础路径
    processed_data_path = os.path.join(args.save_root, args.dataset)
    item2id_file = os.path.join(processed_data_path, f"{args.dataset}.item2id")
    id2item = get_id2item_dict(item2id_file)

    image_base_path = os.path.join(args.image_root, f"amazon{args.data_version}", "Images")
    image_dir = os.path.join(image_base_path, args.dataset)
    images_info_path = os.path.join(image_base_path, f"{args.dataset}_images_info.json")
    images_info = load_json(images_info_path) or {}

    # 文本与 i2i
    text_map = build_text_map(args, id2item)
    i2i_map = build_i2i_map(args, id2item)

    # 模型
    device = torch.device(args.device)
    processor = CLIPProcessor.from_pretrained(args.model_name_or_path, cache_dir=args.model_cache_dir)
    clip = CLIPModel.from_pretrained(args.model_name_or_path, cache_dir=args.model_cache_dir).to(device)
    clip.eval()
    if args.freeze_clip:
        for p in clip.parameters():
            p.requires_grad = False

    proj_dim = clip.config.projection_dim
    fusion = GatedFusion(in_dim=proj_dim*2, mid_dim=max(512, proj_dim), out_dim=512, dropout=args.dropout).to(device)

    # 优化器
    params = list(fusion.parameters())
    if not args.freeze_clip and args.tune_clip_last:
        # 仅解冻投影头/最后层，谨慎：不同 CLIP 实现层名不同，这里给出范例
        for n, p in clip.named_parameters():
            if any(k in n for k in ["text_projection", "visual_projection"]):
                p.requires_grad = True
                params.append(p)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda" and args.amp))

    # 数据
    ds = ItemDataset(args, id2item, text_map, image_dir, images_info, i2i_map)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # 训练
    clip_temperature = args.temperature
    best_loss = float("inf")
    fusion.train()
    for epoch in range(args.epochs):
        pbar = tqdm(dl, desc=f"Train epoch {epoch+1}/{args.epochs}")
        epoch_loss = 0.0
        for batch in pbar:
            texts = batch["text"]
            img_paths = batch["img_path"]
            mapped_ids = batch["mapped_id"]  # list of str
            pos_mapped_ids = batch["pos_mapped_id"]

            # 编码
            with torch.no_grad():
                T = clip_encode_text_batch(processor, clip, texts, device, args.max_sent_len)     # (B, D)
                I = clip_encode_image_batch(processor, clip, img_paths, device)                   # (B, D)

            # 融合到 512
            with torch.cuda.amp.autocast(enabled=(device.type=="cuda" and args.amp)):
                H = fusion(T, I)   # (B, 512)

                # 1) 跨模态对齐（同一 item 的 text<->image）
                L_ti = info_nce_from_pairs(T, I, temperature=clip_temperature) \
                       + info_nce_from_pairs(I, T, temperature=clip_temperature)

                # 2) i2i 正样本（text-text, image-image, fusion-fusion）
                #    构造 batch 内索引映射
                id2row = {m: i for i, m in enumerate(mapped_ids)}
                pos_idx = batch_index_of(pos_mapped_ids, id2row).to(device)
                T_pos = gather_by_index(T, pos_idx)
                I_pos = gather_by_index(I, pos_idx)
                H_pos = gather_by_index(H, pos_idx)

                L_tt = info_nce_from_pairs(T, T_pos, temperature=args.temperature_i2i)
                L_ii = info_nce_from_pairs(I, I_pos, temperature=args.temperature_i2i)
                L_hh = info_nce_from_pairs(H, H_pos, temperature=args.temperature_i2i)

                loss = args.w_ti * L_ti + args.w_tt * L_tt + args.w_ii * L_ii + args.w_hh * L_hh

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
            # 只存融合头（足够用）
            ckpt_dir = os.path.join(args.save_root, args.dataset, "embeddings")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({"fusion": fusion.state_dict()}, os.path.join(ckpt_dir, "fusion_best.pt"))

    # -------- 导出最终 512 维融合向量 --------
    fusion.eval()
    # 重新顺序遍历，确保导出顺序与 item2id 一致
    sorted_ids = sorted(id2item.keys(), key=int)
    texts_all = []
    imgs_all = []
    for mapped_id in sorted_ids:
        orig = id2item[mapped_id]
        texts_all.append(text_map.get(orig, "N/A"))

    # 聚合图像路径
    image_base_path = os.path.join(args.image_root, f"amazon{args.data_version}", "Images")
    image_dir = os.path.join(image_base_path, args.dataset)
    images_info_path = os.path.join(image_base_path, f"{args.dataset}_images_info.json")
    images_info = load_json(images_info_path) or {}
    for mapped_id in sorted_ids:
        orig = id2item[mapped_id]
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
    ap = argparse.ArgumentParser("FORGE-style multimodal fusion with contrastive learning")
    # data
    ap.add_argument("--data_version", type=str, default="14", choices=["14","18"])
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--dataset_type", type=str, default="amazon", choices=["amazon","movielens"])
    ap.add_argument("--image_root", type=str, default="../datasets")
    ap.add_argument("--save_root", type=str, default="../datasets")

    # model
    ap.add_argument("--model_name_or_path", type=str, default="openai/clip-vit-base-patch32")
    ap.add_argument("--model_cache_dir", type=str, default=None)
    ap.add_argument("--freeze_clip", action="store_true", help="默认冻结 CLIP，稳定又省算力")
    ap.add_argument("--tune_clip_last", action="store_true", help="在不冻结时：仅解冻 text/visual_projection 等末端层")
    ap.add_argument("--dropout", type=float, default=0.1)

    # train
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_sent_len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.07, help="跨模态 text-image 温度")
    ap.add_argument("--temperature_i2i", type=float, default=0.07, help="i2i 对比温度")

    # loss weights
    ap.add_argument("--w_ti", type=float, default=1.0, help="跨模态 text<->image")
    ap.add_argument("--w_tt", type=float, default=0.5, help="i2i text-text")
    ap.add_argument("--w_ii", type=float, default=0.5, help="i2i image-image")
    ap.add_argument("--w_hh", type=float, default=1.0, help="i2i fusion-fusion")

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
    train_forge_contrastive(args)


if __name__ == "__main__":
    """
    例子：
    python fuse_contrastive_forge.py \
      --dataset Beauty --dataset_type amazon \
      --image_root ../datasets --save_root ../datasets \
      --model_name_or_path openai/clip-vit-base-patch32 \
      --freeze_clip --epochs 2 --batch_size 512 --amp \
      --w_ti 1.0 --w_tt 0.5 --w_ii 0.5 --w_hh 1.0 \
      --temperature 0.07 --temperature_i2i 0.07 \
      --export_bs 1024 --pca_dim 0

    可选：提供 i2i 文件 ../datasets/Beauty/Beauty.i2i.json （或在 item.json 各条目里给 related_item 字段）
    """
    main()
