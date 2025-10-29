# preprocessing/generate_embeddings/clip_embedding.py
"""
多模态融合 V2 - “黄金标准”对齐版

核心理念：
1. 假设文本 (T) 是下游任务的“黄金标准”，图像 (I) 是辅助信息。
2. 融合头 H = Fusion(T, I) 的目标是生成一个“增强版”的 T。
3. 训练损失（核心修正）：
   - L(H → T) + L(T → H) (双向 InfoNCE)
   - 迫使 H 与 T 在同一语义空间对齐。
   - 融合头被训练去“利用”I 来增强 T，或“忽略”I 来保护 T。
4. 默认冻结 CLIP，只训练融合头。
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


# ----------------- utils (保持不变) -----------------
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


# ----------------- dataset (保持不变) -----------------
class ItemDataset(Dataset):
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


# ----------------- fusion head (保持不变) -----------------
class GatedFusion(nn.Module):
    """
    您原来的 GatedFusion 融合头 (保持不变)。
    这个门控结构非常适合我们的新目标，因为它能学会给 T 和 I 动态分配权重。
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


# ----------------- encode & losses (保持不变) -----------------
def clip_encode_text_batch(processor, model, texts, device, max_len):
    """
    (保持不变) 
    我们默认在 no_grad 下运行 (因为 CLIP 处于 eval() 模式)
    """
    inputs = processor(text=texts, return_tensors="pt", padding=True,
                       truncation=True, max_length=max_len).to(device)
    # 保持 no_grad，因为我们只训练融合头
    with torch.no_grad():
        f = model.get_text_features(**inputs)
    return F.normalize(f, dim=-1)

def clip_encode_image_batch(processor, model, img_paths, device):
    """(保持不变)"""
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
    # 保持 no_grad，因为我们只训练融合头
    with torch.no_grad():
        f = model.get_image_features(**inputs)
    return F.normalize(f, dim=-1)

def info_nce_from_pairs(anchor, positive, temperature=0.07):
    """(保持不变)"""
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    logits = anchor @ positive.t() / temperature
    labels = torch.arange(anchor.size(0), device=anchor.device)
    return F.cross_entropy(logits, labels)


# ----------------- builders (保持不变) -----------------
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


# ----------------- train & export (核心修改) -----------------
def train_fusion(args):
    # 路径 (不变)
    processed_data_path = os.path.join(args.save_root, args.dataset)
    item2id_file = os.path.join(processed_data_path, f"{args.dataset}.item2id")
    id2item = get_id2item_dict(item2id_file)

    image_base_path = os.path.join(args.image_root, f"amazon{args.data_version}", "Images")
    image_dir = os.path.join(image_base_path, args.dataset)
    images_info_path = os.path.join(image_base_path, f"{args.dataset}_images_info.json")
    images_info = load_json(images_info_path) or {}

    # 文本 (不变)
    text_map = build_text_map(args, id2item)

    # 模型 (不变, 默认冻结)
    device = torch.device(args.device)
    processor = CLIPProcessor.from_pretrained(args.model_name_or_path, cache_dir=args.model_cache_dir, use_fast=True)
    clip = CLIPModel.from_pretrained(args.model_name_or_path, cache_dir=args.model_cache_dir).to(device)
    
    # ✅ 关键：始终冻结 CLIP，我们只训练融合头
    clip.eval()
    for p in clip.parameters():
        p.requires_grad = False
    print("[INFO] CLIP model is frozen. Only the GatedFusion head will be trained.")

    proj_dim = clip.config.projection_dim
    fusion = GatedFusion(in_dim=proj_dim*2, mid_dim=max(512, proj_dim),
                         out_dim=512, dropout=args.dropout).to(device)

    # 优化器 (✅ 修正：只优化 fusion 的参数)
    params = list(fusion.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type=="cuda" and args.amp))

    # 数据 (不变)
    ds = ItemDataset(id2item, text_map, image_dir, images_info)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # 训练（✅ 核心修改：修正损失函数）
    best_loss = float("inf")
    fusion.train()
    for epoch in range(args.epochs):
        pbar = tqdm(dl, desc=f"Train epoch {epoch+1}/{args.epochs}")
        epoch_loss = 0.0
        for batch in pbar:
            texts = batch["text"]
            img_paths = batch["img_path"]

            # 编码 (不变, 在 no_grad 下执行)
            T = clip_encode_text_batch(processor, clip, texts, device, args.max_sent_len)  # (B,D)
            I = clip_encode_image_batch(processor, clip, img_paths, device)                # (B,D)

            # 融合 (需要梯度)
            with torch.amp.autocast('cuda', enabled=(device.type=="cuda" and args.amp)):
                H = fusion(T, I)  # (B,512)

                # ✅ 核心修正：
                # 训练 H 去对齐 “黄金标准” T。
                # H 必须学会如何利用 I 来更好地模仿 T。
                # T.detach() 确保梯度只从 H 流向 Fusion 模块，而不是流向 T。
                L_ht = info_nce_from_pairs(H, T.detach(), temperature=args.temperature)
                L_th = info_nce_from_pairs(T.detach(), H, temperature=args.temperature)

                loss = L_ht + L_th # (权重默认为 1.0 + 1.0)
                
                # ❌ 移除旧的、有问题的损失
                # L_ti = ...
                # L_ht = ...
                # L_hi = ... (这是导致“污染”的项)
                # loss = ...

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
            # 保存最佳的融合头
            torch.save({"fusion": fusion.state_dict()},
                       os.path.join(ckpt_dir, "fusion_gold_standard_best.pt"))
            print(f"Saved best fusion head to {ckpt_dir}")

    # -------- 导出最终 512 维融合向量 (逻辑不变) --------
    
    # ✅ 加载我们训练好的最佳融合头
    best_ckpt_path = os.path.join(args.save_root, args.dataset, "embeddings", "fusion_gold_standard_best.pt")
    if os.path.exists(best_ckpt_path):
        print(f"Loading best fusion head from {best_ckpt_path}")
        fusion.load_state_dict(torch.load(best_ckpt_path)["fusion"])
    else:
        print("[WARN] No best fusion head found. Exporting with the last epoch's weights.")

    fusion.eval()
    sorted_ids = sorted(id2item.keys(), key=int)

    # 准备文本与首图 (不变)
    texts_all = []
    imgs_all = []
    for mapped_id in sorted_ids:
        orig = id2item[mapped_id]
        texts_all.append(text_map.get(orig, "N/A"))
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

    # 导出 (不变)
    out_dir = os.path.join(args.save_root, args.dataset, "embeddings")
    os.makedirs(out_dir, exist_ok=True)
    hf_tag = args.model_name_or_path.split("/")[-1].replace("/", "-")
    out_base = os.path.join(out_dir, f"{args.dataset}.emb-fused-{hf_tag}-gold512.npy")
    np.save(out_base, Z)
    print(f"[OK] saved fused 512d: {out_base}  shape={Z.shape}")

    if args.pca_dim > 0 and args.pca_dim < Z.shape[1]:
        print(f"[INFO] PCA -> {args.pca_dim} (whiten={args.whiten})")
        pca = PCA(n_components=args.pca_dim, whiten=args.whiten, svd_solver="auto", random_state=42)
        Zp = pca.fit_transform(Z).astype(np.float32)
        out_pca = os.path.join(out_dir, f"{args.dataset}.emb-fused-{hf_tag}-gold512-pca{args.pca_dim}.npy")
        np.save(out_pca, Zp)
        print(f"[OK] saved PCA: {out_pca}  shape={Zp.shape}  var={pca.explained_variance_ratio_.sum():.4f}")

    return out_base


# ----------------- argparser (✅ 修正) -----------------
def build_parser():
    ap = argparse.ArgumentParser("V2: Gold-Standard Text-Image Fusion")
    # data
    ap.add_argument("--data_version", type=str, default="14", choices=["14","18"])
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--dataset_type", type=str, default="amazon", choices=["amazon","movielens"])
    ap.add_argument("--image_root", type=str, default="../datasets")
    ap.add_argument("--save_root", type=str, default="../datasets")

    # model
    ap.add_argument("--model_name_or_path", type=str, default="openai/clip-vit-base-patch32")
    ap.add_argument("--model_cache_dir", type=str, default=None)
    ap.add_argument("--dropout", type=float, default=0.1)
    # ✅ 移除了 freeze_clip 和 tune_clip_last，默认冻结

    # train
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--max_sent_len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4, help="融合头的学习率")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.07, help="H<->T 对齐温度")

    # ✅ 移除了 w_ti, w_ht, w_hi

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
    print(f"[INFO] Training Philosophy: Gold Standard Alignment (H <-> T)")
    train_fusion(args)


if __name__ == "__main__":
    """
    例子 (命令不变，但意义已变)：
    python fuse_gold_standard.py \
      --dataset Baby --dataset_type amazon \
      --image_root ../datasets --save_root ../datasets \
      --model_name_or_path /home/wj/peiyu/LLM_Models/openai-mirror/clip-vit-base-patch32 \
      --epochs 4 --batch_size 512 --amp \
      --temperature 0.07 \
      --export_bs 1024 --pca_dim 0
    """
    main()