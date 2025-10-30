#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态融合 V3 - 黄金标准对齐 (独立训练脚本)
(V3.1 - 自动路径构建)

职责：
1. 根据模型标签自动构建输入 .npy 文件路径。
2. 训练 GatedFusion 头 (L(H<->T) 损失)。
3. 导出融合后的 H 向量。
"""

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.decomposition import PCA
import joblib 

# (核心) 从父目录导入 utils.py
try:
    # 假设此脚本位于 preprocessing/ 目录下
    sys.path.append(os.path.dirname(os.path.abspath(__file__))) 
    from utils import set_device, apply_pca_and_save, check_path
    print("[INFO] 成功从 utils.py 导入共享函数。")
except ImportError as e:
    # 如果 utils.py 在上一级 (例如 preprocessing/utils.py)
    try:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils import set_device, apply_pca_and_save, check_path
        print("[INFO] 成功从父目录 utils.py 导入共享函数。")
    except ImportError:
        print(f"导入错误: {e}")
        print("错误: 无法从 (preprocessing/) 目录导入 utils.py。")
        sys.exit(1)

# =================================================================
# ================== 1. 融合头模型 (GatedFusion) ==================
# =================================================================
class GatedFusion(nn.Module):
    """
    门控融合头 (修改版 - 支持不同输入维度)。
    """
    # ✅ (修改) __init__ 签名
    def __init__(self, in_dim_text: int, in_dim_image: int, mid_dim: int, out_dim: int, dropout=0.1):
        super().__init__()
        self.in_dim_text = in_dim_text
        self.in_dim_image = in_dim_image
        self.in_dim_concat = in_dim_text + in_dim_image # 拼接后的维度
        self.out_dim = out_dim
        
        # ✅ (修改) LayerNorm 作用于拼接后的维度
        self.ln = nn.LayerNorm(self.in_dim_concat) 
        
        # ✅ (修改) MLP 路径处理拼接后的维度
        self.fc1 = nn.Linear(self.in_dim_concat, mid_dim)
        self.fc2 = nn.Linear(mid_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

        # ✅ (修改) Gate 输入也是拼接后的维度
        self.gate = nn.Linear(self.in_dim_concat, 2) 
        
        # ✅ (修改) 独立投影层，输入维度不同
        self.proj_text = nn.Linear(self.in_dim_text, out_dim, bias=False) 
        self.proj_image = nn.Linear(self.in_dim_image, out_dim, bias=False) 

        self.res_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, txt: torch.Tensor, img: torch.Tensor):
        # txt: (B, D_text), img: (B, D_image)
        
        # 拼接 (现在维度是 D_T + D_I)
        x = torch.cat([txt, img], dim=-1)    # (B, D_T + D_I)
        x_n = self.ln(x)                     # (B, D_T + D_I)
        
        # 1. 门控路径
        g = torch.sigmoid(self.gate(x_n))    # (B, 2)
        # (独立投影)
        t_proj = self.proj_text(txt)         # (B, D_out)
        i_proj = self.proj_image(img)        # (B, D_out)
        gated = g[:, :1] * t_proj + g[:, 1:] * i_proj # (B, D_out)

        # 2. MLP 残差路径 (输入是 x_n)
        h = self.fc2(F.gelu(self.fc1(x_n)))
        h = self.dropout(h)                  # (B, D_out)

        out = gated + self.res_scale * h
        out = F.normalize(out, dim=-1)
        return out

# =================================================================
# ================== 2. 损失函数 ==================
# =================================================================
def info_nce_from_pairs(anchor, positive, temperature):
    # ... (info_nce_from_pairs 函数的代码保持不变) ...
    anchor = F.normalize(anchor, dim=-1); positive = F.normalize(positive, dim=-1)
    logits = torch.matmul(anchor, positive.t()) / temperature
    labels = torch.arange(anchor.shape[0], device=anchor.device)
    return F.cross_entropy(logits, labels)

# =================================================================
# ================== 3. 主训练与提取函数 ==================
# =================================================================
def train_fusion_head(args, T_emb: np.ndarray, I_emb: np.ndarray):
    """
    训练 GatedFusion 模型头 (修改版 - 处理不同维度输入)。
    """
    device = args.device
    
    # --- 1. 验证输入 ---
    if T_emb.shape[0] != I_emb.shape[0]:
        raise ValueError(f"文本 ({T_emb.shape[0]}) 和图像 ({I_emb.shape[0]}) 物品数量不匹配！")
    # 🚨 (移除) 不再需要检查维度是否相等
    # if T_emb.shape[1] != I_emb.shape[1]: ... 
    
    # ✅ (修改) 获取两个输入的维度
    in_dim_text = T_emb.shape[1]
    in_dim_image = I_emb.shape[1]
    
    # 确定输出维度 (如果未指定或为 0，则默认为文本维度)
    out_dim = args.fusion_out_dim if args.fusion_out_dim > 0 else in_dim_text 
    # 确定中间层维度 (可以基于输入或输出维度)
    mid_dim = max(out_dim, (in_dim_text + in_dim_image) // 2) # 例如取拼接维度的一半
    
    print(f"[INFO] 训练 GatedFusion：Text Dim={in_dim_text}, Image Dim={in_dim_image}, Mid Dim={mid_dim}, Output Dim={out_dim}")

    # --- 2. ✅ (修改) 初始化模型 (传入两个 in_dim) ---
    fusion_head = GatedFusion(
        in_dim_text=in_dim_text,     # <--- 传入文本维度
        in_dim_image=in_dim_image,   # <--- 传入图像维度
        mid_dim=mid_dim,
        out_dim=out_dim,
        dropout=args.fusion_dropout
    ).to(device)

    # --- 3. 准备训练 (不变) ---
    optimizer = torch.optim.AdamW(fusion_head.parameters(), lr=args.fusion_lr, weight_decay=args.fusion_weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda' and args.amp))
    print("创建 DataLoader...")
    dataset = TensorDataset(torch.from_numpy(T_emb).float(), torch.from_numpy(I_emb).float())
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))

    # --- 4. 训练融合头 (不变) ---
    print(f"开始训练 GatedFusion 头... (Epochs: {args.fusion_epochs}, LR: {args.fusion_lr})")
    best_loss = float("inf")
    ckpt_dir = os.path.join(args.save_root, args.dataset, "embeddings", "checkpoints")
    check_path(ckpt_dir)
    best_ckpt_path = os.path.join(ckpt_dir, f"fusion_head_{args.output_tag}.pt")

    for epoch in range(args.fusion_epochs):
        # ... (训练循环内部逻辑完全不变，因为 L(H<->T) 仍然适用) ...
        fusion_head.train()
        pbar = tqdm(dataloader, desc=f"Train Fusion Epoch {epoch+1}/{args.fusion_epochs}")
        epoch_loss = 0.0
        for batch_T, batch_I in pbar:
            T, I = batch_T.to(device), batch_I.to(device)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda' and args.amp)):
                H = fusion_head(T, I)  # (B, out_dim)

                # 文本/视觉各自投影到同一 out_dim（如 512）
                T_proj = F.normalize(fusion_head.proj_text(T), dim=-1)
                I_proj = F.normalize(fusion_head.proj_image(I), dim=-1)

                # 文本指导融合（已有思想）
                L_ht = info_nce_from_pairs(H, T_proj.detach(), temperature=args.fusion_temperature)

                # 🔥 文本指导视觉（核心）
                L_it = info_nce_from_pairs(I_proj, T_proj.detach(), temperature=args.fusion_temperature)
                L_kd = 1 - (I_proj * T_proj.detach()).sum(dim=-1).mean()  # 也可换成 F.mse_loss

                # 融合吸收视觉
                L_hi = info_nce_from_pairs(H, I_proj.detach(), temperature=args.fusion_temperature)

                loss = 1.0 * L_ht + 1.0 * L_it + 0.5 * L_hi + 0.1 * L_kd


            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        avg = epoch_loss / len(dataloader) if len(dataloader) > 0 else 0
        print(f"[E{epoch+1}] 融合头训练 Loss = {avg:.4f}")
        if avg < best_loss:
            best_loss = avg; torch.save(fusion_head.state_dict(), best_ckpt_path)
            print(f"  -> 保存最佳融合头到: {best_ckpt_path}")

    # --- 5. 加载最佳模型 (不变) ---
    if os.path.exists(best_ckpt_path):
        print(f"加载最佳融合头: {best_ckpt_path}")
        fusion_head.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    else: print("[警告] 未找到最佳融合头模型，使用最后 epoch 权重。")
    fusion_head.eval()
    return fusion_head

def export_fused_embeddings(args, fusion_head: GatedFusion, T_emb: np.ndarray, I_emb: np.ndarray) -> np.ndarray:
    # ... (export_fused_embeddings 函数的代码保持不变) ...
    print("\n融合头训练完成。开始导出所有物品的 Embedding...")
    device = args.device; fusion_head.eval()
    dataset = TensorDataset(torch.from_numpy(T_emb).float(), torch.from_numpy(I_emb).float())
    inference_loader = DataLoader(dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers)
    all_feats = []
    with torch.no_grad():
        for batch_T, batch_I in tqdm(inference_loader, desc="Exporting Fused Embeddings"):
            T, I = batch_T.to(device), batch_I.to(device)
            H = fusion_head(T, I); all_feats.append(H.cpu())
    if not all_feats: raise RuntimeError("未能生成任何融合 Embedding。")
    Z = torch.cat(all_feats, dim=0).numpy().astype(np.float32)
    if Z.shape[0] != T_emb.shape[0]:
       print(f"[警告] 输出嵌入数量 ({Z.shape[0]}) 与物品数量 ({T_emb.shape[0]}) 不符！将修复。")
       target_len = T_emb.shape[0]; current_len = Z.shape[0]; emb_dim = Z.shape[1]
       if current_len < target_len: Z = np.concatenate([Z, np.zeros((target_len - current_len, emb_dim), dtype=np.float32)], axis=0)
       else: Z = Z[:target_len]
    print(f"融合嵌入提取完成。最终维度: {Z.shape}")
    return Z


# =================================================================
# ================== 4. 辅助函数和主程序入口 ==================
# =================================================================

def build_input_path(base_dir: str, dataset: str, modality: str, model_tag: str, pca_dim: int) -> str:
    """(新增) 辅助函数：根据组件构建输入的 .npy 文件路径"""
    emb_dir = os.path.join(base_dir, dataset, "embeddings")
    
    # 清理 model_tag (以防万一)
    safe_model_tag = model_tag.split('/')[-1].replace('/', '-').replace('\\', '-')
    
    pca_suffix = f"-pca{pca_dim}" if pca_dim > 0 else ""
    filename = f"{dataset}.emb-{modality}-{safe_model_tag}{pca_suffix}.npy"
    return os.path.join(emb_dir, filename)


def parse_args():
    ap = argparse.ArgumentParser("V3.1: 独立的多模态融合训练脚本 (自动路径构建)")
    
    # --- ✅ (修改) 必需的输入参数 ---
    ap.add_argument('--dataset', type=str, required=True, help='数据集名称 (e.g., Baby)')
    ap.add_argument('--text_model_tag', type=str, required=True, 
                        help='文本嵌入的模型标签 (例如: "text-embedding-3-large")')
    ap.add_argument('--image_model_tag', type=str, required=True, 
                        help='图像嵌入的模型标签 (例如: "clip-vit-base-patch32")')
    
    # --- ✅ (新增) 可选的输入 PCA 参数 ---
    ap.add_argument('--text_pca_dim', type=int, default=0,
                        help='(可选) 输入的文本嵌入是否经过 PCA。如果是, 指定维度 (e.g., 512)。')
    ap.add_argument('--image_pca_dim', type=int, default=0,
                        help='(可选) 输入的图像嵌入是否经过 PCA。如果是, 指定维度 (e.g., 512)。')

    # 🚨 (移除) --text_emb_path 和 --image_emb_path
    
    # --- 路径 ---
    ap.add_argument('--save_root', type=str, default='../datasets', help='保存预处理数据的根目录 (用于查找 embeddings 和保存 checkpoints/输出)')
    
    # --- 输出控制 ---
    ap.add_argument('--output_tag', type=str, default='fused-gold', 
                        help='输出文件名中的标签 (e.g., "fused-gold-v1")')
    
    # --- 训练参数 (保持不变) ---
    ap.add_argument('--fusion_epochs', type=int, default=10, help='融合头训练轮数')
    ap.add_argument('--fusion_lr', type=float, default=5e-4, help='融合头学习率')
    ap.add_argument('--fusion_weight_decay', type=float, default=0.01, help='融合头权重衰减')
    ap.add_argument('--fusion_temperature', type=float, default=0.07, help='H<->T 对齐温度')
    ap.add_argument('--fusion_out_dim', type=int, default=512, help='融合头输出维度 (0 表示与输入相同)')
    ap.add_argument('--fusion_dropout', type=float, default=0.1, help='融合头 Dropout')
    ap.add_argument('--amp', action='store_true', help='启用 AMP 混合精度训练')

    # --- 通用参数 (保持不变) ---
    ap.add_argument('--batch_size', type=int, default=4096, help='训练和推理的批处理大小')
    ap.add_argument('--num_workers', type=int, default=16, help='DataLoader num_workers')
    ap.add_argument('--pca_dim', type=int, default=0, help='(可选) 对*最终*融合嵌入应用 PCA (<=0 不降维)')
    ap.add_argument('--gpu_id', type=int, default=0, help='GPU ID (<0 使用 CPU)')
    
    return ap.parse_args()


def main():
    args = parse_args()
    args.device = set_device(args.gpu_id)
    print(f"[CFG] 融合脚本启动: {args.dataset}")

    # --- 1. ✅ (修改) 自动构建输入路径 ---
    try:
        text_emb_path = build_input_path(
            args.save_root, args.dataset, 
            "text", args.text_model_tag, args.text_pca_dim
        )
        # 假设图像模态标签为 "image"
        image_emb_path = build_input_path(
            args.save_root, args.dataset, 
            "image", args.image_model_tag, args.image_pca_dim
        )
        
        print(f"[CFG] 文本输入 (自动构建): {text_emb_path}")
        print(f"[CFG] 图像输入 (自动构建): {image_emb_path}")

        # --- 2. 加载数据 ---
        print("加载文本 Embedding...")
        T_emb = np.load(text_emb_path)
        print(f"  -> 形状: {T_emb.shape}")
        
        print("加载图像 Embedding...")
        I_emb = np.load(image_emb_path)
        print(f"  -> 形状: {I_emb.shape}")
        
    except FileNotFoundError as e:
        print(f"\n❌ 错误: 找不到输入文件 {e}")
        print("请检查 --dataset, --save_root, --text_model_tag, --image_model_tag, --text_pca_dim, --image_pca_dim 参数是否正确。")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 错误: 加载 .npy 文件失败: {e}")
        sys.exit(1)

    # --- 3. 训练融合头 ---
    trained_fusion_head = train_fusion_head(args, T_emb, I_emb)
    
    # --- 4. 导出融合嵌入 ---
    fused_embeddings = export_fused_embeddings(args, trained_fusion_head, T_emb, I_emb)

    # --- 5. ✅ (修改) 构建更具描述性的输出路径并保存 ---
    emb_dir = os.path.join(args.save_root, args.dataset, "embeddings")
    check_path(emb_dir)
    
    # 构建包含源信息的文件名
    # 例如: Baby.emb-fused-gold-(T_text-embedding-3-large-pca512+I_clip-vit-base-patch32).npy
    # ✅ 新版规范化命名
    def clean_tag(tag: str):
        """清理模型名，避免路径符号和空格"""
        return tag.split('/')[-1].replace('/', '-').replace('\\', '-').replace(' ', '').lower()

    text_tag = clean_tag(args.text_model_tag)
    image_tag = clean_tag(args.image_model_tag)

    # 文件名: Baby.emb-fused-textmodel-imagemodel.npy
    output_filename = f"{args.dataset}.emb-fused-{text_tag}-{image_tag}.npy"
    output_path = os.path.join(emb_dir, output_filename)

    
    # apply_pca_and_save 会处理最终的 PCA 逻辑并保存
    final_output_path = apply_pca_and_save(fused_embeddings, args, output_path)
    
    print(f"\n🎉 融合任务完成！最终 Embedding 已保存至: {final_output_path}")


if __name__ == "__main__":
    """
    使用示例:
    
    # 示例 1: 融合两个原始维度的 CLIP 嵌入
    python preprocessing/train_fusion_model.py \
        --dataset Baby \
        --text_model_tag "clip-vit-base-patch32" \
        --image_model_tag "clip-vit-base-patch32" \
        --output_tag "fused-gold-clip" \
        --fusion_epochs 10 \
        --batch_size 4096 \
        --fusion_out_dim 512 \
        --pca_dim 0 \
        --amp
        
    # 示例 2: 融合一个 PCA 降维过的 text-embedding-3-large 和一个原始维度的 CLIP 图像
    python preprocessing/train_fusion_model.py \
        --dataset Baby \
        --text_model_tag "text-embedding-3-large" \
        --text_pca_dim 512 \
        --image_model_tag "clip-vit-base-patch32" \
        --image_pca_dim 0 \
        --output_tag "fused-gold-T_api_pca+I_clip" \
        --fusion_epochs 10 \
        --fusion_out_dim 512 \
        --pca_dim 0 \
        --amp
    """
    main()