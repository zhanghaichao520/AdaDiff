#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UniGenRec Fusion Module - Cross-Modal Attention (SOTA)
------------------------------------------------------
功能：
1. 读取预提取的 Text Embedding (Qwen/T5) 和 Image Embedding (CLIP)。
2. 训练 Cross-Modal Attention 网络 (Text Queries Image)。
3. 将融合后的 Embedding 导出为 .npy 文件，供 RQ-VAE 使用。

架构：
Query = Text, Key/Value = Image
Loss = InfoNCE(Fused, Text) + InfoNCE(Fused, Image)
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# --- 尝试导入 utils (路径兼容性处理) ---
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from utils import set_device, check_path, apply_pca_and_save
    print("[INFO] 成功从 utils.py 导入共享函数。")
except ImportError:
    try:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils import set_device, check_path, apply_pca_and_save
        print("[INFO] 成功从父目录导入 utils.py。")
    except ImportError:
        print("[ERROR] 无法找到 utils.py，请确保文件结构正确。")
        sys.exit(1)


# =================================================================
# 1. 模型定义: Cross-Modal Attention Fusion
# =================================================================
class CrossModalAttentionFusion(nn.Module):
    def __init__(self, in_dim_text, in_dim_image, out_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.out_dim = out_dim
        
        # 1. 独立投影层：将不同维度的图文映射到同一维度
        self.text_proj = nn.Linear(in_dim_text, out_dim)
        self.img_proj = nn.Linear(in_dim_image, out_dim)
        
        # 2. 交叉注意力层 (Cross Attention)
        # batch_first=True -> (Batch, Seq, Dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=out_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        # 3. 前馈网络 (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 4, out_dim)
        )
        
        # 4. Norm 层
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.norm_out = nn.LayerNorm(out_dim)

    def forward(self, txt, img):
        """
        txt: (Batch, in_dim_text)
        img: (Batch, in_dim_image)
        """
        # --- A. 投影并增加序列维度 (Batch, 1, Dim) ---
        q = self.text_proj(txt).unsqueeze(1) 
        k_v = self.img_proj(img).unsqueeze(1) 
        
        # --- B. Cross Attention: Text queries Image ---
        # attn_output: (Batch, 1, Dim)
        attn_output, _ = self.cross_attn(query=q, key=k_v, value=k_v)
        
        # --- C. 残差 + FFN ---
        # 残差加在 Query (文本) 上 -> "增强文本"
        h = self.norm1(q + attn_output)
        h = self.norm2(h + self.ffn(h))
        
        # --- D. 输出 ---
        # 移除序列维度 -> (Batch, Dim)
        return self.norm_out(h.squeeze(1))

# =================================================================
# 2. Loss 函数: InfoNCE
# =================================================================
def info_nce_loss(features_a, features_b, temperature=0.07):
    """计算两个特征集之间的对比损失"""
    # 归一化
    a = F.normalize(features_a, dim=-1)
    b = F.normalize(features_b, dim=-1)
    
    # 相似度矩阵 (Batch, Batch)
    logits = torch.matmul(a, b.T) / temperature
    
    # 标签是对角线 (self-supervised: i-th item in A matches i-th item in B)
    labels = torch.arange(a.shape[0], device=a.device)
    
    return F.cross_entropy(logits, labels)

# =================================================================
# 3. 辅助函数: 路径构建
# =================================================================
def build_input_path(base_dir, dataset, modality, model_tag, pca_dim):
    """自动构建输入 .npy 文件路径"""
    emb_dir = os.path.join(base_dir, dataset, "embeddings")
    # 清理标签
    safe_tag = model_tag.split('/')[-1].replace('/', '-').replace('\\', '-')
    pca_suffix = f"-pca{pca_dim}" if pca_dim > 0 else ""
    filename = f"{dataset}.emb-{modality}-{safe_tag}{pca_suffix}.npy"
    return os.path.join(emb_dir, filename)

def clean_tag_name(tag):
    return tag.split('/')[-1].replace('/', '-').replace(' ', '').lower()

# =================================================================
# 4. 主流程
# =================================================================
def main():
    parser = argparse.ArgumentParser("UniGenRec Attention Fusion Training")
    
    # --- 输入参数 ---
    parser.add_argument('--dataset', type=str, required=True, help='数据集名称 (e.g., Baby)')
    parser.add_argument('--text_model_tag', type=str, required=True, help='文本模型标签 (e.g., text-embedding-3-large)')
    parser.add_argument('--image_model_tag', type=str, required=True, help='图像模型标签 (e.g., clip-vit-base-patch32)')
    parser.add_argument('--save_root', type=str, default='../datasets', help='数据根目录')
    
    # --- PCA 选项 (针对输入) ---
    parser.add_argument('--text_pca_dim', type=int, default=0, help='输入文本是否已 PCA')
    parser.add_argument('--image_pca_dim', type=int, default=0, help='输入图像是否已 PCA')

    # --- 模型与训练参数 ---
    parser.add_argument('--fusion_out_dim', type=int, default=512, help='融合输出维度 (建议与 RQ-VAE 输入一致)')
    parser.add_argument('--num_heads', type=int, default=8, help='Attention 头数')
    parser.add_argument('--epochs', type=int, default=15, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch Size')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--temp', type=float, default=0.07, help='InfoNCE 温度')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--output_tag', type=str, default='attn-fusion', help='输出文件名标签')

    args = parser.parse_args()
    device = set_device(args.gpu_id)

    # -----------------------------------------------------------
    # Step 1: 加载数据
    # -----------------------------------------------------------
    print(f"\n[1/4] 加载输入 Embedding ({args.dataset})...")
    try:
        text_path = build_input_path(args.save_root, args.dataset, "text", args.text_model_tag, args.text_pca_dim)
        image_path = build_input_path(args.save_root, args.dataset, "image", args.image_model_tag, args.image_pca_dim)
        
        print(f"  -> Text: {os.path.basename(text_path)}")
        print(f"  -> Image: {os.path.basename(image_path)}")
        
        T_data = np.load(text_path).astype(np.float32)
        I_data = np.load(image_path).astype(np.float32)

        if T_data.shape[0] != I_data.shape[0]:
            raise ValueError(f"数量不匹配: Text={T_data.shape[0]}, Image={I_data.shape[0]}")
        
        print(f"  -> 数据加载成功。样本数: {T_data.shape[0]}")
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        sys.exit(1)

    # -----------------------------------------------------------
    # Step 2: 初始化模型
    # -----------------------------------------------------------
    print(f"\n[2/4] 初始化 Cross-Modal Attention 模型...")
    model = CrossModalAttentionFusion(
        in_dim_text=T_data.shape[1],
        in_dim_image=I_data.shape[1],
        out_dim=args.fusion_out_dim,
        num_heads=args.num_heads
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # 数据集
    dataset = TensorDataset(torch.from_numpy(T_data), torch.from_numpy(I_data))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # -----------------------------------------------------------
    # Step 3: 训练循环
    # -----------------------------------------------------------
    print(f"\n[3/4] 开始训练 (Epochs: {args.epochs})...")
    model.train()
    
    for epoch in range(args.epochs):
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for b_txt, b_img in pbar:
            b_txt, b_img = b_txt.to(device), b_img.to(device)
            
            # Forward
            fused = model(b_txt, b_img) # (B, out_dim)
            
            # 获取投影后的原始特征 (作为 Target)
            target_txt = model.text_proj(b_txt)
            target_img = model.img_proj(b_img)
            
            # Loss 计算
            # 1. Fidelity: 融合后应该依然像文本 (保真)
            loss_txt = info_nce_loss(fused, target_txt, args.temp)
            
            # 2. Injection: 融合后应该包含图像信息 (注入)
            loss_img = info_nce_loss(fused, target_img, args.temp)
            
            # 总 Loss: 稍微侧重文本保真度，因为文本通常是主导模态
            loss = 0.7 * loss_txt + 0.3 * loss_img
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
    # -----------------------------------------------------------
    # Step 4: 导出与保存
    # -----------------------------------------------------------
    print(f"\n[4/4] 导出融合 Embedding...")
    model.eval()
    export_loader = DataLoader(dataset, batch_size=args.batch_size * 2, shuffle=False)
    
    all_fused = []
    with torch.no_grad():
        for b_txt, b_img in tqdm(export_loader, desc="Exporting"):
            b_txt, b_img = b_txt.to(device), b_img.to(device)
            out = model(b_txt, b_img)
            all_fused.append(out.cpu().numpy())
            
    final_emb = np.concatenate(all_fused, axis=0)
    
    # 构建输出路径
    emb_dir = os.path.join(args.save_root, args.dataset, "embeddings")
    t_tag = clean_tag_name(args.text_model_tag)
    i_tag = clean_tag_name(args.image_model_tag)
    
    # 文件名: Baby.emb-fused-attn-fusion-text_tag-image_tag.npy
    filename = f"{args.dataset}.emb-fused-{args.output_tag}-{t_tag}-{i_tag}.npy"
    save_path = os.path.join(emb_dir, filename)
    
    # 保存 (apply_pca_and_save 会处理保存逻辑，如果不降维直接存)
    # 这里的 args.pca_dim 指的是 *最终输出* 是否要再降维，通常不需要了，因为模型输出就是 512
    saved_path = apply_pca_and_save(final_emb, argparse.Namespace(pca_dim=0), save_path)
    
    print(f"🎉 完成！融合文件已保存: {saved_path}")
    print(f"👉 下一步: 请将此文件路径配置为 RQ-VAE 的输入。")

if __name__ == "__main__":
    main()