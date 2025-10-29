# preprocessing/generate_embeddings/cf_embedding.py

import os
import sys
import argparse
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

# (用于导入上级目录的 utils)
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import load_json, set_device
except ImportError:
    print("警告: 无法自动导入 utils.py。将使用本地的 load_json。")
    
    def load_json(file):
        if not os.path.exists(file): return {}
        with open(file, 'r') as f:
            return json.load(f)
            
    def set_device(gpu_id):
        if torch.cuda.is_available() and gpu_id >= 0:
            return torch.device(f'cuda:{gpu_id}')
        else:
            return torch.device('cpu')

# =================================================================
# ================== 1. 数据集和加载器 ==================
# =================================================================

class SASRecDataset(Dataset):
    """
    用于加载 .jsonl 文件的 PyTorch 数据集。
    
    .jsonl 格式: {"user": "0", "history": ["2803"], "target": "6913"}
    """
    def __init__(self, data_path, max_seq_len, n_items):
        self.data_path = data_path
        self.max_seq_len = max_seq_len
        self.n_items = n_items
        
        self.lines = []
        with open(data_path, 'r') as f:
            for line in f:
                self.lines.append(line.strip())

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        # 1. 加载数据
        line = self.lines[idx]
        data = json.loads(line)
        
        # 2. 转换 ID
        # (重要) 我们将所有 item_id + 1。
        # 因为 0 号索引将保留给 [PAD] 填充标记。
        history_ids = [int(i) + 1 for i in data["history"]]
        target_id = int(data["target"]) + 1
        
        # 3. 截断/填充序列
        
        # 获取序列
        seq = history_ids[-self.max_seq_len:]
        
        # 获取真实长度（用于后续模型提取最后一个 item）
        seq_len = len(seq)
        
        # 填充
        padding_len = self.max_seq_len - seq_len
        seq = seq + [0] * padding_len # 使用 0 作为 [PAD]
        
        # 4. 转换为 Tensors
        # seq 是一长串 0-padded 序列
        # target 是单一的目标 item
        # seq_len 是 history 的真实长度
        return torch.tensor(seq, dtype=torch.long), \
               torch.tensor(target_id, dtype=torch.long), \
               torch.tensor(seq_len, dtype=torch.long)

# =================================================================
# ================== 2. SASRec 模型 ==================
# =================================================================

class SASRecModel(nn.Module):
    def __init__(self, n_items, hidden_dim, max_seq_len, n_layers, n_heads, dropout=0.1):
        super(SASRecModel, self).__init__()
        self.n_items = n_items
        self.hidden_dim = hidden_dim
        
        # n_items + 1 (因为 0 是 [PAD] 标记)
        self.item_embedding = nn.Embedding(self.n_items + 1, hidden_dim, padding_idx=0)
        
        # 位置嵌入
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        
        self.emb_dropout = nn.Dropout(dropout)
        
        # Transformer 编码器
        # (使用 batch_first=True 以便输入是 [B, L, D])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True, 
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, item_seq, seq_lengths):
        """
        Args:
            item_seq (torch.Tensor): [B, L] 0-padded item id 序列
            seq_lengths (torch.Tensor): [B] 每个序列的真实长度

        Returns:
            torch.Tensor: [B, n_items+1] 预测 logits
        """
        # 1. 嵌入
        item_emb = self.item_embedding(item_seq) # [B, L, D]
        
        # 位置嵌入
        # [L] -> [1, L] -> [B, L] -> [B, L, D]
        pos_ids = torch.arange(item_seq.size(1), device=item_seq.device).unsqueeze(0)
        pos_emb = self.position_embedding(pos_ids)
        
        x = item_emb + pos_emb
        x = self.emb_dropout(x) # [B, L, D]
        
        # 2. Transformer 
        # (创建 padding mask)
        # item_seq == 0 的地方是 True, 其他地方是 False
        padding_mask = (item_seq == 0) # [B, L]
        
        # Transformer 期望 True 的地方是 "被遮蔽的"
        transformer_out = self.transformer_encoder(
            x, 
            src_key_padding_mask=padding_mask
        ) # [B, L, D]
        
        transformer_out = self.layer_norm(transformer_out)
        
        # 3. 提取最后一个有效 item 的表示
        # 我们需要从 [B, L, D] 中提取 [B, D]
        # B = 批次大小
        batch_indices = torch.arange(transformer_out.size(0), device=transformer_out.device)
        
        # 真实长度 - 1 = 最后一个 item 的索引
        # (因为 seq_len 是 1-based, 索引是 0-based)
        last_item_indices = seq_lengths - 1 # [B]
        
        # 使用高级索引
        last_item_emb = transformer_out[batch_indices, last_item_indices, :] # [B, D]
        
        # 4. 计算 Logits (预测)
        # 使用 "tied weights": 预测层的权重就是嵌入矩阵的转置
        logits = last_item_emb @ self.item_embedding.weight.T # [B, D] @ [D, n_items+1] -> [B, n_items+1]
        
        return logits

# =================================================================
# ================== 3. 主训练流程 ==================
# =================================================================

def main(args):
    # 1. 设置
    device = set_device(args.gpu_id)
    print(f"使用设备: {device}")
    
    # 2. 构建路径
    data_dir = os.path.join(args.data_root, args.dataset)
    train_path = os.path.join(data_dir, f"{args.dataset}.train.jsonl")
    item_meta_path = os.path.join(data_dir, f"{args.dataset}.item.json")
    
    if not os.path.exists(train_path):
        print(f"错误: 找不到训练文件 {train_path}")
        sys.exit(1)
        
    print("加载 item 元数据以获取 n_items...")
    item_meta = load_json(item_meta_path)
    if not item_meta:
        print(f"错误: 找不到或无法加载 item 元数据 {item_meta_path}")
        sys.exit(1)
    
    n_items = len(item_meta)
    print(f"数据集: {args.dataset}, 物品总数: {n_items}")

    # 3. 创建 Dataset 和 DataLoader
    print("加载训练数据...")
    train_dataset = SASRecDataset(train_path, args.max_seq_len, n_items)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # 4. 初始化模型、损失函数和优化器
    model = SASRecModel(
        n_items=n_items,
        hidden_dim=args.hidden_dim,
        max_seq_len=args.max_seq_len,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout
    ).to(device)
    
    # CrossEntropyLoss 自动处理 logits (不需要 softmax)
    # 我们预测 [B, n_items+1]
    # 目标是 [B] (且 target ID 已经是 1-based 了)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    print("开始训练...")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        
        for batch in pbar:
            # seq: [B, L], target: [B], seq_len: [B]
            seq, target, seq_len = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            
            # 预测 [B, n_items+1]
            logits = model(seq, seq_len)
            
            # 计算损失 (target 是 [B])
            loss = criterion(logits, target)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"Loss": loss.item()})
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} 完成. 平均损失: {avg_loss:.4f}")

    print("训练完成.")
    
    # 5. 提取和保存嵌入
    print("正在提取物品嵌入...")
    model.eval()
    
    # 提取整个嵌入矩阵 [n_items+1, D]
    embeddings = model.item_embedding.weight.data.cpu().numpy()
    
    # (重要) 移除 0 号索引的 [PAD] 嵌入
    embeddings = embeddings[1:] # [n_items, D]
    
    print(f"提取的嵌入维度: {embeddings.shape}")
    
    # 6. 保存
    save_dir = os.path.join(data_dir, "embeddings")
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, f"{args.dataset}.emb-cf-sasrec.npy")
    np.save(save_path, embeddings)
    
    print(f"🎉 协同过滤嵌入已保存到: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="使用 SASRec 训练协同过滤嵌入")
    
    # --- 数据和路径 ---
    parser.add_argument('--dataset', type=str, required=True,
                        help='数据集名称 (例如: Baby, ml-1m)')
    parser.add_argument('--data_root', type=str, default="../datasets",
                        help='数据集的根目录')
    
    # --- 模型超参数 ---
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='嵌入和模型的隐藏维度')
    parser.add_argument('--max_seq_len', type=int, default=50,
                        help='序列的最大长度')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='Transformer 编码器的层数')
    parser.add_argument('--n_heads', type=int, default=2,
                        help='Transformer 的头数')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout 概率')
    
    # --- 训练超参数 ---
    parser.add_argument('--epochs', type=int, default=30,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='批量大小')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='L2 正则化 (权重衰减)')
    
    # --- 设备 ---
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='要使用的 GPU ID (如果 < 0 则使用 CPU)')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)