import argparse
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import json
from utils import load_json, set_device # 假设 utils.py 中有 load_json, set_device


# =============== 1. 数据集和 DataLoader (CF 专有) ===============

class CFDataset(Dataset):
    """用于 SASRec 训练的 Item ID 序列数据集"""
    def __init__(self, data_path, max_len):
        self.max_len = max_len
        self.sequences = []
        self.n_items = 0
        
        data_entries = []
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                # 核心修正：逐行读取 JSONL
                for line in f:
                    line = line.strip()
                    if line:
                        # 确保每行都是一个有效的 JSON 对象
                        data_entries.append(json.loads(line))
        except FileNotFoundError:
            raise FileNotFoundError(f"数据集文件未找到: {data_path}")
        except json.JSONDecodeError as e:
            # 捕获 JSONL 文件中某一行的 JSON 错误
            raise ValueError(f"JSONL 文件解析错误，请检查文件格式。错误信息: {e}")

        # 现在使用 data_entries (列表) 替代原来的 data
        for entry in data_entries:
            history = [int(x) for x in entry.get("history", [])]
            target = int(entry.get("target"))
            # 使用完整的交互序列 [i1, i2, ..., it]
            sequence = history + [target] 
            
            # 记录最大的 Item ID，用于确定嵌入矩阵大小
            self.n_items = max(self.n_items, max(sequence) if sequence else 0)
            
            # 将序列分解为多个 (输入, 目标) 样本
            for t in range(1, len(sequence)):
                # X: [i1, ..., it-1] (取 max_len)
                input_seq = sequence[:t]
                # Y: it
                target_item = sequence[t]

                # 对输入序列进行左侧 Padding/截断 (保持一致)
                if len(input_seq) > max_len:
                    input_seq = input_seq[-max_len:]
                else:
                    input_seq = [0] * (max_len - len(input_seq)) + input_seq

                self.sequences.append({
                    'input_ids': torch.tensor(input_seq, dtype=torch.long),
                    'labels': torch.tensor(target_item, dtype=torch.long)
                })

        # 最终 Item ID (从 1 开始，0 留给 Padding)
        self.n_items += 1 

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

    @staticmethod
    def collate_fn(batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        return {
            'input_ids': input_ids,
            'labels': labels
        }


# =============== 2. SASRec 模型 (简化版) ===============

class SimplifiedSASRec(nn.Module):
    """一个简化的 SASRec 模型，目标是训练 Item 嵌入矩阵 E_cf"""
    def __init__(self, n_items, max_len, d_model, n_layers, n_heads, dropout_rate):
        super().__init__()
        
        # Item 嵌入矩阵 (E_cf)
        self.item_embeddings = nn.Embedding(n_items, d_model, padding_idx=0)
        # 位置编码
        self.position_embeddings = nn.Embedding(max_len + 1, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=4 * d_model, 
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout_rate)
        
        # 最终预测层 (可用于 Logits 匹配)
        self.output_layer = nn.Linear(d_model, d_model)
        
    def forward(self, input_ids):
        seq_len = input_ids.shape[1]
        
        # 1. Item 和位置嵌入
        item_emb = self.item_embeddings(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.position_embeddings(positions)
        
        input_trans = self.dropout(item_emb + pos_emb)
        
        # 2. Attention Mask
        key_padding_mask = (input_ids == 0) # True 意味着忽略
        
        # 3. Transformer 编码
        outputs = self.transformer_encoder(input_trans, src_key_padding_mask=key_padding_mask)
        
        # 4. 提取最后一个有效 Item 的状态 (简化处理：直接取最后一个，Loss 时处理 Padding)
        
        return outputs 
    
    def get_final_state(self, outputs, input_ids):
        """提取每个序列的最后一个有效 Item 的状态"""
        # 计算序列长度 (排除 padding)
        seq_lens = (input_ids != 0).sum(dim=1) 
        
        # 提取最后一个有效时间步的 hidden state
        # 确保 seq_lens > 0，否则索引 seq_lens - 1 会导致 -1
        last_hidden_states = outputs[torch.arange(outputs.size(0)), seq_lens - 1]
        return last_hidden_states

    def calculate_loss(self, outputs, labels, input_ids):
        # 1. 提取最后一个有效状态
        final_states = self.get_final_state(outputs, input_ids) # (B, D)
        
        # 2. 预测头 (可选)
        final_states = self.output_layer(final_states)
        
        # 3. 计算与所有 Item 嵌入的 Logits (全量 Logits)
        # 排除 Padding Item (索引 0) 的嵌入
        all_item_emb = self.item_embeddings.weight[1:] 
        
        # logits: (B, N_items) - N_items 是排除 padding 后的数量
        logits = torch.matmul(final_states, all_item_emb.transpose(0, 1))
        
        # ⚠️ 修正 1: 定义 Loss 函数时设置 ignore_index=-1
        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        # 目标 Label (需要将 Item ID 还原为 0-based 索引)
        labels_0based = labels - 1 

        # ⚠️ 修正 2: 钳制所有小于 -1 的标签（以防万一）
        # 虽然 Item ID 是非负的，但为了安全，我们钳制负值
        labels_0based = torch.clamp(labels_0based, min=-1)

        # CrossEntropyLoss
        loss = loss_fct(logits, labels_0based) 
        return loss

    
# =============== 3. 主程序入口 (训练循环和提取) ===============

def train_and_extract_cf_embeddings(args):
    print(f"🔹 训练 SASRec 模型以提取 CF 嵌入: {args.dataset}")
    
    # 1. 数据加载
    data_path = os.path.join(args.root, f'{args.dataset}.train.jsonl')
    dataset = CFDataset(data_path, args.max_len)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        collate_fn=CFDataset.collate_fn
    )
    
    # 2. 模型初始化
    model = SimplifiedSASRec(
        n_items=dataset.n_items,
        max_len=args.max_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout_rate=args.dropout_rate
    ).to(args.device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 3. 训练循环
    model.train()
    for epoch in range(args.num_epochs):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", ncols=100)
        for batch in pbar:
            input_ids = batch['input_ids'].to(args.device)
            labels = batch['labels'].to(args.device)

            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = model.calculate_loss(outputs, labels, input_ids)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} 训练完成, Avg Loss: {avg_loss:.4f}")

    # 4. 提取和保存 Item 嵌入 (E_cf)
    model.eval()
    with torch.no_grad():
        # 排除 Padding ID 0
        cf_embeddings = model.item_embeddings.weight[1:].cpu().numpy()

    # 5. 保存
    emb_dir = os.path.join(args.root, "embeddings")
    os.makedirs(emb_dir, exist_ok=True)
    save_path = os.path.join(emb_dir, f"{args.dataset}.emb-cf-sasrec.npy")
    np.save(save_path, cf_embeddings)
    
    print(f"\n✅ 协同过滤嵌入已保存至: {save_path}")
    print(f"   E_cf 维度: {cf_embeddings.shape}")
    print("\n🎉 CF 嵌入生成任务完成。")

def _get_all_item_ids(args, dataset_obj):
    """
    根据数据集的实际情况获取所有 Item ID。
    假设所有 Item ID 是从 1 到 N_total_items 连续的。
    我们需要知道真实的 N_total_items。
    这里临时使用训练集最大 ID 来估算，但实际应用中应该读取一个全局 Item 列表文件。
    
    为了演示冷启动，我们假设总共有 900 个 Item，最大的 ID 是 900。
    """
    
    # ⚠️ 临时代码：假设最大 ID 就是 900
    if args.dataset == 'Musical_Instruments' and dataset_obj.n_items == 899:
        # 如果训练集最大 ID 是 899，我们假设总 Item ID 是 900
        N_total_items = 900
    else:
        # 否则使用训练集的 max_id (不理想，但演示用)
        N_total_items = dataset_obj.n_items - 1 
        
    # 返回 Item ID 列表 (从 1 到 N_total_items)
    return list(range(1, N_total_items + 1))
# =============== 4. 主程序入口和参数解析 ===============

def parse_args():
    parser = argparse.ArgumentParser(description="Generate CF Embeddings using Simplified SASRec")
    parser.add_argument('--dataset', type=str, default='Musical_Instruments', help='数据集名称')
    parser.add_argument('--root', type=str, default="../datasets", help='数据集根目录')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_len', type=int, default=50, help='序列最大长度')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader 线程数')
    
    # SASRec 模型参数
    parser.add_argument('--d_model', type=int, default=512, help='嵌入维度 (D_cf)')
    parser.add_argument('--n_layers', type=int, default=2, help='Transformer 层数')
    parser.add_argument('--n_heads', type=int, default=4, help='Attention 头数')
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--num_epochs', type=int, default=30)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.root = os.path.join(args.root, args.dataset)
    os.makedirs(args.root, exist_ok=True)
    args.device = set_device(args.gpu_id)
    
    train_and_extract_cf_embeddings(args)