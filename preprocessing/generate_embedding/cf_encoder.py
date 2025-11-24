# preprocessing/generate_embeddings/cf_encoder.py

import os
import sys
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

# =================================================================
# 1. 导入父目录共享函数 (utils.py)
# =================================================================
try:
    # 将父目录 (preprocessing/) 添加到 Python 路径
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # 注意：这里假设 utils.py 在 preprocessing/ 目录或其父级目录
    # 如果 utils.py 在项目根目录，可能需要再往上一层
    # 这里尝试适配常见的结构：
    # root/
    #   preprocessing/
    #     generate_embeddings/
    #       cf_encoder.py
    #   utils.py
    
    # 尝试从上两级导入 (如果 utils 在 root 下)
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    from utils import load_json, set_device
    print("[INFO] cf_encoder: 成功导入 utils 模块。")
except ImportError:
    # 如果找不到，定义简单的替代函数以防报错 (仅作备用)
    print("[WARN] 无法导入 utils.py，使用本地回退函数。")
    def set_device(gpu_id):
        return torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() and gpu_id >= 0 else "cpu")

# =================================================================
# 2. SASRec 数据集
# =================================================================

class SASRecDataset(Dataset):
    def __init__(self, data_path, max_seq_len): 
        self.data_path = data_path
        self.max_seq_len = max_seq_len
        self.lines = []
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.lines.append(line.strip())
        except FileNotFoundError:
             print(f"错误：SASRec 数据文件未找到: {data_path}")
             raise
        except Exception as e:
             print(f"错误：读取文件失败: {e}")
             raise

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        try:
            line = self.lines[idx]
            data = json.loads(line)
            # 假设数据中的 ID 是从 0 开始的，模型中 reserved 0 作为 padding
            # 所以输入 ID + 1
            history_ids = [int(i) + 1 for i in data["history"]] 
            target_id = int(data["target"]) + 1 
            
            seq = history_ids[-self.max_seq_len:]
            seq_len = len(seq)
            padding_len = self.max_seq_len - seq_len
            
            # 左填充还是右填充？SASRec通常是左侧是历史，如果长度不够，
            # 为了方便处理，通常把有效数据放在最后，前面补0，或者后面补0配合mask
            # 这里采用：[有效序列, 0, 0] (右填充) 并配合 seq_len 使用
            seq = seq + [0] * padding_len
            
            return torch.tensor(seq, dtype=torch.long), \
                   torch.tensor(target_id, dtype=torch.long), \
                   torch.tensor(seq_len, dtype=torch.long)
        except Exception as e:
             print(f"警告：数据解析错误 (行 {idx}): {e}")
             return torch.zeros(self.max_seq_len, dtype=torch.long), \
                    torch.tensor(0, dtype=torch.long), \
                    torch.tensor(0, dtype=torch.long)

# =================================================================
# 3. SASRec 模型
# =================================================================

class SASRecModel(nn.Module):
    def __init__(self, n_items, hidden_dim, max_seq_len, n_layers, n_heads, dropout=0.1):
        super(SASRecModel, self).__init__()
        self.n_items = n_items
        self.hidden_dim = hidden_dim
        
        # padding_idx=0, 所以 embedding size 是 n_items + 1
        self.item_embedding = nn.Embedding(self.n_items + 1, hidden_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.emb_dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, activation='gelu', norm_first=True
        )
        encoder_norm = nn.LayerNorm(hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, norm=encoder_norm)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # 初始化权重 (Xavier initialization usually good for Transformers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, seq_lengths):
        # seq_lengths: [batch_size]
        # item_seq: [batch_size, max_len]
        
        # 边界检查
        seq_lengths = torch.clamp(seq_lengths, min=1)
        
        # 生成 Mask: True 表示是 padding (不需要关注的位置)
        # item_seq == 0 的位置是 padding
        padding_mask = (item_seq == 0)

        item_emb = self.item_embedding(item_seq)
        
        # Positional Embedding
        pos_ids = torch.arange(item_seq.size(1), device=item_seq.device).unsqueeze(0)
        pos_emb = self.position_embedding(pos_ids)
        
        x = self.emb_dropout(item_emb + pos_emb)
        
        # Transformer Encoder
        # src_key_padding_mask: [batch, seq_len] (True for padding)
        transformer_out = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        transformer_out = self.layer_norm(transformer_out) 
        
        # 取序列中最后一个有效 item 的 embedding 作为 User Embedding
        # gather indices: [batch, 1, hidden]
        batch_indices = torch.arange(transformer_out.size(0), device=transformer_out.device)
        last_item_indices = seq_lengths - 1
        
        user_emb = transformer_out[batch_indices, last_item_indices, :] # [batch, hidden]
        
        # 计算 Logits (预测下一个 item)
        # item_embedding.weight: [n_items+1, hidden]
        # 我们通常希望计算所有物品的得分
        logits = user_emb @ self.item_embedding.weight.T # [batch, n_items+1]
        
        return logits

# =================================================================
# 4. 辅助工具：EarlyStopping 和 Metrics
# =================================================================

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = -np.inf # 假设指标越大越好 (如 NDCG)
        self.early_stop = False
        self.best_model_state = None
        self.delta = delta

    def __call__(self, score, model):
        if score > self.best_score + self.delta:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'   [EarlyStop] Counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, score, model):
        if self.verbose:
            print(f'   [EarlyStop] Metric improved to {score:.6f}. Caching model...')
        # 保存参数到 CPU 内存，避免占用显存
        self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

def calculate_metrics(logits, target, k_list=[10, 20]):
    """计算 Recall@K 和 NDCG@K"""
    # logits: [batch, n_items+1]
    # target: [batch]
    
    # 移除 padding (index 0) 的影响，将 logit[0] 设为负无穷
    logits[:, 0] = -float('inf')
    
    batch_size = logits.size(0)
    max_k = max(k_list)
    
    # 获取 TopK 索引
    _, topk_indices = torch.topk(logits, max_k, dim=-1) # [batch, max_k]
    
    target = target.view(-1, 1) # [batch, 1]
    hit = (topk_indices == target) # [batch, max_k]
    
    metrics = {}
    for k in k_list:
        hit_k = hit[:, :k]
        num_hit = hit_k.sum().item()
        
        # Recall
        metrics[f'Recall@{k}'] = num_hit / batch_size
        
        # NDCG
        hit_positions = hit_k.nonzero(as_tuple=False)[:, 1] # rank (0-based)
        if len(hit_positions) > 0:
            # log2(rank + 2) because rank is 0-based
            dcg = 1.0 / torch.log2(hit_positions.float() + 2.0)
            metrics[f'NDCG@{k}'] = dcg.sum().item() / batch_size
        else:
            metrics[f'NDCG@{k}'] = 0.0
            
    return metrics

def evaluate(model, dataloader, device, k_list=[10, 20]):
    model.eval()
    total_metrics = {f'Recall@{k}': 0.0 for k in k_list}
    total_metrics.update({f'NDCG@{k}': 0.0 for k in k_list})
    n_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            seq, target, seq_len = [b.to(device) for b in batch]
            
            # 过滤无效 batch
            valid = seq_len > 0
            if not valid.any(): continue
            seq = seq[valid]
            target = target[valid]
            seq_len = seq_len[valid]
            
            logits = model(seq, seq_len)
            batch_metrics = calculate_metrics(logits, target, k_list)
            
            for k, v in batch_metrics.items():
                total_metrics[k] += v
            n_batches += 1
            
    if n_batches > 0:
        for k in total_metrics:
            total_metrics[k] /= n_batches
            
    return total_metrics

# =================================================================
# 5. 主流程函数
# =================================================================

def train_and_extract_sasrec(args, n_items: int) -> np.ndarray:
    """
    训练 SASRec 模型并提取物品嵌入。
    """
    print(f"\n🔹 [SASRec] 开始训练协同过滤模型 (Target Items: {n_items})...")
    device = args.device

    # --- 1. 路径设置 ---
    data_dir = os.path.join(args.save_root, args.dataset) 
    train_path = os.path.join(data_dir, f"{args.dataset}.train.jsonl")
    valid_path = os.path.join(data_dir, f"{args.dataset}.valid.jsonl")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"错误: 找不到 SASRec 训练文件 {train_path}")

    has_valid = os.path.exists(valid_path)
    if not has_valid:
        print("⚠️ [警告] 未找到验证集 (.valid.jsonl)。Early Stopping 将被禁用，仅按 Epochs 训练。")

    # --- 2. DataLoader ---
    num_workers = getattr(args, 'num_workers', 4)
    print(f"Loading data (Workers: {num_workers})...")
    
    train_dataset = SASRecDataset(train_path, args.sasrec_max_seq_len) 
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, 
        shuffle=True, num_workers=num_workers, pin_memory=True
    )
    
    valid_loader = None
    if has_valid:
        valid_dataset = SASRecDataset(valid_path, args.sasrec_max_seq_len)
        # 验证时 Batch 可以稍微大一点，因为不需要反向传播
        valid_loader = DataLoader(
            valid_dataset, batch_size=args.batch_size, 
            shuffle=False, num_workers=num_workers, pin_memory=True
        )

    # --- 3. 模型构建 ---
    model = SASRecModel(
        n_items=n_items,
        hidden_dim=args.sasrec_hidden_dim,
        max_seq_len=args.sasrec_max_seq_len,
        n_layers=args.sasrec_n_layers,
        n_heads=args.sasrec_n_heads,
        dropout=args.sasrec_dropout
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.sasrec_lr, weight_decay=args.sasrec_weight_decay)
    
    # Early Stopping 设置
    patience = getattr(args, 'patience', 5)
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    # --- 4. 训练循环 ---
    print(f"开始训练 (Epochs: {args.sasrec_epochs}, Patience: {patience})...")
    start_time = time.time()
    
    for epoch in range(1, args.sasrec_epochs + 1):
        # Train
        model.train()
        total_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.sasrec_epochs} [Train]", leave=False)
        
        for batch in train_pbar:
            seq, target, seq_len = [b.to(device) for b in batch]
            
            valid_mask = (seq_len > 0)
            if not valid_mask.any(): continue
            seq = seq[valid_mask]
            target = target[valid_mask]
            seq_len = seq_len[valid_mask]

            optimizer.zero_grad()
            logits = model(seq, seq_len)
            loss = criterion(logits, target)
            
            if torch.isnan(loss):
                continue
                
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            train_pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
        
        # Valid
        log_msg = f"Epoch {epoch} | Loss: {avg_loss:.4f}"
        
        if has_valid:
            val_metrics = evaluate(model, valid_loader, device)
            metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
            log_msg += f" | {metric_str}"
            print(log_msg)
            
            # 使用 NDCG@10 作为 Early Stop 的主要指标
            monitor_score = val_metrics.get('NDCG@10', 0)
            early_stopping(monitor_score, model)
            
            if early_stopping.early_stop:
                print(f"🛑 Early stopping triggered at Epoch {epoch}!")
                break
        else:
            print(log_msg)
            # 如果没有验证集，我们也保存当前模型为“最佳”
            early_stopping.best_model_state = model.state_dict()

    train_duration = time.time() - start_time
    print(f"SASRec 训练完成. 耗时: {train_duration:.2f}s")

    # --- 5. 恢复最佳模型并提取 Embedding ---
    if has_valid and early_stopping.best_model_state is not None:
        print("正在恢复验证集表现最佳的模型参数...")
        model.load_state_dict(early_stopping.best_model_state)
    
    print("正在提取物品嵌入 (Item Embeddings)...")
    model.eval()
    try:
        with torch.no_grad():
            # 获取 embedding weight: [n_items + 1, hidden_dim]
            all_embeddings = model.item_embedding.weight.data.cpu().numpy()
        
        # 去除 index 0 (padding)
        # 我们的 item ID 是 1~N，对应 embedding 索引 1~N
        # 结果需要是 [n_items, hidden_dim]
        embeddings = all_embeddings[1:] 
        
        print(f"原始 Embedding 形状: {embeddings.shape}")
        
        # 维度校验与修复
        if embeddings.shape[0] != n_items:
            print(f"[警告] 嵌入数量 ({embeddings.shape[0]}) 与 n_items ({n_items}) 不一致。")
            if embeddings.shape[0] < n_items:
                print(" -> 填充零向量...")
                pad = np.zeros((n_items - embeddings.shape[0], embeddings.shape[1]), dtype=embeddings.dtype)
                embeddings = np.concatenate([embeddings, pad], axis=0)
            else:
                print(" -> 截断...")
                embeddings = embeddings[:n_items]
                
        return embeddings.astype(np.float32)

    except Exception as e:
        print(f"提取嵌入失败: {e}")
        raise