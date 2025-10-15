# trainer.py
import torch
from tqdm import tqdm
from typing import List, Dict

# 從 metrics.py 導入評估函數
from metrics import calculate_pos_index, recall_at_k, ndcg_at_k

def train_one_epoch(model, train_loader, optimizer, device):
    """執行一個訓練週期"""
    model.train()
    total_loss = 0.0
    
    for batch in tqdm(train_loader, desc="Training"):
        # 1. 將整個 batch 的 Tensors 一起移動到 device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        optimizer.zero_grad()
        
        # 2. 直接將 batch 解包傳給模型，因為 key 已經對齊
        #    並使用 outputs.loss 的標準方式獲取 loss
        outputs = model.forward(batch)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
    return total_loss / len(train_loader)

def evaluate(model, eval_loader, topk_list: List[int], beam_size: int, code_len: int, device):
    """在評估集上評估模型性能"""
    model.eval()
    recalls = {f'Recall@{k}': [] for k in topk_list}
    ndcgs = {f'NDCG@{k}': [] for k in topk_list}
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            # 將輸入和目標 Tensor 移動到 device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 3. generate 方法的參數也是 input_ids，保持一致
            preds = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=beam_size,
                num_return_sequences=beam_size,
                max_new_tokens=code_len,
                early_stopping=False # 在定長生成時通常設為 False
            )
            
            # 移除開頭的 BOS token，並 reshape
            preds = preds[:, 1:1 + code_len]
            preds = preds.view(input_ids.shape[0], beam_size, -1)
            
            pos_index = calculate_pos_index(preds, labels, maxk=beam_size)
            
            for k in topk_list:
                recall = recall_at_k(pos_index, k).mean().item()
                ndcg = ndcg_at_k(pos_index, k).mean().item()
                recalls[f'Recall@{k}'].append(recall)
                ndcgs[f'NDCG@{k}'].append(ndcg)
                
    # 計算每個指標的平均值
    avg_recalls = {k: sum(v) / len(v) for k, v in recalls.items()}
    avg_ndcgs = {k: sum(v) / len(v) for k, v in ndcgs.items()}
    
    return avg_recalls, avg_ndcgs