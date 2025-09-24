# trainer.py
import torch
from tqdm import tqdm
from typing import List

# 从 metrics.py 导入评估函数
from metrics import calculate_pos_index, recall_at_k, ndcg_at_k

def train_one_epoch(model, train_loader, optimizer, device):
    """执行一个训练周期"""
    model.train()
    total_loss = 0.0
    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch['history'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['target'].to(device)

        optimizer.zero_grad()
        loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
    return total_loss / len(train_loader)

def evaluate(model, eval_loader, topk_list: List[int], beam_size: int, code_len: int, device):
    """在评估集上评估模型性能"""
    model.eval()
    recalls = {'Recall@' + str(k): [] for k in topk_list}
    ndcgs = {'NDCG@' + str(k): [] for k in topk_list}
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch['history'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['target'].to(device)

            preds = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=beam_size,
                num_return_sequences=beam_size,
                max_new_tokens=code_len,
                early_stopping=False
            )
            
            # 形状: [B*beam, 1(BOS)+code_len] -> [B, beam, code_len]
            preds = preds[:, 1:1+code_len]
            preds = preds.view(input_ids.shape[0], beam_size, -1)

            pos_index = calculate_pos_index(preds, labels, maxk=beam_size)
            
            for k in topk_list:
                recall = recall_at_k(pos_index, k).mean().item()
                ndcg = ndcg_at_k(pos_index, k).mean().item()
                recalls['Recall@' + str(k)].append(recall)
                ndcgs['NDCG@' + str(k)].append(ndcg)
                
    avg_recalls = {k: sum(v) / len(v) for k, v in recalls.items()}
    avg_ndcgs = {k: sum(v) / len(v) for k, v in ndcgs.items()}
    
    return avg_recalls, avg_ndcgs