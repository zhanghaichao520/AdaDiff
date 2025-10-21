# trainer.py

import torch
from tqdm import tqdm
from typing import List, Dict

# trainer 不再需要关心 metrics 的具体实现
# from metrics import ... (这些引用可以删除了)

def train_one_epoch(model, train_loader, optimizer, device):
    """执行一个训练周期 (此函数不变)"""
    model.train()
    total_loss = 0.0
    for batch in tqdm(train_loader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model.forward(batch)
        if isinstance(outputs, dict):
            loss = outputs['loss']  # 處理 RPG 這類返回字典的模型
        else:
            loss = outputs.loss      # 處理 TIGER 這類返回物件的模型
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, eval_loader, topk_list: List[int], device) -> Dict[str, float]:
    """
    【已解耦】在评估集上评估模型性能。
    它只负责调用 model.evaluate_step 并聚合结果。
    """
    model.eval()
    
    # 初始化一个字典来收集所有批次的结果
    # e.g., {'Recall@10': [0.5, 0.6], 'NDCG@10': [0.4, 0.45], ...}
    total_metrics = {f'Recall@{k}': [] for k in topk_list}
    total_metrics.update({f'NDCG@{k}': [] for k in topk_list})
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            # 将数据移动到设备
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # ✨ 核心改动：直接调用模型自身的评估方法 ✨
            batch_metrics = model.evaluate_step(
                batch=batch, 
                topk_list=topk_list
            )
            
            # 收集当前批次的结果
            for metric, value in batch_metrics.items():
                if metric in total_metrics:
                    total_metrics[metric].append(value)
    
    # 计算所有批次的平均指标
    avg_metrics = {k: (sum(v) / len(v)) if v else 0.0 for k, v in total_metrics.items()}
    
    # 返回一个包含所有平均指标的字典
    return avg_metrics