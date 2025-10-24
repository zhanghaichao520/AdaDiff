# trainer.py

import torch
from tqdm import tqdm
from typing import List, Dict
import logging

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
    【已修正】在评估集上评估模型性能。
    - 此版本修复了“平均值的平均值”错误。
    - 它现在聚合真实的 总和(sum) 和 总计数(count) 来计算准确的平均指标。
    """
    model.eval()
    
    # ✅ 1. 初始化字典来收集 *总和* (不再是 list)
    total_metrics = {f'Recall@{k}': 0.0 for k in topk_list}
    total_metrics.update({f'NDCG@{k}': 0.0 for k in topk_list})
    total_count = 0.0 # ✅ 2. 初始化总计数
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            # 将数据移动到设备
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # ✅ 3. 调用模型自身的评估方法
            # 我们现在 *期望* batch_metrics 是一个包含 'count' 和 *指标总和* 的字典
            # e.g., {'count': 256, 'Recall@10': 15.0, 'NDCG@10': 7.5}
            batch_metrics = model.evaluate_step(
                batch=batch, 
                topk_list=topk_list
            )
            
            # ✅ 4. 累加总和与计数
            # 从返回的字典中弹出 'count' 键
            current_batch_size = batch_metrics.pop('count', 0)
            
            if current_batch_size == 0:
                # 增加一个保护措施，以防模型忘记返回 'count'
                current_batch_size = batch.get('input_ids', torch.empty(0)).shape[0]
                if current_batch_size > 0:
                    # 使用 logging 模块打印警告
                    logging.warning("model.evaluate_step() did not return 'count'. Inferring from batch size.")

            total_count += current_batch_size
            
            # 累加指标的总和
            for metric, value in batch_metrics.items():
                if metric in total_metrics:
                    total_metrics[metric] += value # 不再是 .append()
    
    # ✅ 5. 计算所有批次的 *真实* 平均指标 (总和 / 总计数)
    avg_metrics = {k: v / total_count if total_count > 0 else 0.0 
                   for k, v in total_metrics.items()}
    
    # 返回一个包含所有平均指标的字典
    return avg_metrics