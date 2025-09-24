# evaluate.py
import torch

def calculate_pos_index(preds: torch.Tensor, labels: torch.Tensor, maxk: int):
    """
    计算预测结果中哪些位置是命中的。
    preds: (B, maxk, L=4)
    labels: (B, L=4)
    命中条件：前三位完全相等 && 预测dup >= 真实dup
    """
    preds = preds.detach().cpu()
    labels = labels.detach().cpu()
    assert preds.shape[1] == maxk, f'preds.shape[1] = {preds.shape[1]} != {maxk}'
    B, _, L = preds.shape
    assert L == labels.shape[1] == 4, f"Expect 4-level code, got {L} and {labels.shape[1]}"

    pos_index = torch.zeros((B, maxk), dtype=torch.bool)
    for i in range(B):
        gt = labels[i]
        gt_l012 = gt[:3].tolist()
        gt_dup  = int(gt[3].item())

        for j in range(maxk):
            pj = preds[i, j]
            pj_l012 = pj[:3].tolist()
            pj_dup  = int(pj[3].item())

            if pj_l012 == gt_l012 and pj_dup >= gt_dup:
                pos_index[i, j] = True
                break
    return pos_index

def recall_at_k(pos_index: torch.Tensor, k: int) -> torch.Tensor:
    """计算 Recall@k"""
    return pos_index[:, :k].sum(dim=1).cpu().float()

def ndcg_at_k(pos_index: torch.Tensor, k: int) -> torch.Tensor:
    """计算 NDCG@k"""
    # 假设每个样本只有一个正确答案
    ranks = torch.arange(1, pos_index.shape[-1] + 1, device=pos_index.device)
    dcg = 1.0 / torch.log2(ranks + 1)
    dcg = torch.where(pos_index, dcg, torch.tensor(0.0, dtype=torch.float, device=dcg.device))
    return dcg[:, :k].sum(dim=1).cpu().float()