# evaluator.py
import torch

class Evaluator:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.metric2func = {'recall': self.recall_at_k, 'ndcg': self.ndcg_at_k}
        self.maxk = max(config['topk'])

    def calculate_pos_index(self, preds, labels):
        """
        preds: [B, K] (item ids)
        labels: [B]   (item ids)
        """
        preds = preds.detach().cpu().long()   # [B,K]
        labels = labels.detach().cpu().long() # [B]
        assert preds.shape[1] == self.maxk, f"preds.shape[1]={preds.shape[1]} != {self.maxk}"

        # 逐样本把标签 broadcast 到 K 列，做等值比较
        pos_index = preds.eq(labels.view(-1, 1))  # [B,K] True/False
        return pos_index

    def recall_at_k(self, pos_index, k):
        # 有一个正样本，所以 Recall@k 就是 top-k 内是否命中（0/1）
        return pos_index[:, :k].any(dim=1).float()

    def ndcg_at_k(self, pos_index, k):
        # 单一正样本时，命中位置 j 的增益=1/log2(j+2)，否则0
        B, K = pos_index.shape
        ranks = torch.arange(1, K + 1, dtype=torch.float32)  # 1..K
        gains = 1.0 / torch.log2(ranks + 1.0)                # [K]
        dcg = (pos_index[:, :k].float() * gains[:k]).sum(dim=1)
        return dcg

    def calculate_metrics(self, preds, labels):
        results = {}
        pos_index = self.calculate_pos_index(preds, labels)
        for metric in self.config['metrics']:
            for k in self.config['topk']:
                results[f"{metric}@{k}"] = self.metric2func[metric](pos_index, k)
        return results
