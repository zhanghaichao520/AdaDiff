import json
import re
import argparse
from pathlib import Path
import numpy as np

def extract_codebooks(text):
    """提取形如 <a_x><b_x><c_x><x_x> 的 token"""
    return re.findall(r"<a_\d+><b_\d+><c_\d+><x_\d+>", text)

def recall_at_k(preds, true_item, k):
    return 1.0 if true_item in preds[:k] else 0.0

def ndcg_at_k(preds, true_item, k):
    if true_item in preds[:k]:
        rank = preds.index(true_item) + 1
        return 1.0 / np.log2(rank + 1)
    return 0.0

def evaluate_file(in_file):
    recalls_5, ndcgs_5 = [], []
    recalls_10, ndcgs_10 = [], []
    
    with open(in_file, "r", encoding="utf-8") as fin:
        for line in fin:
            data = json.loads(line)
            predict_tokens = extract_codebooks(data.get("predict", ""))
            label_tokens = extract_codebooks(data.get("label", ""))

            if not label_tokens:
                continue
            true_label = label_tokens[0]  # 只取第一个作为正样本

            recalls_5.append(recall_at_k(predict_tokens, true_label, 5))
            ndcgs_5.append(ndcg_at_k(predict_tokens, true_label, 5))
            recalls_10.append(recall_at_k(predict_tokens, true_label, 10))
            ndcgs_10.append(ndcg_at_k(predict_tokens, true_label, 10))

    return (
        np.mean(recalls_5) if recalls_5 else 0.0,
        np.mean(ndcgs_5) if ndcgs_5 else 0.0,
        np.mean(recalls_10) if recalls_10 else 0.0,
        np.mean(ndcgs_10) if ndcgs_10 else 0.0,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="预测结果的 JSONL 文件")
    args = parser.parse_args()

    in_file = Path(args.input)
    r5, n5, r10, n10 = evaluate_file(in_file)

    print("📊 评估结果：")
    print(f"Recall@5:  {r5:.4f}")
    print(f"NDCG@5:    {n5:.4f}")
    print(f"Recall@10: {r10:.4f}")
    print(f"NDCG@10:   {n10:.4f}")
