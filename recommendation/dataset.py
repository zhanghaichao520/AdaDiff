import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import json
import os

# -----------------------------
# 通用工具
# -----------------------------
def pad_or_truncate(sequence, max_len, PAD_TOKEN=0):
    """对 itemID 序列做左侧 pad 或右侧截断"""
    if len(sequence) > max_len:
        return sequence[-max_len:]
    else:
        return [PAD_TOKEN] * (max_len - len(sequence)) + sequence

def item2code(code_path, vocab_sizes, bases):
    """
    【已自適應】將 codebook 的每一行 [c0, c1, ..., dup] 编码為 N 個 token。
    """
    data = np.load(code_path, allow_pickle=True)
    mat = np.vstack(data) if data.dtype == object else data
    
    num_levels = len(vocab_sizes) # ✅ 從傳入的參數動態獲取總長度
    assert mat.shape[1] == num_levels, f"Expect {num_levels} columns in codebook, got {mat.shape[1]}"

    item_to_code = {}
    code_to_item = {}

    for index, row in enumerate(mat):
        # ✅ 關鍵改動：使用迴圈處理任意長度的 code
        code_values = [int(c) for c in row]
        
        # 範圍校驗
        for i, code_val in enumerate(code_values):
            if not (0 <= code_val < vocab_sizes[i]):
                raise ValueError(f"Out-of-range code {code_val} at index {i} for row {row} with vocab_sizes={vocab_sizes}")

        # Token 偏移計算
        tokens = [code_val + bases[i] + 1 for i, code_val in enumerate(code_values)]
        
        item_id = index + 1
        item_to_code[item_id] = tokens
        code_to_item[tuple(tokens)] = item_id

    return item_to_code, code_to_item



# -----------------------------
# Parquet 读取（原逻辑）
# -----------------------------
def process_parquet(file_path, mode, max_len, PAD_TOKEN=0):
    """
    从 parquet 读取，支持 train(滑动窗口)/evaluation(只取最后一个 target)
    期望 parquet 内已有列：history(list[int])、target(int)
    """
    df = pd.read_parquet(file_path)
    df['sequence'] = df['history'].apply(lambda x: list(x)) + df['target'].apply(lambda x: [x])

    processed_data = []
    if mode == 'train':
        # 滑动窗口
        for row in df.itertuples(index=False):
            sequence = row.sequence
            for i in range(1, len(sequence)):
                processed_data.append({
                    'history': pad_or_truncate(sequence[:i], max_len, PAD_TOKEN),
                    'target': sequence[i]
                })
    elif mode == 'evaluation':
        for row in df.itertuples(index=False):
            sequence = row.sequence
            processed_data.append({
                'history': pad_or_truncate(sequence[:-1], max_len, PAD_TOKEN),
                'target': sequence[-1]
            })
    else:
        raise ValueError("Mode must be 'train' or 'evaluation'.")
    return processed_data

# -----------------------------
# JSONL 读取（新增）
# -----------------------------
def process_jsonl(file_path, max_len, PAD_TOKEN=0):
    """
    从 JSONL 读取，每行：{"user": int, "history": [itemIDs], "target": int}
    不做滑动窗口，默认已展开。
    """
    processed = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # ✅ 关键：字符串 -> int；且本数据是 0 基
            history = [int(x) for x in obj.get("history", [])]
            target = int(obj.get("target"))
            processed.append({
                "history": pad_or_truncate(history, max_len, PAD_TOKEN),
                "target": target
            })
    return processed


# -----------------------------
# 统一 Dataset
# -----------------------------
class GenRecDataset(Dataset):
    def __init__(self, dataset_path, code_path, mode, max_len, PAD_TOKEN=0,
                 vocab_sizes=None, bases=None, input_format=None):
        self.dataset_path = dataset_path
        self.code_path = code_path
        self.mode = mode
        self.max_len = max_len
        self.PAD_TOKEN = PAD_TOKEN

        assert vocab_sizes is not None and bases is not None, \
            "Must pass vocab_sizes and bases from main.py"
        self.vocab_sizes = vocab_sizes
        self.bases = bases
        self.num_levels = len(self.vocab_sizes)  # ✅ 补上


        # 加载 codebook 映射（按位偏移编码）
        self.item_to_code, self.code_to_item = item2code(
            code_path, vocab_sizes=self.vocab_sizes, bases=self.bases
        )

        # 自动判断/显式指定输入格式
        if input_format is None:
            ext = os.path.splitext(dataset_path)[1].lower()
            if ext == ".jsonl":
                input_format = "jsonl"
            elif ext == ".parquet":
                input_format = "parquet"
            else:
                raise ValueError(f"Unknown file extension for dataset_path: {dataset_path}")

        self.input_format = input_format

        # 预处理
        if self.input_format == "jsonl":
            # JSONL 不需要 mode；不滑窗
            processed = process_jsonl(
                self.dataset_path, self.max_len, self.PAD_TOKEN
            )
        elif self.input_format == "parquet":
            processed = process_parquet(
                self.dataset_path, self.mode, self.max_len, self.PAD_TOKEN
            )
        else:
            raise ValueError("input_format must be 'jsonl' or 'parquet'.")

        # itemID -> code 映射
        self.data = []
        for item in processed:
            hist_ids = item['history']
            tgt_id = item['target']

            # 把每个 itemID 映射成 num_levels 长度的 code 列表
            hist_codes = [
                self.item_to_code.get(int(x) + 1, [self.PAD_TOKEN]*self.num_levels) for x in hist_ids
            ]
            tgt_code = self.item_to_code.get(int(tgt_id) + 1, [self.PAD_TOKEN]*self.num_levels)


            self.data.append({
                'history': hist_codes,
                'target_code': tgt_code, # 將原來的 'target' 重新命名
                'target_id': tgt_id      # 新增原始的 item id
            })

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


# --------------- 简单自测 ---------------
if __name__ == "__main__":
    # JSONL 示例
    # 假设 ../data/Beauty/train.jsonl 存在，每行 {user, history, target}
    ds_jsonl = GenRecDataset(
        dataset_path='/home/wj/peiyu/GenRec/MM-RQVAE/datasets/Musical_Instruments/Musical_Instruments.train.jsonl',
        code_path='/home/wj/peiyu/GenRec/MM-RQVAE/datasets/Musical_Instruments/Musical_Instruments.emb-qwen-td.npy',
        mode='train',                 # JSONL 时忽略
        max_len=20,
        input_format='jsonl',         # 可省略，按后缀自动判断
        codebook_size=256,
        num_levels=3
    )
    print("JSONL len:", len(ds_jsonl))
    print("Sample:", ds_jsonl[0])

    # Parquet 示例（保持原行为）
    # ds_parquet = GenRecDataset(
    #     dataset_path='../data/Beauty/train.parquet',
    #     code_path='../data/Beauty/Beauty_t5_rqvae.npy',
    #     mode='train',
    #     max_len=20,
    #     input_format='parquet',
    #     codebook_size=256,
    #     num_levels=3
    # )
    # print("Parquet len:", len(ds_parquet))
    # print("Sample:", ds_parquet[0])
