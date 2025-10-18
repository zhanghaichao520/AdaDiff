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
    """
    【最終推薦版 - 兼容 TIGER & RPG】
    此版本在初始化時根據 config 加載原始數據和 item2code 映射，
    並在 __getitem__ 中返回所有下游 collate_fn 可能需要的數據格式。
    """
    def __init__(self, config: dict, mode: str):
        self.config = config
        self.mode = mode
        self.dataset_path = self.config[f'{mode}_json']
        self.max_len = self.config['model_params']['max_len']
        # ✅ 需要 PAD_TOKEN ID
        self.PAD_TOKEN_ID = self.config['token_params']['pad_token_id'] 
        
        # ✅ 關鍵改動：在 __init__ 中載入 item2code 映射
        self.vocab_sizes = self.config['vocab_sizes']
        self.bases = self.config['bases']
        self.num_levels = len(self.vocab_sizes)
        self.item_to_code, _ = item2code(
            self.config['code_path'], self.vocab_sizes, self.bases
        )
        
        # 載入原始數據 (Item IDs)
        # 注意: process_jsonl 返回的是 {'history': [padded_0based_ids], 'target': 0based_id}
        self.raw_data = process_jsonl(self.dataset_path, self.max_len, PAD_TOKEN=0) # Item ID 用 0 做 padding

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, index):
        # 獲取原始數據 (已經 padding 過的 0-based ID 序列)
        item = self.raw_data[index]
        hist_ids_0based_padded = item['history']
        tgt_id_0based = item['target']

        # --- 動態準備 Code Tokens ---
        code_pad_token_list = [self.PAD_TOKEN_ID] * self.num_levels
        # ✅ 使用 padding 過的 history ID 進行查找，遇到 0 (padding ID) 時返回 padding code
        hist_codes = [self.item_to_code.get(x + 1, code_pad_token_list) if x != 0 else code_pad_token_list 
                      for x in hist_ids_0based_padded]
        tgt_code = self.item_to_code.get(tgt_id_0based + 1, code_pad_token_list)
        
        # --- 準備 1-based Item IDs (去除 padding) ---
        hist_ids_1based = [x + 1 for x in hist_ids_0based_padded if x != 0]

        # ✅ 返回所有可能需要的數據
        return {
            # 原始 0-based ID 序列 (含 padding)，用於可能的未來模型
            'history_raw_padded': hist_ids_0based_padded, 
            # 1-based 原始 ID 序列 (不含 padding)，用於 RPG collate
            'hist_ids': hist_ids_1based,    
            # Code Token 序列 (含 padding code)，用於 TIGER collate
            'history': hist_codes,          
            # Target Code Token
            'target_code': tgt_code,        
            # 0-based Target 原始 ID
            'target_id': tgt_id_0based      
        }


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
