import os
import json
import torch
from tqdm import tqdm
import numpy as np

# 假设这两个导入路径是正确的
from dataset import AbstractDataset
from tokenizer import AbstractTokenizer

class Tokenizer(AbstractTokenizer):
    """
    最终版 Tokenizer：
    以 codebook.json 为唯一事实标准，负责所有与 item -> token 相关的映射。
    """
    def __init__(self, config: dict):
        # super 初始化不再需要 dataset 参数，因为item相关信息都将自给自足
        super(Tokenizer, self).__init__(config, dataset=None) 
        
        # 1. 从配置中读取RQ-VAE参数和设备信息
        self.rqvae_config = config["RQ-VAE"]
        self.n_codebooks = self.rqvae_config["num_layers"]
        self.codebook_size = self.rqvae_config["code_book_size"]
        self.device = config['device']
        
        # 2. --- 核心修复 1：将特殊 token 的定义和张量化提前 ---
        # 先定义python整数类型的特殊token
        self.eos_token = self.n_codebooks * self.codebook_size + 1
        self.pad_token = 0
        self.ignored_label = -100
        
        # 然后立即创建位于正确设备上的张量版本，以备后用，解决设备不匹配问题
        self.eos_token_tensor = torch.tensor([self.eos_token], device=self.device, dtype=torch.long)
        
        # 3. 初始化item到tokens的映射，并确定物品总数
        self.item_name2tokens = self._init_tokenizer()
        self.n_items = max(int(k) for k in self.item_name2tokens.keys()) + 1
        self.log(f"✅ [Tokenizer] 根据 codebook.json 确定物品总数: {self.n_items}")
        
        # 4. 构建高效的查找张量，并移动到目标设备
        #    调用此函数时，self.pad_token 已经存在，解决了 AttributeError
        self.item_id2tokens_tensor = self._build_item_tokens_tensor().to(self.device)
        
        # 5. 分配 Collate 函数
        self.collate_fn = {'train': self.collate_fn_train, 'val': self.collate_fn_eval, 'test': self.collate_fn_eval}

    @property
    def n_digit(self):
        return self.n_codebooks

    @property
    def vocab_size(self):
        return self.n_codebooks * self.codebook_size + 2
        
    @property
    def max_token_seq_len(self):
        return self.config['max_item_seq_len'] * self.n_digit

    def _init_tokenizer(self) -> dict:
        category = self.config['category']
        data_dir = self.config.get('data_dir', '../datasets')
        codes_path = os.path.join(data_dir, category, "codebook.json")

        self.log(f"✅ [Tokenizer] 正在加载 Item Codes: {codes_path}")
        if not os.path.exists(codes_path):
            raise FileNotFoundError(f"错误: Item Code 文件 '{codes_path}' 不存在。")

        with open(codes_path, 'r') as f:
            item_id_str_map = json.load(f)
        
        item_name2tokens = {}
        for item_name, codes in item_id_str_map.items():
            adjusted_tokens = [c + i * self.codebook_size + 1 for i, c in enumerate(codes)]
            item_name2tokens[item_name] = tuple(adjusted_tokens)

        self.log(f"[Tokenizer] 成功加载了 {len(item_name2tokens)} 个物品的 RQ-VAE codes。")
        return item_name2tokens
        
    def _build_item_tokens_tensor(self) -> torch.Tensor:
        """根据 self.n_items 和 self.item_name2tokens 创建查找张量。"""
        # 使用 self.pad_token (现在已经定义好了)
        tensor = torch.full((self.n_items, self.n_digit), self.pad_token, dtype=torch.long)

        for item_name, tokens in self.item_name2tokens.items():
            item_id = int(item_name)
            if item_id < self.n_items:
                tensor[item_id] = torch.as_tensor(tokens, dtype=torch.long)
        
        self.log(f"构建了 item_id -> tokens 的查找张量，维度: {tensor.shape}")
        return tensor

    # --- 新增一个小工具：把 item 序列转成“展平的全局 token 序列” ---
    def _items_to_flat_tokens(self, ids: list) -> torch.Tensor:
        ids_tensor = torch.as_tensor(ids, dtype=torch.long, device=self.device)     # [L]
        # 越界保护（可留可去）
        if ids_tensor.numel() > 0 and int(ids_tensor.max()) >= self.item_id2tokens_tensor.size(0):
            raise ValueError(f"[Tokenizer] item_id 超出范围: {int(ids_tensor.max())} >= {self.item_id2tokens_tensor.size(0)}")
        tok = self.item_id2tokens_tensor.index_select(0, ids_tensor)               # [L, D]
        return tok.reshape(-1)                                                      # [L*D]

    # --- 修改：支持 split 的 tokenize_function ---
    def tokenize_function(self, example: dict, split: str) -> dict:
        """
        train:
        - input_ids / attention_mask: 展平 token 序列
        - labels: next-token（展平）+ 末尾 EOS
        valid/test:
        - input_ids / attention_mask: 仅历史 items 的展平 token 序列
        - labels: 下一物品 item_id（标量）
        """
        device = self.device
        D = self.n_digit
        max_item_len = int(self.config['max_item_seq_len'])

        # 取出最内层序列并转 int
        raw_seq = example['item_seq']
        if len(raw_seq) > 0 and isinstance(raw_seq[0], (list, tuple)):
            raw_seq = raw_seq[0]
        ids = [int(x) for x in raw_seq]

        if split == 'train':
            # 训练：保留最近 max_item_len 个 item，做 token 级 next-token 任务
            if len(ids) > max_item_len:
                ids = ids[-max_item_len:]
            if len(ids) == 0:
                # 空样本兜底
                empty = torch.empty(0, dtype=torch.long, device=device)
                return {'input_ids': empty, 'attention_mask': empty, 'labels': empty}

            input_ids = self._items_to_flat_tokens(ids)                   # [L*D]
            attention_mask = torch.ones_like(input_ids)

            # next-token + EOS
            labels = torch.empty_like(input_ids)
            labels[:-1] = input_ids[1:]
            labels[-1] = self.eos_token

            # batched=True, batch_size=1 的兼容（如果你的 map 没用 batched，可以省略）
            if 'item_seq' in example and isinstance(example['item_seq'], list) and \
            len(example['item_seq']) == 1 and isinstance(example['item_seq'][0], (list, tuple)):
                return {'input_ids': [input_ids], 'attention_mask': [attention_mask], 'labels': [labels]}
            else:
                return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

        else:
            # 验证/测试：上下文=最近 max_item_len 个历史 + 1 个目标
            if len(ids) == 0:
                # 没有任何历史，直接跳过/返回空（按需改）
                empty = torch.empty(0, dtype=torch.long, device=device)
                return {'input_ids': empty, 'attention_mask': empty, 'labels': torch.tensor(-1, dtype=torch.long)}
            ctx = ids[-(max_item_len + 1):]
            if len(ctx) == 1:
                # 只有 1 个 item → 没法拆出历史和目标；这里直接当作无效样本
                empty = torch.empty(0, dtype=torch.long, device=device)
                return {'input_ids': empty, 'attention_mask': empty, 'labels': torch.tensor(-1, dtype=torch.long)}

            hist, tgt = ctx[:-1], int(ctx[-1])

            # 输入给模型：仅 “历史 items 的展平 tokens”
            input_ids = self._items_to_flat_tokens(hist)                  # [T_hist*D]
            attention_mask = torch.ones_like(input_ids)

            # 评估标签：目标物品 ID（标量）
            # 为了兼容 datasets 的张量化，这里返回 Python int，collate 时再转 tensor
            return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': tgt}

    # --- 修改：把 split 传入 map，禁用缓存更稳 ---
    def tokenize(self, datasets: dict) -> dict:
        tokenized_datasets = {}
        for split in datasets:
            tokenized_datasets[split] = datasets[split].map(
                lambda ex, s=split: self.tokenize_function(ex, s),
                remove_columns=datasets[split].column_names,
                num_proc=self.config.get('num_proc', 1),
                load_from_cache_file=False,                 # 建议关掉 cache，避免旧字段干扰
                desc=f'Tokenizing {split} set'
            )
            tokenized_datasets[split].set_format(type='torch')
        return tokenized_datasets

    # --- 训练集 collate：保留你原来的 padding 逻辑 ---
    def collate_fn_train(self, examples: list) -> dict:
        batch = {}
        max_len = max(len(ex['input_ids']) for ex in examples)
        for key in ['input_ids', 'attention_mask', 'labels']:
            padded_sequences = []
            for ex in examples:
                seq = ex[key]
                pad_len = max_len - len(seq)
                if key == 'input_ids':
                    pad_value = self.pad_token
                elif key == 'attention_mask':
                    pad_value = 0
                else:
                    pad_value = self.ignored_label
                padded_seq = torch.cat(
                    [seq, torch.full((pad_len,), pad_value, dtype=seq.dtype, device=seq.device)],
                    dim=0
                )
                padded_sequences.append(padded_seq)
            batch[key] = torch.stack(padded_sequences)
        return batch

    # --- 新增：验证/测试 collate（labels → [B] 的 LongTensor） ---
    def collate_fn_eval(self, examples: list) -> dict:
        batch = {}
        if len(examples) == 0:
            return {'input_ids': torch.empty(0, dtype=torch.long),
                    'attention_mask': torch.empty(0, dtype=torch.long),
                    'labels': torch.empty(0, dtype=torch.long)}
        max_len = max(len(ex['input_ids']) for ex in examples)

        # pad input_ids / attention_mask
        for key in ['input_ids', 'attention_mask']:
            padded_sequences = []
            for ex in examples:
                seq = ex[key]
                pad_len = max_len - len(seq)
                pad_value = self.pad_token if key == 'input_ids' else 0
                padded_seq = torch.cat(
                    [seq, torch.full((pad_len,), pad_value, dtype=seq.dtype, device=seq.device)],
                    dim=0
                )
                padded_sequences.append(padded_seq)
            batch[key] = torch.stack(padded_sequences)

        # labels: 目标物品 ID（标量）→ [B]
        labels_list = []
        for ex in examples:
            lab = ex['labels']
            if isinstance(lab, torch.Tensor):
                lab = int(lab.item())
            labels_list.append(lab)
        batch['labels'] = torch.tensor(labels_list, dtype=torch.long)

        return batch