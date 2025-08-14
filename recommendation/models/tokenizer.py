import os
import json
import torch
from tqdm import tqdm
import numpy as np

from decoder.dataset import AbstractDataset
from decoder.tokenizer import AbstractTokenizer

class Tokenizer(AbstractTokenizer):
    """
    改造后的 Tokenizer。
    它不仅加载 RQ-VAE codes，还负责创建和维护最终的 item_id -> tokens 映射表。
    """
    def __init__(self, config: dict, dataset: AbstractDataset):
        print("🔑 config keys:", list(config.keys()))
        print("🔑 RQ-VAE config:", config.get("RQ-VAE"))
        self.rqvae_config = config["RQ-VAE"]
        self.n_codebooks = self.rqvae_config["num_layers"]
        self.codebook_size = self.rqvae_config["code_book_size"]
        
        super(Tokenizer, self).__init__(config, dataset)
        self.item2id = dataset.item2id
        self.user2id = dataset.user2id
        self.id2item = dataset.id_mapping['id2item']
        
        # item_name -> tokens 字典
        self.item2tokens = self._init_tokenizer()
        # item_id -> tokens 张量 (核心改动)
        self.item_id2tokens = self._map_item_tokens_tensor(dataset).to(config['device'])
        
        self.eos_token = self.n_digit * self.codebook_size + 1
        self.ignored_label = -100
        self.collate_fn = {'train': self.collate_fn_train, 'val': self.collate_fn_eval, 'test': self.collate_fn_eval}

    @property
    def n_digit(self):
        return self.n_codebooks

    @property
    def vocab_size(self):
        return self.n_codebooks * self.codebook_size + 2
        
    @property
    def max_token_seq_len(self):
        return self.config['max_item_seq_len']

    def _init_tokenizer(self) -> dict:
        dataset_name = self.config['dataset']
        category = self.config['category']
        
        # 自动构建 codebook.json 路径，无需手动指定
        codes_path = f"cache/{dataset_name}/{category}/codebook/codebook.json"
        
        self.log(f"✅ [Tokenizer] 正在从 RQ-VAE 的最终成果加载 Item Codes: {codes_path}")
        
        if not os.path.exists(codes_path):
            raise FileNotFoundError(f"错误: 指定的 Item Code 文件 '{codes_path}' 不存在。请确保已运行 RQ-VAE 训练生成 codebook。")
        
        # 这里的调试打印也要使用实际加载的 codes_path
        print("🔍 [DEBUG] Tokenizer 实际加载的 item_codes_path =", codes_path)

        with open(codes_path, 'r') as f:
            item_id_str_map = json.load(f)
            item_id_map = {int(k): v for k, v in item_id_str_map.items()}
        
        print(f"id2item 长度: {len(self.id2item)}")
        print(f"codebook.json 的最大 item_id: {max(item_id_map.keys())}")

        print("📦 id2item 示例 (前5):")
        for i in range(min(5, len(self.id2item))):
            print(f"  {i} -> {self.id2item[i]}")

        print("📦 codebook.json 示例 (前5):")
        for i, (item_id, codes) in enumerate(item_id_map.items()):
            print(f"  {item_id} -> {codes}")
            if i >= 4:
                break


        item2tokens = {}
        for item_id, codes in item_id_map.items():
            if item_id == 0:
                continue
            item_name = self.id2item[item_id]
            # 这里使用的 self.codebook_size 和 self.n_codebooks 已经在 __init__ 中正确设置
            adjusted_tokens = [c + i * self.codebook_size + 1 for i, c in enumerate(codes)]
            item2tokens[item_name] = tuple(adjusted_tokens)

        self.log(f"[Tokenizer] 成功加载了 {len(item2tokens)} 个物品的 RQ-VAE codes。")
        return item2tokens
        
    def _map_item_tokens_tensor(self, dataset: AbstractDataset) -> torch.Tensor:
        """ (新增方法) 创建从 item_id 到全局 token ID 序列的映射张量。"""
        # 计算实际需要的张量大小
        max_item_id = 0
        for item_name in self.item2tokens.keys():
            item_id = dataset.item2id.get(item_name)
            if item_id is not None:
                max_item_id = max(max_item_id, item_id)
        
        # 确保张量大小足够容纳所有 item_id
        tensor_size = max(dataset.n_items, max_item_id + 1)
        tensor = torch.zeros((tensor_size, self.n_digit), dtype=torch.long)
        
        for item_name, tokens in self.item2tokens.items():
            item_id = dataset.item2id.get(item_name)
            if item_id is not None:
                tensor[item_id] = torch.LongTensor(tokens)
        return tensor

    # --- Tokenize 和 Collate 函数保持不变，此处省略以保持简洁 ---
    # ... (将您原来的 _tokenize_first_n_items, _tokenize_later_items, tokenize_function, tokenize, 和 collate 函数粘贴在这里)
    def _tokenize_first_n_items(self, item_seq: list) -> tuple:
        input_ids = [self.item2id[item] for item in item_seq[:-1]]
        seq_lens = len(input_ids)
        attention_mask = [1] * seq_lens
        pad_lens = self.config['max_item_seq_len'] - seq_lens
        input_ids.extend([self.padding_token] * pad_lens)
        attention_mask.extend([0] * pad_lens)
        labels = [self.item2id[item] for item in item_seq[1:]]
        labels.extend([self.ignored_label] * pad_lens)
        return input_ids, attention_mask, labels, seq_lens

    def _tokenize_later_items(self, item_seq: list, pad_labels: bool = True) -> tuple:
        input_ids = [self.item2id[item] for item in item_seq[:-1]]
        seq_lens = len(input_ids)
        attention_mask = [1] * seq_lens
        labels = [self.ignored_label] * seq_lens
        labels[-1] = self.item2id[item_seq[-1]]
        pad_lens = self.config['max_item_seq_len'] - seq_lens
        input_ids.extend([self.padding_token] * pad_lens)
        attention_mask.extend([0] * pad_lens)
        if pad_labels:
            labels.extend([self.ignored_label] * pad_lens)
        return input_ids, attention_mask, labels, seq_lens

    def tokenize_function(self, example: dict, split: str) -> dict:
        max_item_seq_len = self.config['max_item_seq_len']
        item_seq = example['item_seq'][0]
        if split == 'train':
            n_return_examples = max(len(item_seq) - max_item_seq_len, 1)
            input_ids, attention_mask, labels, seq_lens = self._tokenize_first_n_items(
                item_seq=item_seq[:min(len(item_seq), max_item_seq_len + 1)]
            )
            all_input_ids, all_attention_mask, all_labels, all_seq_lens = \
                [input_ids], [attention_mask], [labels], [seq_lens]
            for i in range(1, n_return_examples):
                cur_item_seq = item_seq[i:i+max_item_seq_len+1]
                input_ids, attention_mask, labels, seq_lens = self._tokenize_later_items(cur_item_seq)
                all_input_ids.append(input_ids)
                all_attention_mask.append(attention_mask)
                all_labels.append(labels)
                all_seq_lens.append(seq_lens)
            return {'input_ids': all_input_ids, 'attention_mask': all_attention_mask, 'labels': all_labels, 'seq_lens': all_seq_lens}
        else:
            input_ids, attention_mask, labels, seq_lens = self._tokenize_later_items(
                item_seq=item_seq[-(max_item_seq_len+1):], pad_labels=False
            )
            return {'input_ids': [input_ids], 'attention_mask': [attention_mask], 'labels': [labels[-1:]], 'seq_lens': [seq_lens]}

    def tokenize(self, datasets: dict) -> dict:
        tokenized_datasets = {}
        for split in datasets:
            tokenized_datasets[split] = datasets[split].map(
                lambda t: self.tokenize_function(t, split),
                batched=True, batch_size=1,
                remove_columns=datasets[split].column_names,
                num_proc=self.config['num_proc'],
                desc=f'Tokenizing {split} set: '
            )
            tokenized_datasets[split] = tokenized_datasets[split].flatten()
            tokenized_datasets[split].set_format(type='torch')
        return tokenized_datasets

    def collate_fn_train(self, a_list_of_examples: list) -> dict:
        batch = {}
        for key in a_list_of_examples[0].keys():
            batch[key] = torch.stack([example[key] for example in a_list_of_examples])
        return batch

    def collate_fn_eval(self, a_list_of_examples: list) -> dict:
        batch = {}
        for key in a_list_of_examples[0].keys():
            if key == 'labels':
                batch[key] = torch.tensor([example[key] for example in a_list_of_examples], dtype=torch.long)
            else:
                batch[key] = torch.stack([example[key] for example in a_list_of_examples])
        return batch