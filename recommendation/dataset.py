# /recommendation/dataset.py (最终调试版)

from logging import getLogger
from datasets import Dataset
import os
import json # <-- 新增导入

class AbstractDataset:
    # ... (这部分代码保持不变) ...
    def __init__(self, config: dict):
        self.config = config
        self.accelerator = self.config['accelerator']
        self.logger = getLogger()
        self.all_item_seqs = {}
        self.id_mapping = {'user2id': {}, 'item2id': {}, 'id2user': [], 'id2item': []}
        self.item2meta = None
        self.split_data = None
    def __str__(self) -> str:
        return (f"[Dataset] {self.__class__.__name__}\n"
                f"\tNumber of users: {self.n_users}\n"
                f"\tNumber of items: {self.n_items}\n"
                f"\tNumber of interactions: {self.n_interactions}\n"
                f"\tAverage item sequence length: {self.avg_item_seq_len:.2f}")
    @property
    def n_users(self) -> int: return len(self.id_mapping['id2user'])
    @property
    def n_items(self) -> int: return len(self.id_mapping['id2item'])
    @property
    def n_interactions(self) -> int:
        total = 0
        for u in self.all_item_seqs: total += len(self.all_item_seqs[u])
        return total
    @property
    def avg_item_seq_len(self) -> float: return self.n_interactions / max(1, len(self.all_item_seqs))
    def split(self):
        if self.split_data is None: raise RuntimeError("split() not prepared.")
        return self.split_data
    def log(self, message, level='info'):
        from utils import log
        return log(message, self.config['accelerator'], self.logger, level=level)


class InterDataset(AbstractDataset):
    REQ_FILES = ('train', 'valid', 'test')

    def __init__(self, config: dict):
        super().__init__(config)
        self.logger = getLogger()
        self.category = config.get('category')
        self.data_dir = config.get('data_dir', '../datasets')
        if not self.category:
            raise ValueError("[InterDataset] config['category'] is required.")

        split2rows = {}
        raw_user_set, raw_item_set = set(), set()

        for split in self.REQ_FILES:
            path = os.path.join(self.data_dir, self.category, f"{self.category}.{split}.inter")
            # 兼容 valid / val
            if not os.path.exists(path) and split == 'valid':
                alt = os.path.join(self.data_dir, self.category, f"{self.category}.val.inter")
                if os.path.exists(alt): path = alt
            if not os.path.exists(path):
                raise FileNotFoundError(f"[InterDataset] Not found: {path}")

            rows = []
            with open(path, 'r', encoding='utf-8') as f:
                header_raw = f.readline()
                if not header_raw:
                    raise ValueError(f"[InterDataset] Empty file: {path}")
                header = header_raw.strip().lstrip('\ufeff')  # 处理 BOM

                # —— 检测列分隔符（优先 \t，否则用“两个及以上空格”）——
                if '\t' in header:
                    delim = '\t'
                    cols = header.split('\t')
                else:
                    delim = None  # 用 regex: 两个及以上空格
                    cols = re.split(r'\s{2,}', header.strip())

                # 基于列名建立索引（顺序可以变）
                name2idx = {c.strip(): i for i, c in enumerate(cols)}
                required = ['user_id:token', 'item_id_list:token_seq', 'item_id:token']
                for need in required:
                    if need not in name2idx:
                        raise ValueError(f"[InterDataset] Missing column '{need}' in header of {path}: {header}")

                # —— 逐行解析 —— #
                for ln, raw in enumerate(f, start=2):
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        parts = line.split('\t') if delim == '\t' else re.split(r'\s{2,}', line)
                        # 有些文件可能存在对齐异常，兜底：用“首列 + 末列 + 中间合并”为三列
                        if len(parts) < 3:
                            toks = line.split()
                            if len(toks) < 3:
                                raise ValueError(f"无法切成3列: '{line[:200]}'")
                            c0 = toks[0]
                            c2 = toks[-1]
                            c1 = ' '.join(toks[1:-1])
                        else:
                            # 按列名取对应字段
                            c0 = parts[name2idx['user_id:token']].strip()
                            c1 = parts[name2idx['item_id_list:token_seq']].strip()
                            c2 = parts[name2idx['item_id:token']].strip()

                        u_raw = int(c0)
                        seq_raw = [int(x) for x in c1.split()] if c1 else []
                        tgt_raw = int(c2)
                    except Exception as e:
                        # 明确指出是“列分隔符/脏行”问题，不中断整个数据集
                        self.log(f"警告: 解析文件 {path} 第 {ln} 行失败: {e} | 内容: '{line}'", level='warning')
                        continue

                    if len(seq_raw) == 0:
                        # 纯冷启动样本，通常对自回归训练无用，默认跳过（需要可改成保留）
                        continue

                    rows.append((u_raw, seq_raw, tgt_raw))
                    raw_user_set.add(u_raw)
                    raw_item_set.update(seq_raw); raw_item_set.add(tgt_raw)

            split2rows[split] = rows

        # —— 构建映射（内部ID == 原始ID；键用字符串，便于与你现有 Tokenizer 对齐） —— #
        self.id_mapping['id2user'] = [str(u) for u in sorted(raw_user_set)]
        self.id_mapping['user2id'] = {str(u): u for u in sorted(raw_user_set)}
        self.id_mapping['id2item'] = [str(i) for i in sorted(raw_item_set)]
        self.id_mapping['item2id'] = {str(i): i for i in sorted(raw_item_set)}
        self.id_mapping['raw_item2id'] = {i: i for i in sorted(raw_item_set)}
        self.id_mapping['id2raw_item'] = {i: i for i in sorted(raw_item_set)}

        # —— 组装 HF Datasets（每行一个样本：历史+目标） —— #
        dict_splits = {s: {'user': [], 'item_seq': []} for s in ('train', 'valid', 'test')}
        for split, rows in split2rows.items():
            for (u_raw, seq_raw, tgt_raw) in rows:
                dict_splits[split]['user'].append(str(u_raw))
                dict_splits[split]['item_seq'].append([str(x) for x in (seq_raw + [tgt_raw])])

        for split in dict_splits:
            dict_splits[split] = Dataset.from_dict(dict_splits[split])
        self.split_data = dict_splits

        # —— 打印基本信息 —— #
        max_uid = max(raw_user_set) if raw_user_set else -1
        max_iid = max(raw_item_set) if raw_item_set else -1
        self.log(f"[InterDataset] 成功解析 .inter | users={len(raw_user_set)}, items={len(raw_item_set)}, "
                 f"max_user_id={max_uid}, max_item_id={max_iid}")