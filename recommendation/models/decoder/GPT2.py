# models/tiger_gpt2.py
from typing import Any, Dict, List
import torch
import transformers
from ..abstract_model import AbstractModel

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from metrics import recall_at_k, ndcg_at_k

GPT2LMHeadModel = transformers.GPT2LMHeadModel
GPT2Config = transformers.GPT2Config


class GPT2(AbstractModel):
    """
    Decoder-only 版本的 TIGER，基于 GPT-2（无预训练权重）。
    约定：
      - batch 包含 input_ids / attention_mask / labels
      - code_len 从 config['code_len'] 读取
      - 评估使用与 T5 版一致的 beam search 和 pos_index 逻辑
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        model_params = config["model_params"]          # GPT-2 结构超参（n_layer/n_head/n_embd等）
        token_params = config["token_params"]          # 词表、特殊符号等
        vocab_size = token_params["vocab_size"]
        bos_token_id = token_params.get("bos_token_id", 1)
        eos_token_id = token_params.get("eos_token_id", 2)
        pad_token_id = token_params.get("pad_token_id", 0)

        # ⚠️ GPT-2 默认没有 pad/bos/eos，这里显式配置
        gpt2cfg = GPT2Config(
            **model_params,
            vocab_size=vocab_size,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            n_positions=token_params.get("n_positions", 1024),
            n_ctx=token_params.get("n_positions", 1024),
        )
        self.gpt2 = GPT2LMHeadModel(config=gpt2cfg)

        # 确保词表大小
        self.gpt2.resize_token_embeddings(vocab_size)

        # 保存一个字符串版参数统计
        self.n_params_str = self._calculate_n_parameters()

        # 保存 pad/eos 以便 generate 使用
        self._pad_id = pad_token_id
        self._eos_id = eos_token_id

    @property
    def task_type(self) -> str:
        return "generative"

    @property
    def n_parameters(self) -> str:
        return self.n_params_str

    def _calculate_n_parameters(self) -> str:
        num_params = lambda ps: sum(p.numel() for p in ps if p.requires_grad)
        total_params = num_params(self.parameters())
        emb_params = num_params(self.gpt2.get_input_embeddings().parameters())
        return (
            f"# Embedding parameters: {emb_params:,}\n"
            f"# Non-embedding parameters: {total_params - emb_params:,}\n"
            f"# Total trainable parameters: {total_params:,}\n"
        )

    # --- 训练/前向 ---
    def forward(self, batch: Dict) -> Dict:
        """
        通用 forward：
        - 若 labels 长度 < input_ids，自动扩展并在历史段 mask 掉 (-100)
        - 兼容 decoder-only GPT2
        """
        known = {"input_ids", "attention_mask", "labels"}
        inputs = {k: v for k, v in batch.items() if k in known}

        # 🔍 自动检测是否为 GPT-2 decoder-only 模型
        if isinstance(self.gpt2, transformers.GPT2LMHeadModel):
            input_ids = inputs["input_ids"]
            labels = inputs.get("labels")

            if labels is not None:
                # case: labels shape 不匹配 input_ids
                if labels.shape[1] < input_ids.shape[1]:
                    B, seq_len = input_ids.shape
                    new_labels = torch.full_like(input_ids, -100)

                    # 把目标 code_len 段贴在序列末尾
                    code_len = labels.shape[1]
                    new_labels[:, -code_len:] = labels
                    inputs["labels"] = new_labels

        # ✅ 正常前向
        return self.gpt2(**inputs)


    # --- 生成 ---
    def generate(self, **kwargs: Any) -> torch.Tensor:
        """
        调用 GPT-2 的标准 generate。需要注意：
          - decoder-only 不需要 encoder 输入
          - 需提供 pad_token_id/eos_token_id
        """
        kwargs.setdefault("eos_token_id", self._eos_id)
        kwargs.setdefault("pad_token_id", self._pad_id)
        return self.gpt2.generate(**kwargs)

    # --- 评估（与 T5 版保持一致的度量口径） ---
    def evaluate_step(self, batch: Dict[str, torch.Tensor], topk_list: List[int]) -> Dict[str, float]:
        beam_size = self.config["evaluation_params"]["beam_size"]
        code_len = self.config["code_len"]

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        device = input_ids.device

        # 1) 生成多样本（beam）
        preds = self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=beam_size,
            num_return_sequences=beam_size,
            max_new_tokens=code_len,
            do_sample=False,
            early_stopping=False,
            eos_token_id=self._eos_id,
            pad_token_id=self._pad_id,
        )
        # 2) 对齐形状：取新生成的 code_len 段（这里假设 input 已经包含 BOS 或历史）
        #    GPT-2 generate 的输出是 [prompt + new_tokens]，取末尾 code_len 个 token
        preds = preds[:, -code_len:].contiguous().view(input_ids.shape[0], beam_size, -1)

        # 3) 命中计算（与 T5 版一致：前 L-1 全相等，最后一位 >= 真值）
        pos_index = self._calculate_pos_index(preds, labels, maxk=beam_size).to(device)

        # 4) 指标
        out = {}
        for k in topk_list:
            out[f"Recall@{k}"] = recall_at_k(pos_index, k).mean().item()
            out[f"NDCG@{k}"]   = ndcg_at_k(pos_index, k).mean().item()
        return out

    @staticmethod
    def _calculate_pos_index(preds: torch.Tensor, labels: torch.Tensor, maxk: int) -> torch.Tensor:
        """
        preds: (B, maxk, L)
        labels: (B, L)
        命中：前 L-1 完全一致 && 最后一位 (dup) 预测 >= 真实
        """
        preds = preds.detach().cpu()
        labels = labels.detach().cpu()
        B, _, L = preds.shape
        assert L == labels.shape[1], f"Code length mismatch: preds {L} vs labels {labels.shape[1]}"

        pos_index = torch.zeros((B, maxk), dtype=torch.bool)
        for i in range(B):
            gt = labels[i]
            gt_sem, gt_dup = gt[:-1].tolist(), int(gt[-1].item())
            for j in range(maxk):
                pj = preds[i, j]
                pj_sem, pj_dup = pj[:-1].tolist(), int(pj[-1].item())
                if pj_sem == gt_sem and pj_dup >= gt_dup:
                    pos_index[i, j] = True
                    break
        return pos_index
