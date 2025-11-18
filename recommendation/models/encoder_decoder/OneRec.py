# models/OneRec.py - OneRec-style session-wise generative recommender (MoE-T5)

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Config, T5ForConditionalGeneration

import sys
from pathlib import Path

root = Path(__file__).resolve().parents[3]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from recommendation.metrics import recall_at_k, ndcg_at_k
from recommendation.models.generation.prefix_tree import Trie
from recommendation.models.abstract_model import AbstractModel

logger = logging.getLogger(__name__)


# ------------------------------
#  Simple MoE FFN for Decoder
# ------------------------------
class SimpleFFN(nn.Module):
    """A simple T5-like FFN block: Linear -> GELU -> Dropout -> Linear"""

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.wi = nn.Linear(d_model, d_ff)
        self.wo = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (B, S, D)
        x = self.wi(hidden_states)
        x = self.act(x)
        x = self.dropout(x)
        x = self.wo(x)
        return x


class MoEFFN(nn.Module):
    """
    Soft Mixture-of-Experts FFN (dense MoE，架构版，不搞极致加速)
    - Inputs:  (B, S, D)
    - Outputs: (B, S, D)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [SimpleFFN(d_model, d_ff, dropout) for _ in range(num_experts)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (B, S, D)
        B, S, D = hidden_states.shape

        # (B, S, E)
        gate_logits = self.gate(hidden_states)
        gate_probs = F.softmax(gate_logits, dim=-1)

        # 每个 expert 看到同一批 token（dense MoE）
        expert_outputs = []
        for expert in self.experts:
            y = expert(hidden_states)  # (B, S, D)
            expert_outputs.append(y)

        # (B, S, E, D)
        expert_stack = torch.stack(expert_outputs, dim=2)
        gate_probs = gate_probs.unsqueeze(-1)  # (B, S, E, 1)

        # (B, S, D)
        output = torch.sum(expert_stack * gate_probs, dim=2)
        return output


# ------------------------------
#        OneRec main model
# ------------------------------
class OneRec(AbstractModel):
    """
    OneRec-style session-wise generative recommender (architecture version)

    核心特性：
    - T5 encoder-decoder backbone
    - Decoder FFN 替换为 MoE（类似 OneRec 中的 MoE 解码器）
    - Prefix Trie 约束生成（只允许合法 SID 序列）
    - Beam search 生成 session-wise 目标 code sequence
    """

    def __init__(
        self,
        config: Dict[str, Any],
        prefix_trie: Optional[Trie] = None,
    ):
        super().__init__(config)

        self.cfg = config               # 实验 / 训练配置（字典）
        model_params = config["model_params"]
        token_params = config["token_params"]

        # -------- 1. 构建 T5 配置 --------
        t5_config = T5Config(
            vocab_size=token_params["vocab_size"],
            d_model=model_params["d_model"],
            d_ff=model_params["d_ff"],
            num_heads=model_params["num_heads"],
            num_layers=model_params["num_encoder_layers"],
            num_decoder_layers=model_params["num_decoder_layers"],
            dropout_rate=model_params["dropout"],
            decoder_start_token_id=0,
            use_cache=True,
            is_encoder_decoder=True,
        )

        # -------- 2. 初始化 T5 模型 --------
        self.t5 = T5ForConditionalGeneration(config=t5_config)
        self.t5.resize_token_embeddings(token_params["vocab_size"])

        # -------- 3. 用 MoE 替换 Decoder FFN --------
        d_model = model_params["d_model"]
        d_ff = model_params["d_ff"]
        moe_num_experts = model_params.get("moe_num_experts", 4)
        moe_dropout = model_params.get("moe_dropout", model_params["dropout"])

        moe_layer_count = 0
        for block in self.t5.decoder.block:
            # 对于 decoder：layer 结构一般为 [self-attn, cross-attn, ff]
            if len(block.layer) == 3 and hasattr(block.layer[2], "DenseReluDense"):
                block.layer[2].DenseReluDense = MoEFFN(
                    d_model=d_model,
                    d_ff=d_ff,
                    num_experts=moe_num_experts,
                    dropout=moe_dropout,
                )
                moe_layer_count += 1

        logger.info(f"[OneRec] Replaced decoder FFN with MoE in {moe_layer_count} layers.")

        # -------- 4. Prefix Trie 支持 --------
        self.prefix_trie_fn = None
        if prefix_trie is not None:
            # 这里约定 Trie.get_allowed_next_tokens(batch_id, input_ids) 的签名
            self.prefix_trie_fn = prefix_trie.get_allowed_next_tokens
            logger.info("[OneRec] Loaded Prefix Trie for constrained generation.")
        else:
            logger.info("[OneRec] No Prefix Trie provided; generation is unconstrained.")

        # 预计算参数量信息
        self.n_params_str = self._calculate_n_parameters()
        logger.info("OneRec model initialized successfully.")

    # ---------------- basic properties ----------------
    @property
    def task_type(self) -> str:
        return "generative"

    @property
    def n_parameters(self) -> str:
        return self.n_params_str

    def _calculate_n_parameters(self) -> str:
        def num_params(ps):
            return sum(p.numel() for p in ps if p.requires_grad)

        total_params = num_params(self.parameters())
        t5_params = num_params(self.t5.parameters())
        other_params = total_params - t5_params

        return (
            f"# T5 backbone parameters: {t5_params:,}\n"
            f"# Other modules parameters: {other_params:,}\n"
            f"# Total trainable parameters: {total_params:,}\n"
        )

    # ---------------- training forward ----------------
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        标准 seq2seq 训练前向：
        batch 需要包含:
          - input_ids: (B, L_in)
          - attention_mask: (B, L_in)
          - labels: (B, L_out)
        """
        t5_inputs = {
            key: value
            for key, value in batch.items()
            if key in {"input_ids", "attention_mask", "labels"}
        }

        outputs = self.t5(**t5_inputs)  # 返回 Seq2SeqLMOutput(loss, logits, ...)
        return outputs

    # ---------------- generation ----------------
    def generate(self, **kwargs: Any) -> torch.Tensor:
        """
        生成推荐 SID 序列：
        - 支持 prefix_allowed_tokens_fn (Prefix Trie)
        - 支持 beam search / sampling 等策略
        """
        eval_params = self.cfg.get("evaluation_params", {})

        # 注入前缀约束
        if self.prefix_trie_fn is not None and "prefix_allowed_tokens_fn" not in kwargs:
            kwargs["prefix_allowed_tokens_fn"] = self.prefix_trie_fn

        # 默认生成参数（可以被 kwargs 覆盖）
        kwargs.setdefault("do_sample", eval_params.get("do_sample", False))
        kwargs.setdefault("temperature", eval_params.get("temperature", 1.0))
        kwargs.setdefault("top_k", eval_params.get("top_k", 50))
        kwargs.setdefault("top_p", eval_params.get("top_p", 0.9))
        kwargs.setdefault("length_penalty", eval_params.get("length_penalty", 1.0))
        kwargs.setdefault("early_stopping", eval_params.get("early_stopping", True))

        return self.t5.generate(**kwargs)

    # ---------------- evaluation ----------------
    def evaluate_step(self, batch: Dict[str, torch.Tensor], topk_list: List[int]) -> Dict[str, float]:
        """
        单个 batch 的评估逻辑：
        - 使用 beam search 生成多个候选 SID
        - 与 ground-truth code 进行匹配，计算 Recall@k / NDCG@k
        """
        eval_params = self.cfg["evaluation_params"]
        beam_size = eval_params["beam_size"]
        code_len = self.cfg["code_len"]

        input_ids = batch["input_ids"]      # (B, L_in)
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]           # (B, L_label)
        device = input_ids.device
        batch_size = input_ids.size(0)

        with torch.no_grad():
            preds = self.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=beam_size,
                num_return_sequences=beam_size,
                max_new_tokens=code_len,
                length_penalty=eval_params.get("length_penalty", 1.0),
                early_stopping=eval_params.get("early_stopping", True),
            )

        # 假设 decoder_start_token_id = 0，去掉第一个 start token，只取接下来的 code_len 位
        preds = preds[:, 1 : 1 + code_len]  # (B * beam_size, code_len)
        preds = preds.view(batch_size, beam_size, -1)  # (B, beam_size, L_pred)

        # 计算每个样本的“正确位置索引矩阵” (B, beam_size)
        pos_index = self._calculate_pos_index(preds, labels, maxk=beam_size).to(device)

        batch_metrics: Dict[str, float] = {"count": float(batch_size)}
        for k in topk_list:
            recall_sum = recall_at_k(pos_index, k).sum().item()
            ndcg_sum = ndcg_at_k(pos_index, k).sum().item()
            batch_metrics[f"Recall@{k}"] = recall_sum
            batch_metrics[f"NDCG@{k}"] = ndcg_sum

        return batch_metrics

    def _calculate_pos_index(
        self,
        preds: torch.Tensor,   # (B, beam_size, L_pred)
        labels: torch.Tensor,  # (B, L_label)
        maxk: int,
    ) -> torch.Tensor:
        """
        简单版位置索引：
        - 完整 SID 序列完全相等视为正确
        - pos_index[i, j] = True 表示第 i 个样本预测的第 j 个 beam 命中
        """
        preds = preds.detach().cpu()
        labels = labels.detach().cpu()
        B, _, L_pred = preds.shape
        L_label = labels.size(1)

        # 长度对齐：截断 / padding（用 0 补）
        if L_pred < L_label:
            pad = torch.zeros((B, maxk, L_label - L_pred), dtype=preds.dtype)
            preds = torch.cat([preds, pad], dim=2)
        elif L_pred > L_label:
            preds = preds[:, :, :L_label]

        pos_index = torch.zeros((B, maxk), dtype=torch.bool)

        for i in range(B):
            gt = labels[i].tolist()
            for j in range(maxk):
                pj = preds[i, j].tolist()
                if pj == gt:
                    pos_index[i, j] = True
                    break

        return pos_index
