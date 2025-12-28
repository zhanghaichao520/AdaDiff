# models/TIGER.py (遵守新契约 + MMR 连续可控版)

from typing import Any, Dict, List, Optional, Tuple
import math
import torch
import logging
logger = logging.getLogger(__name__)
import transformers

import sys
from pathlib import Path
root = Path(__file__).resolve().parents[3]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from recommendation.metrics import (
    recall_at_k,
    ndcg_at_k,
    calculate_diversity_at_n,
    calculate_alpha_ndcg_at_k,
    compute_scd,
    compute_weighted_scd,
)
from recommendation.models.generation.prefix_tree import Trie
from recommendation.models.abstract_model import AbstractModel

T5ForConditionalGeneration = transformers.T5ForConditionalGeneration
T5Config = transformers.T5Config


class TIGER_MMR(AbstractModel):
    def __init__(
        self,
        config: Dict[str, Any],
        item_to_code_map: Optional[Dict[int, List[int]]] = None,
        code_to_item_map: Optional[Dict[Tuple[int, ...], int]] = None,
        item_to_cate_map: Optional[Dict[int, Any]] = None,
        prefix_trie: Optional[Trie] = None,
    ):
        super().__init__(config)

        model_params = config["model_params"]
        token_params = config["token_params"]
        t5config = T5Config(
            **model_params,
            **token_params,
            decoder_start_token_id=0,
        )

        self.t5 = T5ForConditionalGeneration(config=t5config)
        self.t5.resize_token_embeddings(config["token_params"]["vocab_size"])
        self.n_params_str = self._calculate_n_parameters()

        # Prefix trie
        self.prefix_trie_fn = None
        if prefix_trie is not None:
            self.prefix_trie_fn = prefix_trie.get_allowed_next_tokens
            logger.info("TIGER MMR 模型已成功加载前缀树 (Prefix Trie)。")
        else:
            logger.info("TIGER MMR 模型未加载前缀树 (Prefix Trie)。")

        eval_params = config.get("evaluation_params", {})
        self.use_mmr = bool(eval_params.get("use_mmr", False))
        self.alpha_diversity = float(eval_params.get("alpha_diversity", 0.5))
        if self.use_mmr:
            logger.info("TIGER MMR 模型 use_mmr 已开启")

        # maps
        self.code_to_item_map = code_to_item_map or {}

        # cate maps
        self.item_to_cate_map, self.item_to_cates_map = self._build_item_cate_maps(
            item_to_cate_map, item_to_code_map
        )
        self.num_semantic_categories = len(set(self.item_to_cate_map.values())) if self.item_to_cate_map else 0

        # category tensor (0-based item id)
        max_item_id = max(self.item_to_cate_map.keys()) if self.item_to_cate_map else 0
        cate_tensor = torch.full((max_item_id + 1,), -1, dtype=torch.long)
        for iid, cate in self.item_to_cate_map.items():
            if 0 <= iid <= max_item_id:
                cate_tensor[iid] = int(cate)
        self.register_buffer("item_category_tensor", cate_tensor, persistent=False)

    # ---------------- Basic Properties ----------------
    @property
    def task_type(self) -> str:
        return "generative"

    @property
    def n_parameters(self) -> str:
        return self.n_params_str

    def _calculate_n_parameters(self) -> str:
        num_params = lambda ps: sum(p.numel() for p in ps if p.requires_grad)
        total_params = num_params(self.parameters())
        emb_params = num_params(self.t5.get_input_embeddings().parameters())
        return (
            f"# Embedding parameters: {emb_params:,}\n"
            f"# Non-embedding parameters: {total_params - emb_params:,}\n"
            f"# Total trainable parameters: {total_params:,}\n"
        )

    # ---------------- Contract Methods ----------------
    def forward(self, batch: Dict) -> Dict:
        t5_known_args = {"input_ids", "attention_mask", "labels"}
        t5_inputs = {k: v for k, v in batch.items() if k in t5_known_args}
        return self.t5(**t5_inputs)

    def generate(self, **kwargs: Any) -> torch.Tensor:
        if self.prefix_trie_fn is not None:
            kwargs.setdefault("prefix_allowed_tokens_fn", self.prefix_trie_fn)
        return self.t5.generate(**kwargs)

    # ---------------- Evaluation ----------------
    def evaluate_step(self, batch: Dict[str, torch.Tensor], topk_list: List[int]) -> Dict[str, float]:
        eval_cfg = self.config.get("evaluation_params", {})
        beam_size = int(eval_cfg["beam_size"])
        code_len = self.config["code_len"]
        use_mmr = bool(eval_cfg.get("use_mmr", False))
        mmr_lambda = float(eval_cfg.get("mmr_lambda", 1.0))

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        device = input_ids.device
        batch_size = input_ids.size(0)

        # 1) generation (need sequence scores)
        with torch.no_grad():
            gen_out = self.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=beam_size,
                num_return_sequences=beam_size,
                max_new_tokens=code_len,
                early_stopping=False,
                output_scores=True,
                return_dict_in_generate=True,
            )

        raw_seqs = gen_out.sequences.view(batch_size, beam_size, -1)
        preds_codes = raw_seqs[:, :, 1 : 1 + code_len]               # (B, Beam, L)
        beam_scores = gen_out.sequences_scores.view(batch_size, beam_size)  # (B, Beam)

        # 2) rerank pool size = max(topk_list)
        eval_max_k = max(topk_list)

        if use_mmr:
            preds_codes, rerank_items, rerank_cates = self.apply_mmr_to_preds_codes(
                preds_codes=preds_codes,
                beam_scores=beam_scores,
                code_to_item_map=self.code_to_item_map,
                item_category_tensor=self.item_category_tensor,
                top_k=eval_max_k,
                lambda_=mmr_lambda,
            )
        else:
            preds_codes = preds_codes[:, :eval_max_k]
            rerank_items, rerank_cates = None, None

        K = preds_codes.size(1)  # should be eval_max_k

        # 3) accuracy metrics
        pos_index = self._calculate_pos_index(preds_codes, labels, maxk=K).to(device)

        batch_metrics: Dict[str, float] = {"count": float(batch_size)}
        for k in topk_list:
            if k <= K:
                batch_metrics[f"Recall@{k}"] = recall_at_k(pos_index[:, :k], k).sum().item()
                batch_metrics[f"NDCG@{k}"] = ndcg_at_k(pos_index[:, :k], k).sum().item()

        # 4) diversity metrics (0-based item ids)
        if self.code_to_item_map and self.item_category_tensor.numel() > 0:
            metric_k = min(10, K)
            cate_map = self.item_category_tensor.to(device)

            # items: prefer rerank_items (fast path)
            if rerank_items is None:
                rerank_items = torch.full((batch_size, K), -1, device=device, dtype=torch.long)
                for b in range(batch_size):
                    for i in range(K):
                        key = tuple(int(x) for x in preds_codes[b, i].tolist())
                        rerank_items[b, i] = self.code_to_item_map.get(key, -1)

            cand_items = rerank_items[:, :metric_k]
            valid_mask = (cand_items >= 0) & (cand_items < cate_map.numel())

            div_sum, div_cnt = calculate_diversity_at_n(cand_items, cate_map, valid_mask=valid_mask)
            batch_metrics[f"Diversity@{metric_k}"] = div_sum
            batch_metrics[f"_valid_Diversity@{metric_k}"] = div_cnt

            # GT items for alpha-ndcg (strict, no inflation)
            gt_items = torch.full((batch_size, 1), -1, device=device, dtype=torch.long)
            for b in range(batch_size):
                gt_code = tuple(int(x) for x in labels[b].tolist())
                gt_item = self.code_to_item_map.get(gt_code, -1)
                if gt_item >= 0:
                    gt_items[b, 0] = gt_item

            alpha_sum, alpha_cnt = calculate_alpha_ndcg_at_k(
                candidates=cand_items,
                ground_truth=gt_items,
                item_category_map=cate_map,
                k=metric_k,
                alpha=self.alpha_diversity,
                valid_mask=valid_mask,
            )
            batch_metrics[f"AlphaNDCG@{metric_k}"] = alpha_sum
            batch_metrics[f"_valid_AlphaNDCG@{metric_k}"] = alpha_cnt

            # SCD / WSCD
            if self.num_semantic_categories > 0 and self.item_to_cate_map:
                scd_sum, wscd_sum, scd_cnt = 0.0, 0.0, 0
                for b in range(batch_size):
                    rec_items = [int(cand_items[b, i]) for i in range(metric_k) if valid_mask[b, i]]
                    if not rec_items:
                        continue
                    try:
                        scd_sum += compute_scd(rec_items, self.item_to_cate_map, self.num_semantic_categories)
                        wscd_sum += compute_weighted_scd(rec_items, self.item_to_cate_map)
                        scd_cnt += 1
                    except Exception:
                        continue
                if scd_cnt > 0:
                    batch_metrics[f"SCD@{metric_k}"] = scd_sum
                    batch_metrics[f"WSCD@{metric_k}"] = wscd_sum
                    batch_metrics[f"_valid_SCD@{metric_k}"] = scd_cnt

        return batch_metrics

    # ---------------- TIGER Helper ----------------
    @staticmethod
    def _calculate_pos_index(preds: torch.Tensor, labels: torch.Tensor, maxk: int) -> torch.Tensor:
        preds = preds.detach().cpu()
        labels = labels.detach().cpu()
        B, _, L_pred = preds.shape
        L_label = labels.shape[1]

        if L_pred < L_label:
            padding = torch.zeros((B, maxk, L_label - L_pred), dtype=preds.dtype)
            preds = torch.cat([preds, padding], dim=2)
        elif L_pred > L_label:
            preds = preds[:, :, :L_label]

        pos_index = torch.zeros((B, maxk), dtype=torch.bool)
        for i in range(B):
            gt = labels[i]
            gt_semantic = gt[:-1].tolist()
            gt_dup = int(gt[-1].item())

            for j in range(maxk):
                pj = preds[i, j]
                pj_semantic = pj[:-1].tolist()
                pj_dup = int(pj[-1].item())
                if pj_semantic == gt_semantic and pj_dup == gt_dup:
                    pos_index[i, j] = True
                    break
        return pos_index

    # =========================================================
    # ✅ 连续可控 MMR：把惩罚从 0/1 改为 “按已选同类频次的连续惩罚”
    # =========================================================
    @staticmethod
    @torch.no_grad()
    def mmr_rerank_category_continuous(
        beam_item_ids: torch.Tensor,           # (B, Beam)
        beam_scores: torch.Tensor,             # (B, Beam)
        item_category_tensor: torch.Tensor,    # (N,)
        top_k: int,
        lambda_: float,
        invalid_cate_id: int = -1,
        tie_break_by_score: bool = True,
        penalty_power: float = 1.0,
    ):
        """
        Continuous category-aware MMR:
        - relevance: normalized to [0,1]
        - redundancy penalty: freq(selected_same_category)/k  (continuous in [0,1])
          optionally raise to penalty_power to control curvature.

        This fixes the "lambda becomes a switch" problem.
        """
        assert 0.0 <= float(lambda_) <= 1.0, "lambda_ must be in [0,1]"
        B, Beam = beam_item_ids.shape
        device = beam_item_ids.device
        cate_map = item_category_tensor.to(device)

        valid_item = (beam_item_ids >= 0) & (beam_item_ids < cate_map.numel())

        beam_cates = torch.full_like(beam_item_ids, invalid_cate_id)
        beam_cates[valid_item] = cate_map[beam_item_ids[valid_item]]

        # normalize relevance to [0,1]
        min_s = beam_scores.min(dim=1, keepdim=True).values
        max_s = beam_scores.max(dim=1, keepdim=True).values
        rel = (beam_scores - min_s) / (max_s - min_s + 1e-12)

        reranked_indices = torch.full((B, top_k), 0, device=device, dtype=torch.long)
        NEG_INF = -1e9

        for b in range(B):
            remaining = valid_item[b].clone()
            selected_counts: Dict[int, int] = {}  # cate -> count

            for t in range(top_k):
                if not remaining.any():
                    reranked_indices[b, t] = reranked_indices[b, t - 1] if t > 0 else 0
                    continue

                # continuous redundancy penalty in [0,1]
                cates_b = beam_cates[b]  # (Beam,)
                pen = torch.zeros((Beam,), device=device)

                # current step denominator: t+1 keeps penalty scale stable across steps
                denom = float(max(1, t + 1))

                # fill penalty using counts dict (only for valid cate)
                # vectorized-ish: loop over unique selected categories (small)
                for cate, cnt in selected_counts.items():
                    if cate < 0:
                        continue
                    mask = (cates_b == cate)
                    pen[mask] = max(pen[mask].max().item(), cnt / denom)  # overwrite with same value

                if penalty_power != 1.0:
                    pen = pen.clamp(0.0, 1.0).pow(float(penalty_power))

                mmr_score = lambda_ * rel[b] - (1.0 - lambda_) * pen
                mmr_score[~remaining] = NEG_INF

                if tie_break_by_score:
                    mmr_score = mmr_score + 1e-6 * rel[b]

                idx = int(torch.argmax(mmr_score).item())
                reranked_indices[b, t] = idx
                remaining[idx] = False

                chosen_cate = int(cates_b[idx].item())
                if chosen_cate >= 0:
                    selected_counts[chosen_cate] = selected_counts.get(chosen_cate, 0) + 1

        reranked_items = torch.gather(beam_item_ids, 1, reranked_indices)
        reranked_cates = torch.gather(beam_cates, 1, reranked_indices)
        return reranked_indices, reranked_items, reranked_cates

    @staticmethod
    @torch.no_grad()
    def apply_mmr_to_preds_codes(
        preds_codes: torch.Tensor,             # (B, Beam, L)
        beam_scores: torch.Tensor,             # (B, Beam)
        code_to_item_map: dict,
        item_category_tensor: torch.Tensor,
        top_k: int,
        lambda_: float,
    ):
        B, Beam, L = preds_codes.shape
        device = preds_codes.device

        beam_item_ids = torch.full((B, Beam), -1, device=device, dtype=torch.long)
        codes_cpu = preds_codes.detach().cpu()
        for b in range(B):
            for i in range(Beam):
                key = tuple(int(x) for x in codes_cpu[b, i].tolist())
                beam_item_ids[b, i] = int(code_to_item_map.get(key, -1))

        # ✅ 使用“连续惩罚”版本，保证 lambda 0~1 连续生效
        rerank_idx, rerank_items, rerank_cates = TIGER_MMR.mmr_rerank_category_continuous(
            beam_item_ids=beam_item_ids,
            beam_scores=beam_scores,
            item_category_tensor=item_category_tensor,
            top_k=top_k,
            lambda_=lambda_,
            tie_break_by_score=True,
            penalty_power=1.0,
        )

        idx_expanded = rerank_idx.unsqueeze(-1).expand(-1, -1, L)
        reranked_codes = torch.gather(preds_codes, 1, idx_expanded)
        return reranked_codes, rerank_items, rerank_cates

    @staticmethod
    def _build_item_cate_maps(
        item_to_cate_map: Optional[Dict[int, Any]],
        item_to_code_map: Optional[Dict[int, List[int]]],
    ) -> Tuple[Dict[int, Any], Dict[int, List[Any]]]:
        cate_map: Dict[int, Any] = {}
        if item_to_cate_map:
            cate_map.update(item_to_cate_map)
        if item_to_code_map:
            for iid, codes in item_to_code_map.items():
                if iid not in cate_map and len(codes) > 0:
                    cate_map[iid] = codes[0]
        return cate_map, {}


