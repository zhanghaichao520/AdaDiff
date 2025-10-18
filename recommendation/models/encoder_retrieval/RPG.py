# 檔案路徑: recommendation/models/retrieval/RPG.py (建議放在新子目錄)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model
import numpy as np
from typing import Dict, List
import logging

from ..abstract_model import AbstractModel 
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from metrics import recall_at_k, ndcg_at_k

class ResBlock(nn.Module):
    # ... (ResBlock 程式碼保持不變) ...
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.zeros_(self.linear.weight)
        self.act = nn.SiLU()
    def forward(self, x):
        return x + self.act(self.linear(x))

class RPG(AbstractModel):
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # ✅ 關鍵修正 1：在 __init__ 中初始化 self.logger
        self.logger = logging.getLogger(self.__class__.__name__) # 使用類別名稱作為 logger 名稱

        model_params = self.config['model_params']
        token_params = self.config['token_params']
        
        self.item_id2tokens = self._load_codebook_as_tensor(config['code_path'])
        self.n_items = self.item_id2tokens.shape[0]

        gpt2config = GPT2Config(
            vocab_size=token_params['vocab_size'], n_positions=model_params['max_len'],
            n_embd=model_params['n_embd'], n_layer=model_params['n_layer'],
            n_head=model_params['n_head'], n_inner=model_params['n_inner'],
            activation_function=model_params['activation_function'], resid_pdrop=model_params['resid_pdrop'],
            embd_pdrop=model_params['embd_pdrop'], attn_pdrop=model_params['attn_pdrop'],
            layer_norm_epsilon=float(model_params['layer_norm_epsilon']), initializer_range=model_params['initializer_range'],
            eos_token_id=token_params['eos_token_id'],
        )
        self.gpt2 = GPT2Model(gpt2config)
        self.gpt2.resize_token_embeddings(token_params['vocab_size'])

        self.n_pred_head = self.config['code_len']
        self.pred_heads = nn.ModuleList([ResBlock(model_params['n_embd']) for _ in range(self.n_pred_head)])
        self.temperature = model_params['temperature']
        self.loss_fct = nn.CrossEntropyLoss()
        self._debug_printed_forward = False

    @property
    def task_type(self) -> str: return 'retrieval'

    def _load_codebook_as_tensor(self, code_path: str) -> torch.Tensor:
        codes_arr = np.load(code_path, allow_pickle=True)
        codes_mat = np.vstack(codes_arr) if codes_arr.dtype == object else codes_arr
        dummy_row = np.zeros((1, codes_mat.shape[1]), dtype=codes_mat.dtype)
        full_codes = np.vstack([dummy_row, codes_mat])
        return torch.from_numpy(full_codes).long()

    def forward(self, batch: Dict) -> Dict:
        input_item_ids = batch['input_ids']
        target_tokens = batch['target_codes'] # 注意：Loss 計算用 target_codes
        batch_size = input_item_ids.shape[0]
        current_device = input_item_ids.device

        # --- 1. Item ID -> Item Embedding ---
        if self.item_id2tokens.device != current_device:
            self.item_id2tokens = self.item_id2tokens.to(current_device)
        
        input_tokens = self.item_id2tokens[input_item_ids] # Shape: (B, L_item, L_code)
        token_embs = self.gpt2.wte(input_tokens)           # Shape: (B, L_item, L_code, D)
        
        # ✅ 關鍵修正：刪除下面這行 view
        # item_code_embs = token_embs.view(batch_size, -1, self.config['code_len'], self.config['model_params']['n_embd'])
        
        # ✅ 直接從 token_embs 計算 mean
        input_embs = token_embs.mean(dim=2)              # Shape: (B, L_item, D)

        # --- 2. 重建 Attention Mask (保持不變) ---
        # 注意：這裡需要 batch['attention_mask'] 來自 DataLoader 的 Item-level mask
        item_level_attention_mask = batch['attention_mask'] 

        # --- 3. GPT-2 (保持不變) ---
        outputs = self.gpt2(inputs_embs=input_embs, attention_mask=item_level_attention_mask)
        
        # --- 4. 獲取最後狀態 & Heads (保持不變) ---
        seq_lens = item_level_attention_mask.sum(dim=1); valid_lens = torch.clamp(seq_lens - 1, min=0) 
        last_hidden_states = outputs.last_hidden_state[torch.arange(batch_size, device=current_device), valid_lens]
        final_states = [head(last_hidden_states) for head in self.pred_heads]; final_states = torch.stack(final_states, dim=1); final_states = F.normalize(final_states, dim=-1)

        # --- 5. 計算 Loss (保持不變) ---
        token_emb = self.gpt2.wte.weight; token_emb = F.normalize(token_emb, dim=-1)
        bases = torch.tensor(self.config['bases'], device=current_device); vocab_sizes = torch.tensor(self.config['vocab_sizes'], device=current_device)
        # ... (後續 Loss 計算不變) ...
        # ... 使用 target_tokens 計算 labels_i ...
        losses = []
        # (Debug 打印可以暫時保留或移除)
        for i in range(self.n_pred_head):
            start_idx = bases[i] + 1; end_idx = min(start_idx + vocab_sizes[i], token_emb.shape[0]) 
            token_embs_i = token_emb[start_idx:end_idx]
            logits_i = torch.matmul(final_states[:, i, :], token_embs_i.T) / self.temperature
            labels_i = target_tokens[:, i] - bases[i] - 1
            if not (torch.all(labels_i >= 0) and torch.all(labels_i < logits_i.shape[1])):
                 self.logger.error(f"FATAL: Head {i} labels out of bounds! Skipping loss."); losses.append(torch.tensor(100.0, device=current_device, requires_grad=True)); continue
            losses.append(self.loss_fct(logits_i, labels_i))
        loss = torch.mean(torch.stack(losses)) if losses else torch.tensor(0.0, device=current_device, requires_grad=True)
        return {'loss': loss}

    # _generate_ranklist 保持不變 (它只用最後一個狀態，評估邏輯不變)
    def _generate_ranklist(self, batch: Dict, topk: int) -> torch.Tensor:
        input_item_ids = batch['input_ids']
        attention_mask = batch['attention_mask'] # Item-level mask
        batch_size = input_item_ids.shape[0]
        current_device = input_item_ids.device

        # --- 1. Item ID -> Item Embedding ---
        if self.item_id2tokens.device != current_device:
            self.item_id2tokens = self.item_id2tokens.to(current_device)
        input_tokens = self.item_id2tokens[input_item_ids] # Shape: (B, L_item, L_code)
        token_embs = self.gpt2.wte(input_tokens)           # Shape: (B, L_item, L_code, D)
        
        # ✅ 關鍵修正：刪除 view，直接 mean
        input_embs = token_embs.mean(dim=2)              # Shape: (B, L_item, D)

        # --- 2. GPT-2 ---
        outputs = self.gpt2(inputs_embs=input_embs, attention_mask=attention_mask)
        
        # --- 3. 獲取最後狀態 & Heads (保持不變) ---
        seq_lens = attention_mask.sum(dim=1); valid_lens = torch.clamp(seq_lens - 1, min=0)
        last_hidden_states = outputs.last_hidden_state[torch.arange(batch_size, device=current_device), valid_lens]
        final_states = [head(last_hidden_states) for head in self.pred_heads]; final_states = torch.stack(final_states, dim=1); final_states = F.normalize(final_states, dim=-1)

        # --- 4. 計算與所有 Item 的相似度 (保持不變) ---
        token_emb = self.gpt2.wte.weight; token_emb = F.normalize(token_emb, dim=-1)
        all_item_embs = self.gpt2.wte(self.item_id2tokens); all_item_embs = F.normalize(all_item_embs, dim=-1)
        item_scores = torch.einsum('bld,ild->bi', final_states, all_item_embs)
        
        # --- 5. 獲取 Top-K (保持不變) ---
        _, topk_indices = torch.topk(item_scores, k=topk, dim=1)
        
        return topk_indices

    # evaluate_step 保持不變 (它只調用 _generate_ranklist)
    def evaluate_step(self, batch: Dict, topk_list: List[int]) -> Dict[str, float]:
        max_k = max(topk_list)
        ranked_item_indices = self._generate_ranklist(batch, topk=max_k)
        # ✅ labels 在 _collate_retrieval_ids 中就是 target_ids_tensor (0-based)
        target_ids = batch['labels'].unsqueeze(1) 
        # ranked_item_indices 是 0-based index, 對應 1-based item id
        # 所以 ranked_item_indices - 1 才是 0-based item id? 不對，看上面 _generate_ranklist
        # topk_indices 是 item_scores (B, N+1) 的 index, index i 對應 item id i (1-based)
        # 所以 topk_indices 本身就是 1-based item id (0 是 dummy)
        # 因此比較時需要 -1
        hits = ((ranked_item_indices - 1) == target_ids).cpu() 
        batch_metrics = {}
        for k in topk_list:
            pos_index_k = hits[:, :k]
            if pos_index_k.numel() > 0: recall = recall_at_k(pos_index_k, k).mean().item(); ndcg = ndcg_at_k(pos_index_k, k).mean().item()
            else: recall = 0.0; ndcg = 0.0
            batch_metrics[f'Recall@{k}'] = recall; batch_metrics[f'NDCG@{k}'] = ndcg
        return batch_metrics