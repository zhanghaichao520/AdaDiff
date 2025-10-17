# 檔案路徑: recommendation/models/retrieval/RPG.py (建議放在新子目錄)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model
import numpy as np
from typing import Dict, List

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
        
        # 從 config 載入參數
        model_params = self.config['model_params']
        token_params = self.config['token_params']
        
        # 1. 載入 codebook 作為 item id 到 token 的映射
        self.item_id2tokens = self._load_codebook_as_tensor(config['code_path'])
        self.n_items = self.item_id2tokens.shape[0]

        # 2. 初始化 GPT-2 模型
        gpt2config = GPT2Config(
            vocab_size=token_params['vocab_size'],
            n_positions=model_params['max_len'],
            n_embd=model_params['n_embd'],
            n_layer=model_params['n_layer'],
            n_head=model_params['n_head'],
            n_inner=model_params['n_inner'],
            activation_function=model_params['activation_function'],
            resid_pdrop=model_params['resid_pdrop'],
            embd_pdrop=model_params['embd_pdrop'],
            attn_pdrop=model_params['attn_pdrop'],
            layer_norm_epsilon=float(model_params['layer_norm_epsilon']),
            initializer_range=model_params['initializer_range'],
            eos_token_id=token_params['eos_token_id'],
        )
        self.gpt2 = GPT2Model(gpt2config)

        # 3. 初始化預測頭
        self.n_pred_head = self.config['code_len'] # 使用 config 中的 code_len
        self.pred_heads = nn.ModuleList([ResBlock(model_params['n_embd']) for _ in range(self.n_pred_head)])

        # 4. 初始化其他參數
        self.temperature = model_params['temperature']
        self.loss_fct = nn.CrossEntropyLoss()

    @property
    def task_type(self) -> str:
        return 'retrieval'

    def _load_codebook_as_tensor(self, code_path: str) -> torch.Tensor:
        codes_arr = np.load(code_path, allow_pickle=True)
        codes_mat = np.vstack(codes_arr) if codes_arr.dtype == object else codes_arr
        # 假設 codebook 的 index 0 對應 item_id 1
        # 我們需要在前面加上一個 dummy row for padding (item_id 0)
        dummy_row = np.zeros((1, codes_mat.shape[1]), dtype=codes_mat.dtype)
        full_codes = np.vstack([dummy_row, codes_mat])
        return torch.from_numpy(full_codes).long()

    def forward(self, batch: Dict) -> Dict:
        # --- (前面 1, 2, 3, 4 步驟的數據準備和 GPT-2 forward 不變) ---
        input_tokens = batch['input_ids']
        target_tokens = batch['labels']
        batch_size = input_tokens.shape[0]
        current_device = input_tokens.device
        token_embs = self.gpt2.wte(input_tokens)
        item_code_embs = token_embs.view(batch_size, -1, self.config['code_len'], self.config['model_params']['n_embd'])
        input_embs = item_code_embs.mean(dim=2)
        token_level_attention_mask = batch['attention_mask']
        reshaped_mask = token_level_attention_mask.view(batch_size, input_embs.shape[1], self.config['code_len'])
        item_level_attention_mask = reshaped_mask.any(dim=2).long()
        outputs = self.gpt2(inputs_embeds=input_embs, attention_mask=item_level_attention_mask)
        seq_lens = item_level_attention_mask.sum(dim=1)
        last_hidden_states = outputs.last_hidden_state[torch.arange(batch_size, device=current_device), seq_lens - 1]
        final_states = [head(last_hidden_states) for head in self.pred_heads]
        final_states = torch.stack(final_states, dim=1)
        final_states = F.normalize(final_states, dim=-1)

        # --- 5. 計算 Loss (採用 Chunk 方式) ---
        token_emb = self.gpt2.wte.weight
        token_emb = F.normalize(token_emb, dim=-1)
        
        # ✅ 關鍵修正：模仿原始 RPG 的 chunk 邏輯
        bases = self.config['bases']
        vocab_sizes = self.config['vocab_sizes']
        
        # 檢查是否所有 vocab_sizes 都相同 (RPG/OPQ/PQ 應該是這種情況)
        all_same_size = all(vs == vocab_sizes[0] for vs in vocab_sizes)
        
        if all_same_size and len(vocab_sizes) == self.n_pred_head:
            # 如果大小都相同，我們可以安全地使用 chunk
            codebook_size_k = vocab_sizes[0]
            # 假設有效的 token 從 index 1 開始，到 bases[-1] + vocab_sizes[-1] 結束
            start_idx = 1
            # 注意：這裡假設 token ID 是連續的
            end_idx = start_idx + self.n_pred_head * codebook_size_k 
            usable_token_embs = token_emb[start_idx:end_idx]
            
            # 檢查切片後的維度是否能被 chunk 整除
            if usable_token_embs.shape[0] == self.n_pred_head * codebook_size_k:
                token_embs_chunks = torch.chunk(usable_token_embs, self.n_pred_head, dim=0)
            else:
                # 如果因為某些原因（例如 vocab_size 計算包含了 EOS 等）導致不能整除，
                # 我們回退到原來的動態切片方法，並打印警告
                print("Warning: Cannot use torch.chunk due to shape mismatch. Falling back to dynamic slicing.")
                all_same_size = False # 標記為 False，強制使用下面的 else 分支
        
        losses = []
        for i in range(self.n_pred_head):
            if all_same_size:
                # 使用 chunk 獲取的 embedding
                token_embs_i = token_embs_chunks[i]
            else:
                # 使用原來的動態切片方法
                start_idx = bases[i] + 1
                end_idx = start_idx + vocab_sizes[i]
                token_embs_i = token_emb[start_idx:end_idx]

            # 計算 logits (這部分不變)
            logits_i = torch.matmul(final_states[:, i, :], token_embs_i.T) / self.temperature
            
            # 獲取第 i 層的 label (這部分也不變，它本身是正確的)
            # labels_i 的值域是 [0, vocab_sizes[i]-1]
            labels_i = target_tokens[:, i] - bases[i] - 1
            
            # 增加一個斷言，確保 labels_i 的值在有效範圍內
            assert torch.all(labels_i >= 0) and torch.all(labels_i < vocab_sizes[i]), \
                   f"Label index out of range for head {i}! Min: {labels_i.min()}, Max: {labels_i.max()}, Vocab Size: {vocab_sizes[i]}"

            losses.append(self.loss_fct(logits_i, labels_i))
            
        loss = torch.mean(torch.stack(losses))
        
        return {'loss': loss}


    def _generate_ranklist(self, batch: Dict, topk: int) -> torch.Tensor:
        """ 
        【完整修正版】
        內部輔助函數，執行排名生成。
        此版本整合了與 forward 方法完全一致的數據處理邏輯。
        """
        input_tokens = batch['input_ids']
        batch_size = input_tokens.shape[0]

        # ✅ 修正 1：動態地從輸入張量中獲取當前的設備
        current_device = input_tokens.device

        # ✅ 修正 2：與 forward 方法完全一致的輸入重組邏輯
        # (B, L_flat) -> (B, L_flat, D)
        token_embs = self.gpt2.wte(input_tokens)
        # (B, L_flat, D) -> (B, num_items, code_len, D)
        item_code_embs = token_embs.view(
            batch_size, -1, self.config['code_len'], self.config['model_params']['n_embd']
        )
        # (B, num_items, code_len, D) -> (B, num_items, D)
        input_embs = item_code_embs.mean(dim=2)
        
        # ✅ 同樣，重建 item-level 的 attention_mask
        token_level_attention_mask = batch['attention_mask']
        reshaped_mask = token_level_attention_mask.view(batch_size, input_embs.shape[1], self.config['code_len'])
        item_level_attention_mask = reshaped_mask.any(dim=2).long()
        
        outputs = self.gpt2(inputs_embeds=input_embs, attention_mask=item_level_attention_mask)
        
        # --- 後續的排名生成邏輯完全不變 ---
        seq_lens = item_level_attention_mask.sum(dim=1)
        last_hidden_states = outputs.last_hidden_state[torch.arange(batch_size, device=current_device), seq_lens - 1]

        final_states = [head(last_hidden_states) for head in self.pred_heads]
        final_states = torch.stack(final_states, dim=1)
        final_states = F.normalize(final_states, dim=-1)

        token_emb = self.gpt2.wte.weight
        token_emb = F.normalize(token_emb, dim=-1)
        
        if self.item_id2tokens.device != current_device:
            self.item_id2tokens = self.item_id2tokens.to(current_device)
            
        all_item_embs = self.gpt2.wte(self.item_id2tokens)
        all_item_embs = F.normalize(all_item_embs, dim=-1)
        
        item_scores = torch.einsum('bld,ild->bi', final_states, all_item_embs)
        
        _, topk_indices = torch.topk(item_scores, k=topk, dim=1)
        
        return topk_indices

    def evaluate_step(self, batch: Dict, topk_list: List[int]) -> Dict[str, float]:
        """
        【RPG 專屬評估邏輯】
        生成一個 item_id 的排名列表，並計算 Recall 和 NDCG。
        """
        max_k = max(topk_list)
        
        # 1. 生成排名列表 (0-based item index)
        ranked_item_indices = self._generate_ranklist(batch, topk=max_k)
        
        # 2. 獲取真實的 target item id (0-based)
        target_ids = batch['target_ids'].unsqueeze(1) # (B, 1)

        # 3. 計算命中位置
        # ranked_item_indices: (B, K), target_ids: (B, 1)
        # hits: (B, K), True 表示命中
        hits = (ranked_item_indices == target_ids).cpu()
        
        # 4. 計算各項指標
        batch_metrics = {}
        for k in topk_list:
            pos_index_k = hits[:, :k]
            recall = recall_at_k(pos_index_k, k).mean().item()
            ndcg = ndcg_at_k(pos_index_k, k).mean().item()
            batch_metrics[f'Recall@{k}'] = recall
            batch_metrics[f'NDCG@{k}'] = ndcg
            
        return batch_metrics