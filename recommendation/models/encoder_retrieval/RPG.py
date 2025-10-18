# 檔案路徑: recommendation/models/retrieval/RPG.py (深度診斷版)

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

# Get logger for this module
logger = logging.getLogger(__name__)

class ResBlock(nn.Module):
    def __init__(self, hidden_size): super().__init__(); self.linear = nn.Linear(hidden_size, hidden_size); torch.nn.init.zeros_(self.linear.weight); self.act = nn.SiLU()
    def forward(self, x): return x + self.act(self.linear(x))

class RPG(AbstractModel):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)
        model_params = self.config['model_params']; token_params = self.config['token_params']
        # Load codebook to CPU first
        self.item_id2tokens_cpu = self._load_codebook_as_tensor(config['code_path']) 
        self.item_id2tokens = None # Will be moved to device on first use
        self.n_items = self.item_id2tokens_cpu.shape[0]
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
        try:
            codes_arr = np.load(code_path, allow_pickle=True)
            codes_mat = np.vstack(codes_arr) if codes_arr.dtype == object else codes_arr
            dummy_row = np.zeros((1, codes_mat.shape[1]), dtype=codes_mat.dtype)
            full_codes = np.vstack([dummy_row, codes_mat])
            tensor = torch.from_numpy(full_codes).long()
            self.logger.info(f"Codebook loaded successfully from {code_path}, shape: {tensor.shape}")
            # Basic validation
            max_val = tensor.max().item()
            min_val = tensor.min().item()
            expected_max = self.config['token_params']['vocab_size'] - 1 # Approximation
            self.logger.info(f"Codebook value range: [{min_val}, {max_val}] (Expected max ~{expected_max})")
            if max_val >= self.config['token_params']['vocab_size'] or min_val < 0:
                 self.logger.error(f"FATAL: Codebook values out of expected range [0, {self.config['token_params']['vocab_size']-1}]!")
            return tensor
        except Exception as e:
            self.logger.error(f"Failed to load codebook from {code_path}: {e}")
            raise

    def _ensure_item_id2tokens_on_device(self, device):
        """ Ensure item_id2tokens tensor is on the correct device """
        if self.item_id2tokens is None or self.item_id2tokens.device != device:
             self.item_id2tokens = self.item_id2tokens_cpu.to(device)
             self.logger.debug(f"Moved item_id2tokens to device: {device}")


    def forward(self, batch: Dict) -> Dict:
        input_item_ids = batch['input_ids']     # (B, L_item) 1-based IDs
        attention_mask = batch['attention_mask'] # (B, L_item) Item-level Mask
        labels_seq = batch['labels_seq']        # (B, L_item) Target ID sequence (-100 padding)
        target_tokens = batch['target_codes']   # (B, L_code) Target Code Tokens
        batch_size = input_item_ids.shape[0]
        current_device = input_item_ids.device

        self._ensure_item_id2tokens_on_device(current_device)

        # --- 1. Item ID -> Item Embedding ---
        try:
            input_tokens = self.item_id2tokens[input_item_ids] # (B, L_item, L_code)
        except IndexError as e:
            self.logger.error(f"IndexError during item_id2tokens lookup! Input IDs range: [{input_item_ids.min().item()}, {input_item_ids.max().item()}], item_id2tokens shape: {self.item_id2tokens.shape}. Error: {e}")
            raise
        
        token_embs = self.gpt2.wte(input_tokens)           # (B, L_item, L_code, D)
        input_embs = token_embs.mean(dim=2)              # (B, L_item, D)

        # --- 2. GPT-2 ---
        outputs = self.gpt2(inputs_embeds=input_embs, attention_mask=attention_mask)
        gpt2_output_states = outputs.last_hidden_state # (B, L_item, D)

        # --- 3. Prediction Heads ---
        seq_lens = attention_mask.sum(dim=1); valid_lens = torch.clamp(seq_lens - 1, min=0) 
        last_hidden_states = gpt2_output_states[torch.arange(batch_size, device=current_device), valid_lens] # (B, D)
        
        head_outputs = [head(last_hidden_states) for head in self.pred_heads]
        final_states = torch.stack(head_outputs, dim=1) # (B, L_code, D) - Use last state only for prediction
        final_states_norm = F.normalize(final_states, dim=-1)

        # --- 4. Calculate Loss ---
        label_mask = labels_seq.view(-1) != -100 # (B * L_item,)
        valid_indices = label_mask.nonzero(as_tuple=True)[0] 
        
        if valid_indices.numel() == 0: return {'loss': torch.tensor(0.0, device=current_device, requires_grad=True)}
            
        # Get target code tokens corresponding to valid labels
        batch_indices = torch.div(valid_indices, labels_seq.shape[1], rounding_mode='floor')
        valid_target_tokens = target_tokens[batch_indices] # (Num_Valid, L_code)

        # Get the corresponding user representations (only need representations for batches with valid labels)
        # We use the final_states calculated from the *last* hidden state
        selected_user_states = final_states_norm[batch_indices] # (Num_Valid, L_code, D)

        # Prepare global token embeddings
        token_emb = self.gpt2.wte.weight; token_emb_norm = F.normalize(token_emb, dim=-1)
        bases = torch.tensor(self.config['bases'], device=current_device); vocab_sizes = torch.tensor(self.config['vocab_sizes'], device=current_device)
        
        losses = []
        
        # +++++++++++++++++++++ DEEP DEBUG BLOCK +++++++++++++++++++++
        # Keep debug flag at class level
        if not self._debug_printed_forward and selected_user_states.shape[0] > 0:
            self.logger.info("\n" + "="*30 + " DEEP FORWARD DEBUG (First Valid Sample) " + "="*30)
            sample_idx_in_valid = 0
            original_flat_idx = valid_indices[sample_idx_in_valid].item()
            original_batch_idx = torch.div(original_flat_idx, labels_seq.shape[1], rounding_mode='floor').item()
            original_seq_idx = original_flat_idx % labels_seq.shape[1]
            
            self.logger.info(f"--- Input Check (Sample {original_batch_idx}, Seq Pos {original_seq_idx}) ---")
            # Log raw data before transformations for clarity
            self.logger.info(f"  Input Item IDs (1-based, padded): {batch['input_ids'][original_batch_idx].cpu().numpy()}")
            self.logger.info(f"  Attention Mask: {batch['attention_mask'][original_batch_idx].cpu().numpy()}")
            self.logger.info(f"  Labels Seq (with -100): {batch['labels_seq'][original_batch_idx].cpu().numpy()}")
            self.logger.info(f"  Target Codes (Global Offset): {batch['target_codes'][original_batch_idx].cpu().numpy()}")
            self.logger.info(f"--- Intermediate Check ---")
            self.logger.info(f"  Input Embeddings Shape (Mean): {input_embs.shape}")
            self.logger.info(f"  GPT-2 Output States Shape: {gpt2_output_states.shape}")
            self.logger.info(f"  Last Hidden State Shape: {last_hidden_states.shape}")
            self.logger.info(f"  Final States Shape (After Heads): {final_states.shape}")
            self.logger.info(f"  Final States Norm Shape (Normalized): {final_states_norm.shape}")
            self.logger.info(f"--- Masking Check ---")
            self.logger.info(f"  Label Mask Shape: {label_mask.shape}")
            self.logger.info(f"  Num Valid Labels in Batch: {valid_indices.numel()}")
            self.logger.info(f"  Selected User States Shape: {selected_user_states.shape}")
            self.logger.info(f"  Valid Target Tokens Shape: {valid_target_tokens.shape}")
            self.logger.info(f"--- Target for Loss (Sample {sample_idx_in_valid} in valid subset) ---")
            target_codes_sample = valid_target_tokens[sample_idx_in_valid].cpu().numpy()
            self.logger.info(f"  Target Codes (Global Offset): {target_codes_sample}")
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        for i in range(self.n_pred_head):
            # --- Get Embeddings for this head ---
            start_idx = bases[i] + 1; end_idx = min(start_idx + vocab_sizes[i], token_emb.shape[0]) 
            token_embs_i = token_emb_norm[start_idx:end_idx] # (K, D)
            actual_K_for_logits = token_embs_i.shape[0]

            # --- Calculate Logits ---
            # Use the user representation corresponding to the *correct batch index*
            user_repr_i = selected_user_states[:, i, :] # (Num_Valid, D)
            logits_i = torch.matmul(user_repr_i, token_embs_i.T) / self.temperature # (Num_Valid, K)
            
            # --- Calculate 0-based Labels ---
            labels_i = valid_target_tokens[:, i] - bases[i] - 1 # (Num_Valid,)

            # +++++++++++++++++++++ DEEP DEBUG BLOCK (Logits Check) ++++++++++++++
            if not self._debug_printed_forward and selected_user_states.shape[0] > 0:
                logits_sample_i = logits_i[sample_idx_in_valid]; label_sample_i = labels_i[sample_idx_in_valid].item(); vocab_size_i = vocab_sizes[i].item()
                if label_sample_i < 0 or label_sample_i >= actual_K_for_logits: self.logger.error(f"❌ Head {i}: Label {label_sample_i} OUT OF BOUNDS {actual_K_for_logits}"); target_logit_val = -float('inf')
                else: target_logit_val = logits_sample_i[label_sample_i].item()
                max_logit_val, max_logit_idx = torch.max(logits_sample_i, dim=0); max_logit_val, max_logit_idx = max_logit_val.item(), max_logit_idx.item()
                is_aligned = (label_sample_i == max_logit_idx); status_symbol = "✅" if is_aligned else "❌"
                self.logger.info(f"--- Head {i} Loss Calc Debug (Sample {sample_idx_in_valid}) ---")
                self.logger.info(f"  Target Token (Global): {target_codes_sample[i]} | Label (0-based): {label_sample_i}")
                # Add check for embedding values
                self.logger.info(f"  User Repr (Head {i}) Mean/Std: {user_repr_i[sample_idx_in_valid].mean().item():.4f} / {user_repr_i[sample_idx_in_valid].std().item():.4f} | IsFinite: {torch.isfinite(user_repr_i[sample_idx_in_valid]).all().item()}")
                self.logger.info(f"  Code Embs (Head {i}) Mean/Std: {token_embs_i.mean().item():.4f} / {token_embs_i.std().item():.4f} | IsFinite: {torch.isfinite(token_embs_i).all().item()}")
                self.logger.info(f"  Logits Shape (NumValid, K): {logits_i.shape} | Logits[0] Mean/Std: {logits_sample_i.mean().item():.4f} / {logits_sample_i.std().item():.4f} | IsFinite: {torch.isfinite(logits_sample_i).all().item()}")
                self.logger.info(f"  Logit for Target[{label_sample_i}]: {target_logit_val:.4f}")
                self.logger.info(f"  Max Logit Idx/Val: {max_logit_idx} / {max_logit_val:.4f}")
                self.logger.info(f"  {status_symbol} Alignment: {'MATCHED' if is_aligned else 'MISALIGNED!'}")
                if not is_aligned: self.logger.warning(f"    Gap: {max_logit_val - target_logit_val:.4f}")
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            if not (torch.all(labels_i >= 0) and torch.all(labels_i < logits_i.shape[1])):
                 self.logger.error(f"FATAL: Head {i} labels out of bounds! Min: {labels_i.min()}, Max: {labels_i.max()}, Logits Dim 1: {logits_i.shape[1]}. Skipping."); losses.append(torch.tensor(100.0, device=current_device, requires_grad=True)); continue
            
            losses.append(self.loss_fct(logits_i, labels_i))

        if not self._debug_printed_forward and selected_user_states.shape[0] > 0: 
             self.logger.info("="*80 + "\n")
             self._debug_printed_forward = True
        
        if losses: loss = torch.mean(torch.stack(losses))
        else: loss = torch.tensor(0.0, device=current_device, requires_grad=True)

        # Additional Debugging: Log final loss value
        if not self._debug_printed_forward: # Log only once
            self.logger.info(f"--- Final Loss Calculation ---")
            self.logger.info(f"  Number of Losses Stacked: {len(losses)}")
            self.logger.info(f"  Mean Loss: {loss.item():.6f}")

        return {'loss': loss}

    def _generate_ranklist(self, batch: Dict, topk: int) -> torch.Tensor:
        input_item_ids = batch['input_ids']; attention_mask = batch['attention_mask']; batch_size = input_item_ids.shape[0]
        current_device = input_item_ids.device
        self._ensure_item_id2tokens_on_device(current_device)
        input_tokens = self.item_id2tokens[input_item_ids]; token_embs = self.gpt2.wte(input_tokens); input_embs = token_embs.mean(dim=2)
        outputs = self.gpt2(inputs_embeds=input_embs, attention_mask=attention_mask)
        seq_lens = attention_mask.sum(dim=1); valid_lens = torch.clamp(seq_lens - 1, min=0)
        last_hidden_states = outputs.last_hidden_state[torch.arange(batch_size, device=current_device), valid_lens]
        final_states = [head(last_hidden_states) for head in self.pred_heads]; final_states = torch.stack(final_states, dim=1); final_states = F.normalize(final_states, dim=-1)
        token_emb = self.gpt2.wte.weight; token_emb_norm = F.normalize(token_emb, dim=-1)
        # ✅ Generate uses all item embeddings, need to load them to device
        all_item_embs = self.gpt2.wte(self.item_id2tokens); all_item_embs = F.normalize(all_item_embs, dim=-1)
        item_scores = torch.einsum('bld,ild->bi', final_states, all_item_embs)
        _, topk_indices = torch.topk(item_scores, k=topk, dim=1) # index is 1-based item id
        return topk_indices 

    def evaluate_step(self, batch: Dict, topk_list: List[int]) -> Dict[str, float]:
        max_k = max(topk_list)
        ranked_item_indices = self._generate_ranklist(batch, topk=max_k) # 1-based IDs
        target_ids = batch['target_ids'].unsqueeze(1) # 0-based IDs

        # Convert 1-based ranked IDs to 0-based for comparison
        hits = ((ranked_item_indices - 1) == target_ids).cpu() 
        
        batch_metrics = {}
        for k in topk_list:
            pos_index_k = hits[:, :k]
            if pos_index_k.numel() > 0: recall = recall_at_k(pos_index_k, k).mean().item(); ndcg = ndcg_at_k(pos_index_k, k).mean().item()
            else: recall = 0.0; ndcg = 0.0
            batch_metrics[f'Recall@{k}'] = recall; batch_metrics[f'NDCG@{k}'] = ndcg
        return batch_metrics