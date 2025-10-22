import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Any, Callable
import numpy as np
import logging

logger = logging.getLogger(__name__)

class BaseTokenizer:
    """Tokenizer 基類 (保持不變)"""
    def __init__(self, config: Dict[str, Any], item_to_code_map: Dict[int, List[int]]):
        self.config = config
        self.item_to_code_map = item_to_code_map
        self.pad_token_id = config['token_params']['pad_token_id']
        self.code_len = config['code_len']
        self.item_pad_id = 0 
        self.code_pad_list = [self.pad_token_id] * self.code_len

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def _pad_sequences(self, sequences: List[torch.Tensor], padding_side: str, padding_value: int) -> torch.Tensor:
        """
        輔助函數：執行 Padding (使用更可靠的 left-padding)。
        """
        # sequences: list of tensors, where each tensor is a flattened sequence of valid codes
        
        # 獲取每個序列的實際長度
        lengths = [len(s) for s in sequences]
        max_len = max(lengths) if lengths else 0

        if padding_side == 'right':
            padded_sequences = []
            for s in sequences:
                pad_len = max_len - len(s)
                # torch.nn.functional.pad only supports same padding on both sides easily
                # Manual padding is clearer here
                if pad_len > 0:
                     padding = torch.full((pad_len,), padding_value, dtype=s.dtype)
                     padded_sequences.append(torch.cat((s, padding)))
                else:
                     padded_sequences.append(s)
            if padded_sequences:
                 return torch.stack(padded_sequences)
            else:
                 return torch.empty((0, max_len), dtype=torch.long) # Handle empty batch

        elif padding_side == 'left':
            padded_sequences = []
            for s in sequences:
                pad_len = max_len - len(s)
                if pad_len > 0:
                     padding = torch.full((pad_len,), padding_value, dtype=s.dtype)
                     padded_sequences.append(torch.cat((padding, s)))
                else:
                     padded_sequences.append(s)
            if padded_sequences:
                 return torch.stack(padded_sequences)
            else:
                 return torch.empty((0, max_len), dtype=torch.long) # Handle empty batch
        else:
            raise ValueError(f"不支持的 padding_side: {padding_side}")

class GenerativeTokenizer(BaseTokenizer):
    """
    為 TIGER 和 GPT-2 準備數據 (最終修正版 v3 - 移除 Padding)。
    核心：
    1. 移除 Item Padding 0。
    2. 壓平有效 Code。
    3. 使用可靠的 Padding 函數填充。
    """
    def __init__(self, config: Dict[str, Any], item_to_code_map: Dict[int, List[int]], padding_side: str):
        super().__init__(config, item_to_code_map)
        if padding_side not in ('left', 'right'):
            raise ValueError("padding_side 必須是 'left' 或 'right'")
        self.padding_side = padding_side
        logger.info(f"GenerativeTokenizer 初始化, padding_side='{padding_side}'")

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        
        batch_sequences_flat_valid = [] # 存儲每個樣本壓平後的 *有效* code token
        target_codes = []

        logger.debug(f"--- Tokenizer Start Batch (Size: {len(batch)}) ---") 

        for i, item in enumerate(batch):
            hist_ids_0based = item['history'] # e.g., [0, 0, 5, 10] (left-padded from dataset)
            tgt_id_0based = item['target']
            
            logger.debug(f"Sample {i} Raw History Item IDs: {hist_ids_0based}") 

            # 1. ✅ 過濾掉 Item Padding (0)
            valid_hist_ids = [x for x in hist_ids_0based if x != self.item_pad_id]
            logger.debug(f"Sample {i} Valid History Item IDs: {valid_hist_ids}") 

            # 2. 將有效 Item IDs 轉換為 Code 列表
            valid_hist_codes = []
            if valid_hist_ids: 
                valid_hist_codes = [self.item_to_code_map.get(x + 1, self.code_pad_list) 
                                    for x in valid_hist_ids]

            # 3. 壓平 *有效* Code Token 序列
            seq_flat_valid = [code for item_codes in valid_hist_codes for code in item_codes]
            logger.debug(f"Sample {i} Flattened Valid Codes (len={len(seq_flat_valid)}): {seq_flat_valid[:20]}...{seq_flat_valid[-20:]}") 
            
            batch_sequences_flat_valid.append(torch.tensor(seq_flat_valid, dtype=torch.long))

            # Target (保持不變)
            t_code = self.item_to_code_map.get(tgt_id_0based + 1, self.code_pad_list)
            target_codes.append(t_code)

        # 4. ✅ 對壓平後的 *有效* 序列進行 Padding (使用修正後的函數)
        padded_histories = self._pad_sequences(
            batch_sequences_flat_valid, 
            padding_side=self.padding_side, 
            padding_value=self.pad_token_id 
        )
        logger.debug(f"Padded Histories Shape: {padded_histories.shape}") 
        if len(batch) > 0 and padded_histories.numel() > 0: # Add check for empty tensor
            logger.debug(f"Padded Histories[0] (first 30): {padded_histories[0][:30].tolist()}") 
            logger.debug(f"Padded Histories[0] (last 30): {padded_histories[0][-30:].tolist()}") 
        
        # 5. 生成 Attention Mask
        attention_masks = (padded_histories != self.pad_token_id).long() 
        if len(batch) > 0 and attention_masks.numel() > 0: # Add check for empty tensor
            logger.debug(f"Attention Mask[0] (first 30): {attention_masks[0][:30].tolist()}") 
            logger.debug(f"Attention Mask[0] (last 30): {attention_masks[0][-30:].tolist()}") 
        
        # 6. Target Tensor (保持不變)
        target_codes_tensor = torch.tensor(target_codes, dtype=torch.long)
        
        logger.debug(f"--- Tokenizer End Batch ---") 

        # 預期輸出:
        # GPT-2 (left): input_ids=[0,...,0, C6.., C11..] mask=[0,...,0, 1.., 1..]
        # TIGER (right): input_ids=[C6.., C11.., 0,...,0] mask=[1.., 1.., 0,...,0]
        
        return {
            'input_ids': padded_histories,
            'attention_mask': attention_masks,
            'labels': target_codes_tensor,
        }

# --- RetrievalTokenizer 和 get_tokenizer 保持不變 ---
class RetrievalTokenizer(BaseTokenizer):
     # (保持之前的版本不變)
    def __init__(self, config: Dict[str, Any], item_to_code_map: Dict[int, List[int]]):
        super().__init__(config, item_to_code_map)
        logger.info(f"RetrievalTokenizer (RPG) 初始化")
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # (保持之前的版本不變)
        pass # 省略代碼

def get_tokenizer(model_name: str, config: Dict[str, Any], item_to_code_map: Dict[int, List[int]]) -> Callable:
    """
    (保持之前的版本不變)
    """
    model_name = model_name.upper()
    if model_name == 'TIGER':
        return GenerativeTokenizer(config, item_to_code_map, padding_side='right')
    elif 'GPT2' in model_name or 'LLM' in model_name:
        return GenerativeTokenizer(config, item_to_code_map, padding_side='left')
    elif model_name == 'RPG':
        return RetrievalTokenizer(config, item_to_code_map)
    else:
        logger.warning(f"未知模型: {model_name}，使用 left-padding GenerativeTokenizer")
        return GenerativeTokenizer(config, item_to_code_map, padding_side='left')