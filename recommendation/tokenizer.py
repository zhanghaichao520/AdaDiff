import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Any, Callable
import numpy as np
import logging
import traceback

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
    """
    為 RPG 準備數據 (增加健壯性)。
    """
    def __init__(self, config: Dict[str, Any], item_to_code_map: Dict[int, List[int]]):
        super().__init__(config, item_to_code_map)
        logger.info(f"RetrievalTokenizer (RPG) 初始化")

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # ✅ 在最外層添加 try...except
        try:
            histories_item_ids_1based = []
            target_codes = []
            target_ids_0based_scalar = []
            
            # 遍歷 batch，進行初步處理和數據收集
            for i, item in enumerate(batch):
                # 使用 .get 提供默認值，增加健壯性
                hist_ids_0based = item.get('history', []) 
                tgt_id_0based = item.get('target', None) 

                # 如果 target ID 缺失，記錄警告並跳過此樣本
                if tgt_id_0based is None:
                    logger.warning(f"Skipping sample {i} due to missing 'target' key.")
                    continue 
                    
                # 確保 history 是列表 (以防萬一數據格式錯誤)
                if not isinstance(hist_ids_0based, list):
                     logger.warning(f"Sample {i} 'history' is not a list ({type(hist_ids_0based)}). Treating as empty.")
                     hist_ids_0based = []

                # 過濾 Item Padding
                current_hist_1based = [x + 1 for x in hist_ids_0based if x != self.item_pad_id]
                
                # 檢查 target ID 是否有效
                try:
                    tgt_id_int = int(tgt_id_0based)
                    # 檢查 target 是否在 item_to_code_map 中 (可選，但有助於 Debug)
                    if (tgt_id_int + 1) not in self.item_to_code_map:
                         logger.warning(f"Target item ID {tgt_id_int} (0-based) not found in item_to_code_map for sample {i}.")
                         # 可以選擇跳過此樣本，或繼續（使用 code_pad_list）
                         # continue 
                except (ValueError, TypeError):
                     logger.warning(f"Sample {i} has invalid target ID '{tgt_id_0based}'. Skipping.")
                     continue
                
                histories_item_ids_1based.append(current_hist_1based)
                target_ids_0based_scalar.append(tgt_id_int)
                
                # 獲取 target codes
                t_code = self.item_to_code_map.get(tgt_id_int + 1, self.code_pad_list)
                target_codes.append(t_code)

            # 如果過濾後 batch 為空，返回空字典或根據需要處理
            if not histories_item_ids_1based:
                 logger.warning("RetrievalTokenizer called with an empty or fully invalid batch.")
                 # 返回空字典可能導致後續錯誤，返回包含空 Tensor 的字典更安全
                 return {
                     'input_ids': torch.empty((0, 0), dtype=torch.long),
                     'attention_mask': torch.empty((0, 0), dtype=torch.long),
                     'labels_seq': torch.empty((0, 0), dtype=torch.long),
                     'target_codes': torch.empty((0, self.code_len), dtype=torch.long),
                     'target_ids': torch.empty((0,), dtype=torch.long),
                 }


            # --- Padding history (RPG 需要 right-padding) ---
            # ✅ 添加邊界處理: 檢查是否所有歷史都為空
            max_item_seq_len = max(len(h) for h in histories_item_ids_1based) if histories_item_ids_1based else 0

            if max_item_seq_len == 0:
                 # 所有歷史都為空
                 logger.debug("All valid histories in the batch are empty.")
                 batch_size_actual = len(histories_item_ids_1based) # 實際處理的樣本數
                 padded_histories = torch.zeros((batch_size_actual, 0), dtype=torch.long)
                 attention_masks = torch.zeros((batch_size_actual, 0), dtype=torch.long)
                 max_len = 0 
            else:
                 # 正常 Padding
                 try:
                     padded_histories = pad_sequence(
                         [torch.tensor(h, dtype=torch.long) for h in histories_item_ids_1based],
                         batch_first=True, padding_value=self.item_pad_id
                     )
                     attention_masks = (padded_histories != self.item_pad_id).long()
                     max_len = padded_histories.shape[1] 
                 except Exception as pad_exc: # 捕獲 pad_sequence 可能的錯誤
                      logger.error(f"Error during history padding: {pad_exc}")
                      logger.error(f"History sequences lengths: {[len(h) for h in histories_item_ids_1based]}")
                      raise # 重新拋出錯誤，讓 except 塊處理

            # --- 準備帶 padding 的 label 序列 ---
            label_pad_value = -100
            target_ids_sequence = []

            for i in range(len(histories_item_ids_1based)): # 使用過濾後的實際長度
                num_valid_items_in_sample = len(histories_item_ids_1based[i])
                last_valid_idx_new = num_valid_items_in_sample - 1
                
                # 使用上面計算出的 max_len
                current_labels = [label_pad_value] * max_len
                # 確保索引有效
                if last_valid_idx_new >= 0 and last_valid_idx_new < max_len: 
                    # 檢查 target_ids_0based_scalar 是否有對應索引
                    if i < len(target_ids_0based_scalar):
                         current_labels[last_valid_idx_new] = target_ids_0based_scalar[i]
                    else:
                         logger.warning(f"Index mismatch when assigning label for sample {i}.")
                         # 可以選擇填充-100 或跳過，填充-100 更安全

                target_ids_sequence.append(current_labels)

            # --- Padding label sequence ---
            # 使用 pad_sequence 更安全
            if target_ids_sequence:
                 try:
                     padded_labels = pad_sequence(
                         [torch.tensor(l, dtype=torch.long) for l in target_ids_sequence],
                         batch_first=True, padding_value=label_pad_value
                     )
                     # 確保長度一致
                     current_label_len = padded_labels.shape[1]
                     if current_label_len < max_len:
                         pad_width = max_len - current_label_len
                         padding = torch.full((padded_labels.shape[0], pad_width), label_pad_value, dtype=torch.long)
                         padded_labels = torch.cat([padded_labels, padding], dim=1)
                     elif current_label_len > max_len:
                         padded_labels = padded_labels[:, :max_len]
                 except Exception as label_pad_exc:
                     logger.error(f"Error during label padding: {label_pad_exc}")
                     logger.error(f"Label sequences lengths: {[len(l) for l in target_ids_sequence]}")
                     raise # 重新拋出

            else: 
                 padded_labels = torch.empty((len(histories_item_ids_1based), max_len), dtype=torch.long).fill_(label_pad_value)

            # --- Target Tensors ---
            # 確保 target_codes 和 target_ids_scalar 長度與 histories_item_ids_1based 一致
            if len(target_codes) != len(histories_item_ids_1based) or \
               len(target_ids_0based_scalar) != len(histories_item_ids_1based):
                 logger.error("Length mismatch between processed histories and targets. This indicates a bug.")
                 # 返回空字典或拋出錯誤
                 raise ValueError("Internal length mismatch in collate function.")

            target_codes_tensor = torch.tensor(target_codes, dtype=torch.long)
            target_ids_scalar_tensor = torch.tensor(target_ids_0based_scalar, dtype=torch.long)
            
            # --- 返回結果 ---
            return {
                'input_ids': padded_histories,         
                'attention_mask': attention_masks,     
                'labels_seq': padded_labels,           
                'target_codes': target_codes_tensor,    
                'target_ids': target_ids_scalar_tensor 
            }
        # ✅ 添加 except 塊
        except Exception as e: 
            logger.error(f"!!! Unhandled Error in RetrievalTokenizer collate_fn !!!")
            # 打印詳細的 traceback 信息到日誌
            logger.error(traceback.format_exc()) 
            # 返回 None，觸發主線程的 AttributeError，提示我們這裡出錯了
            return None

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