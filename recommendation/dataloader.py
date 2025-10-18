import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from functools import partial
import numpy as np # 需要 numpy

class GenRecDataLoader(DataLoader):
    """
    【最終強化版 - 高效且完全解耦】
    Dataset 只提供原始 Item IDs。
    Collate Function 在這裡執行模型特定的轉換（Item ID -> Code Token）。
    """
    def __init__(self, dataset, model, item_to_code_map: dict, # ✅ 新增 item_to_code_map
                 batch_size=32, shuffle=True, num_workers=4, 
                 pad_token_id=0, code_len=4, # ✅ 新增 code_len (用於 RPG target_codes)
                 **kwargs):
        
        task_type = model.task_type
        
        # ✅ Collate Function 現在需要 item_to_code_map
        if task_type == 'generative':
            final_collate_fn = partial(self._collate_generative_tokens, 
                                       item_to_code=item_to_code_map, 
                                       pad_token_id=pad_token_id, 
                                       num_levels=code_len) # 傳入 num_levels
        elif task_type == 'retrieval':
            final_collate_fn = partial(self._collate_retrieval_ids, 
                                       item_to_code=item_to_code_map, 
                                       pad_token_id=pad_token_id, 
                                       num_levels=code_len) # 傳入 num_levels
        else:
            raise ValueError(f"不支援的模型任務類型: {task_type}")

        is_gpu_training = (torch.cuda.is_available() and num_workers > 0)

        super().__init__(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, collate_fn=final_collate_fn,
            pin_memory=is_gpu_training,
            persistent_workers=is_gpu_training if num_workers > 0 else False,
            **kwargs
        )
    
    @staticmethod
    def _collate_generative_tokens(batch, item_to_code, pad_token_id, num_levels):
        """
        【專用方法 - 生成式 - 已修正】
        直接從 Dataset 獲取 Code Token 序列，進行壓扁和 Padding。
        """
        # Batch 是一個列表: [{'history': [[c1,c2..],[..]], 'target_code': [tc1, tc2..]}, ...]
        
        # ✅ 關鍵修正：直接獲取 Dataset 準備好的 Code Token 序列
        histories_codes = [item['history'] for item in batch] # 這是 List[List[List[int]]]
        target_codes = [item['target_code'] for item in batch] # 這是 List[List[int]]

        # 壓扁 Code Token 序列 (這部分邏輯不變)
        seqs = [torch.tensor([code for item_codes in h for code in item_codes], dtype=torch.long) 
                for h in histories_codes]
        
        padded_histories = pad_sequence(seqs, batch_first=True, padding_value=pad_token_id)
        attention_masks = (padded_histories != pad_token_id).long() # Token-level mask

        # Target Codes 已經是 List[List[int]]，直接轉 Tensor
        target_codes_tensor = torch.tensor(target_codes, dtype=torch.long) # Shape: (B, L_code)
        
        # 返回 TIGER/SIMPLE_GPT 需要的字段
        return {
            'input_ids': padded_histories,     # 壓扁的 Code Token 序列
            'attention_mask': attention_masks, # Token-level mask
            'labels': target_codes_tensor,     # Target Code Token 序列
        }

    @staticmethod
    def _collate_retrieval_ids(batch, item_to_code, pad_token_id, num_levels):
        """
        【專用方法 - 檢索式 - 已修正 Key 錯誤】
        接收原始 Item ID，Padding Item ID 序列，準備 Target Codes/IDs。
        """
        # Batch 是一個列表: [{'hist_ids': [1based], 'history': [codes], 'target_code': [tcs], 'target_id': 0based}, ...]
        histories_item_ids_1based = [item['hist_ids'] for item in batch] # 獲取 1-based history for RPG input
        target_codes = [item['target_code'] for item in batch]           # 獲取 target codes for loss
        target_ids_0based_scalar = [item['target_id'] for item in batch]  # 獲取 0-based target id for evaluation

        # --- 準備帶 padding 的 label 序列 (用於 forward loss) ---
        label_pad_value = -100
        target_ids_sequence = []
        for item in batch:
            # ✅ 關鍵修正 1：使用 'history_raw_padded' (0-based padded sequence) 來確定長度
            #    因為 labels_seq 的長度需要和 GPT-2 輸入序列的長度嚴格一致
            original_hist_len = len(item['history_raw_padded']) # 獲取原始 padding 後的長度
            
            # ✅ 關鍵修正 2：使用 'target_id' 獲取目標 ID
            tgt_id_0based = item['target_id']
            
            # 創建 label 序列，只有最後一個非 padding 位置是真實 ID
            current_labels = [label_pad_value] * original_hist_len
            # 找到最後一個非 padding 的位置 (需要原始的 0-based ID 列表來判斷 padding)
            hist_ids_0based_padded = item['history_raw_padded']
            last_valid_idx = -1
            for idx in range(len(hist_ids_0based_padded) - 1, -1, -1):
                 if hist_ids_0based_padded[idx] != 0: # 假設 0 是 Item ID padding
                      last_valid_idx = idx
                      break
            if last_valid_idx != -1:
                 current_labels[last_valid_idx] = tgt_id_0based

            target_ids_sequence.append(current_labels)
        
        # --- Padding history (1-based item ids for RPG input) ---
        padded_histories = pad_sequence(
            [torch.tensor(h, dtype=torch.long) for h in histories_item_ids_1based],
            batch_first=True, padding_value=0 # Item ID pad is 0
        )
        attention_masks = (padded_histories != 0).long()

        # --- Padding label sequence ---
        padded_labels = pad_sequence(
            [torch.tensor(l[:padded_histories.shape[1]], dtype=torch.long) # 確保 label 序列不超過 history 長度
             for l in target_ids_sequence],
            batch_first=True, padding_value=label_pad_value
        )
        # 確保 padded_labels 和 padded_histories 的序列長度維度一致
        if padded_labels.shape[1] != padded_histories.shape[1]:
             # 如果不一致，以 history 為準進行調整 (通常是 labels 較短)
             pad_width = padded_histories.shape[1] - padded_labels.shape[1]
             if pad_width > 0:
                  padding = torch.full((padded_labels.shape[0], pad_width), label_pad_value, dtype=torch.long)
                  padded_labels = torch.cat([padded_labels, padding], dim=1)
             elif pad_width < 0: # Labels 比 history 長，截斷 (理論上不應發生)
                  padded_labels = padded_labels[:, :padded_histories.shape[1]]


        target_codes_tensor = torch.tensor(target_codes, dtype=torch.long)
        target_ids_scalar_tensor = torch.tensor(target_ids_0based_scalar, dtype=torch.long) # (B,)
        
        return {
            'input_ids': padded_histories,          # 1-based Item ID 序列 (B, L_item)
            'attention_mask': attention_masks,      # Item-level Mask (B, L_item)
            'labels_seq': padded_labels,            # Target ID 序列帶 padding (B, L_item)
            'target_codes': target_codes_tensor,    # Target Code Token 序列 (B, L_code)
            'target_ids': target_ids_scalar_tensor  # 單個 Target Item ID (B,)
        }