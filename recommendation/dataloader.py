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
        【專用方法 - 生成式】
        接收原始 Item ID，執行 item2code 轉換，壓扁，Padding。
        """
        # Batch 是一個列表: [{'history': [id1, id2...], 'target': id_tgt}, ...]
        histories_codes = []
        target_codes = []
        # TIGER 需要的 PAD token (通常是 0)
        code_pad_token_list = [pad_token_id] * num_levels 

        for item in batch:
            hist_ids_0based = item['history']
            tgt_id_0based = item['target']
            
            # ✅ 在 Collate 內部執行 item2code
            h_codes = [item_to_code.get(x + 1, code_pad_token_list) for x in hist_ids_0based]
            t_code = item_to_code.get(tgt_id_0based + 1, code_pad_token_list)
            
            histories_codes.append(h_codes)
            target_codes.append(t_code)

        # 壓扁 Code Token 序列
        seqs = [torch.tensor([code for item_codes in h for code in item_codes], dtype=torch.long) 
                for h in histories_codes]
        
        padded_histories = pad_sequence(seqs, batch_first=True, padding_value=pad_token_id)
        attention_masks = (padded_histories != pad_token_id).long() # Token-level mask

        target_codes_tensor = torch.tensor(target_codes, dtype=torch.long)
        
        return {
            'input_ids': padded_histories,
            'attention_mask': attention_masks,
            'labels': target_codes_tensor,
        }

    @staticmethod
    def _collate_retrieval_ids(batch, item_to_code, pad_token_id, num_levels):
        histories_item_ids = []
        target_codes = []
        target_ids_list = [] # 用於構建帶 padding 的 label 序列
        
        id_pad_token = 0
        code_pad_token_list = [pad_token_id] * num_levels
        label_pad_value = -100 # 用於 label padding

        max_hist_len = 0 # 記錄 batch 內最大歷史長度

        for item in batch:
            hist_ids_0based = item['history']
            tgt_id_0based = item['target']

            # RPG 需要 1-based Item IDs for history
            current_hist = [x + 1 for x in hist_ids_0based]
            histories_item_ids.append(current_hist)
            max_hist_len = max(max_hist_len, len(current_hist))
            
            # 準備 target codes (不變)
            t_code = item_to_code.get(tgt_id_0based + 1, code_pad_token_list)
            target_codes.append(t_code)
            
            # ✅ 準備 target ID label 序列 (模仿原始 Tokenizer)
            # 只有最後一個位置是真實 target ID (0-based)，其餘是 -100
            current_labels = [label_pad_value] * len(current_hist)
            if current_labels: # 確保歷史不為空
                current_labels[-1] = tgt_id_0based 
            target_ids_list.append(current_labels)


        # Padding 1-based Item ID 序列 (history)
        padded_histories = pad_sequence(
            [torch.tensor(h, dtype=torch.long) for h in histories_item_ids],
            batch_first=True, padding_value=id_pad_token
        )
        attention_masks = (padded_histories != id_pad_token).long()

        # Padding target ID label 序列
        padded_labels = pad_sequence(
            [torch.tensor(l, dtype=torch.long) for l in target_ids_list],
            batch_first=True, padding_value=label_pad_value
        )

        target_codes_tensor = torch.tensor(target_codes, dtype=torch.long)
        # target_ids_tensor 不再是單個 ID，而是序列
        # target_ids_tensor = torch.tensor(target_ids, dtype=torch.long)
        
        return {
            'input_ids': padded_histories,     # 1-based Item ID 序列
            'attention_mask': attention_masks, # Item-level Mask
            'labels': padded_labels,           # ✅ Target Item ID 序列 (帶 -100 padding)
            'target_codes': target_codes_tensor # Target Item 的 Code Token 序列
        }