import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from functools import partial

class GenRecDataLoader(DataLoader):
    """
    【效能優化版】
    此版本在初始化時根據模型類型選擇專用的 collate_fn，
    避免在訓練迴圈中進行不必要的數據處理，以恢復最高性能。
    """
    def __init__(self, dataset, model, batch_size=32, shuffle=True, num_workers=4, pad_token_id=0, **kwargs):
        
        # ✅ 關鍵改動 1：從模型實例中獲取其任務類型
        task_type = model.task_type
        
        if task_type == 'generative':
            # TIGER 等生成式模型，使用專用的、更輕量的 collate_fn
            final_collate_fn = partial(self._collate_generative, pad_token_id=pad_token_id)
        elif task_type == 'retrieval':
            # RPG 等檢索式模型，使用功能更全的 collate_fn
            final_collate_fn = partial(self._collate_retrieval, pad_token_id=pad_token_id)
        else:
            raise ValueError(f"不支援的模型任務類型: {task_type}")

        is_gpu_training = (torch.cuda.is_available() and num_workers > 0)

        super().__init__(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=num_workers, 
            collate_fn=final_collate_fn, # <--- 傳入選定好的專用方法
            pin_memory=is_gpu_training,
            persistent_workers=is_gpu_training,
            **kwargs
        )
    
    @staticmethod
    def _collate_generative(batch, pad_token_id=0):
        """
        【專用方法】只為 TIGER 等生成式模型準備數據。
        它不處理 'target_id'，性能更高。
        """
        histories = [item['history'] for item in batch]
        target_codes = [item['target_code'] for item in batch]

        seqs = [torch.tensor([code for item_codes in h for code in item_codes], dtype=torch.long) for h in histories]
        
        padded_histories = pad_sequence(seqs, batch_first=True, padding_value=pad_token_id)
        attention_masks = (padded_histories != pad_token_id).long()

        target_codes_tensor = torch.tensor(target_codes, dtype=torch.long)
        
        return {
            'input_ids': padded_histories,
            'attention_mask': attention_masks,
            'labels': target_codes_tensor,
        }

    @staticmethod
    def _collate_retrieval(batch, pad_token_id=0):
        """
        【專用方法】為 RPG 等檢索式模型準備數據。
        它會同時準備 'labels' (用於算 loss) 和 'target_ids' (用於評估)。
        """
        histories = [item['history'] for item in batch]
        target_codes = [item['target_code'] for item in batch]
        target_ids = [item['target_id'] for item in batch]

        seqs = [torch.tensor([code for item_codes in h for code in item_codes], dtype=torch.long) for h in histories]
        
        padded_histories = pad_sequence(seqs, batch_first=True, padding_value=pad_token_id)
        attention_masks = (padded_histories != pad_token_id).long()

        target_codes_tensor = torch.tensor(target_codes, dtype=torch.long)
        target_ids_tensor = torch.tensor(target_ids, dtype=torch.long)
        
        return {
            'input_ids': padded_histories,
            'attention_mask': attention_masks,
            'labels': target_codes_tensor,
            'target_ids': target_ids_tensor
        }