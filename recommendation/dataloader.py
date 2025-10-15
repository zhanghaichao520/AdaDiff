import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from functools import partial

class GenRecDataLoader(DataLoader):
    """
    為生成式推薦任務優化後的 DataLoader。
    - 使用 @staticmethod 修正了多進程 collate_fn 的問題。
    - 透過 partial 函數靈活傳遞 padding token ID。
    - 預設開啟 pin_memory 和 persistent_workers 以提升效能。
    """
    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=4, pad_token_id=0, **kwargs):
        # 使用 functools.partial 來綁定 pad_token_id 參數
        # 這樣 collate_fn 就能接收到正確的 padding 值
        collate_with_padding = partial(self.collate_fn, pad_token_id=pad_token_id)
        
        # 在 GPU 訓練時，強烈建議開啟 pin_memory 和 persistent_workers
        is_gpu_training = (torch.cuda.is_available() and num_workers > 0)

        super().__init__(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=num_workers, 
            collate_fn=collate_with_padding,
            pin_memory=is_gpu_training,
            persistent_workers=is_gpu_training,
            **kwargs
        )
    
    @staticmethod
    def collate_fn(batch, pad_token_id=0):
        """
        靜態的 collate_fn，安全用於多進程。
        將 batch 內的樣本處理成模型可用的 Tensor 格式。
        """
        # 從 batch 中提取 history 和 target
        histories = [item['history'] for item in batch]
        targets = [item['target'] for item in batch]

        # 將每個用戶的 history code 序列展平 (flatten)
        # 例如：[[1,2], [3,4]] -> [1, 2, 3, 4]
        seqs = [torch.tensor([code for item_codes in h for code in item_codes], dtype=torch.long) for h in histories]
        
        # 對序列進行填充，使 batch 內所有序列長度一致
        padded_histories = pad_sequence(seqs, batch_first=True, padding_value=pad_token_id)
        
        # 創建 attention mask，其中 padding 的位置為 0，非 padding 為 1
        attention_masks = (padded_histories != pad_token_id).long()

        # 將 target list 轉換為一個 Tensor
        targets = torch.tensor(targets, dtype=torch.long)
        
        # 返回符合 Hugging Face T5 模型慣例的字典
        return {
            'input_ids': padded_histories,
            'attention_mask': attention_masks,
            'labels': targets
        }