# 檔案路徑: tokenlization_stage/models/pq.py

import torch
from torch import nn
import numpy as np
import faiss
import os
import json
import logging
from sklearn.model_selection import train_test_split
import math
import time

from .abstract_vq import AbstractVQ

class PQ(AbstractVQ):
    """
    Product Quantization (PQ) 量化器，基於 Faiss 實現。
    
    此版本經過優化，專注於穩定性和清晰度。它採用「一次性擬合」策略，
    直接使用 Faiss 核心的 ProductQuantizer 元件進行訓練和編碼。
    """
    def __init__(self, config: dict, input_size: int):
        """
        初始化 PQ 量化器。
        """
        super().__init__(config)
        self.config = config
        self.input_size = input_size

        # 從 config 的 'pq' 節點讀取參數
        model_params = config['pq']['model_params']
        self.num_levels = model_params['num_levels']
        self.codebook_size = model_params['codebook_size']
        
        if 'faiss_omp_num_threads' in model_params:
            faiss.omp_set_num_threads(model_params['faiss_omp_num_threads'])
        
        # 初始化內部狀態
        self.fitted = False
        self.pq = None                   # 直接儲存 ProductQuantizer 物件
        self._embedding_buffer = []
        self._total_item_count = None

        logging.info("PQ 量化器已初始化 (等待数据攒齐进行拟合)。")

    @property
    def is_iterative(self) -> bool:
        # 覆寫父類屬性，告訴 Trainer 這是一次性擬合的模型
        return False

    def _fit_faiss(self):
        """
        在數據攢齊後，執行一次性的 Faiss 擬合來訓練 ProductQuantizer 碼本。
        """
        if self.fitted:
            return

        logging.info("PQ 数据已攒齐，开始一次性 Faiss 拟合...")
        start_time = time.time()
        
        embeddings_np = torch.cat(self._embedding_buffer, dim=0).cpu().numpy().astype('float32')
        self._embedding_buffer = [] # 釋放記憶體

        # 為了訓練碼本，從數據中隨機抽樣一部分即可
        train_size = min(len(embeddings_np), 256 * self.codebook_size)
        train_indices = np.random.choice(len(embeddings_np), size=train_size, replace=False)
        train_vectors = embeddings_np[train_indices]

        # 核心：我們只創建一個 ProductQuantizer 物件，而不是完整的 Faiss Index
        n_codebook_bits = int(math.log2(self.codebook_size))
        self.pq = faiss.ProductQuantizer(self.input_size, self.num_levels, n_codebook_bits)

        logging.info(f"使用 {len(train_vectors)} 個向量訓練 PQ 碼本...")
        self.pq.train(train_vectors)
        
        self.fitted = True
        logging.info(f"PQ 碼本擬合完成，總耗時 {time.time() - start_time:.2f} 秒。")

    def forward(self, batch_data: torch.Tensor) -> tuple:
        """
        forward 的職責是「攢數據」，直到數據量足夠觸發一次性的擬合。
        """
        if self.fitted:
            return (None, None, None)

        self._embedding_buffer.append(batch_data.detach().cpu())

        if self._total_item_count is None:
            self._total_item_count = self.config.get('total_item_count', -1)
            if self._total_item_count == -1:
                raise ValueError("PQ 模型需要 'total_item_count' 在 config 中被設置。")

        current_count = sum(len(b) for b in self._embedding_buffer)
        
        if current_count >= self._total_item_count:
            self._fit_faiss()

        # Trainer 看到 loss 為 None 就不會執行 backward
        return (None, None, None)

    def compute_loss(self, forward_outputs, batch_data) -> dict:
        """PQ 是一次性擬合的，沒有可迭代優化的損失函數。"""
        return {'loss_total': 0.0}

    @torch.no_grad()
    def get_codes(self, batch_data: torch.Tensor) -> torch.Tensor:
        """
        使用已擬合好的 ProductQuantizer 對輸入的 batch 進行量化，獲取 PQ 碼。
        """
        if not self.fitted:
            logging.warning("PQ 在 get_codes 時仍未擬合，將執行緊急擬合。")
            self._fit_faiss()
            if not self.fitted:
                 raise RuntimeError("PQ 緊急擬合失敗，無法生成 codes。")
        
        # 將輸入數據轉為 CPU 上的 float32 numpy 陣列
        batch_np = batch_data.cpu().numpy().astype('float32')
        
        # 使用 self.pq.compute_codes() 是獲取 PQ 碼最直接、最高效的方式
        codes_np = self.pq.compute_codes(batch_np)
        
        # 將 numpy uint8 陣列轉換為 PyTorch Long Tensor，並移回原始設備
        return torch.from_numpy(codes_np.astype(np.int64)).long().to(batch_data.device)