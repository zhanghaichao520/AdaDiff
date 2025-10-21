# 檔案路徑: tokenlization_stage/models/opq.py

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

class OPQ(AbstractVQ):
    """
    Optimized Product Quantization (OPQ) 量化器，基於 Faiss 實現。
    
    此版本經過重構，以確保邏輯正確性和高效性。它採用「一次性擬合」策略，
    分別訓練 OPQ 旋轉矩陣和 ProductQuantizer 碼本，用於對向量進行編碼。
    """
    def __init__(self, config: dict, input_size: int):
        """
        初始化 OPQ 量化器。
        """
        super().__init__(config)
        self.config = config
        self.input_size = input_size

        # 從 config 的 'opq' 節點讀取參數
        model_params = config['opq']['model_params']
        self.num_levels = model_params['num_levels']
        self.codebook_size = model_params['codebook_size']
        
        if 'faiss_omp_num_threads' in model_params:
            faiss.omp_set_num_threads(model_params['faiss_omp_num_threads'])

        # 初始化內部狀態
        self.fitted = False
        self.opq_matrix = None           # 用於儲存訓練好的旋轉矩陣
        self.pq = None                   # 用於儲存訓練好的 ProductQuantizer
        self._embedding_buffer = []
        self._total_item_count = None

        logging.info("OPQ 量化器已初始化 (等待数据攒齐进行拟合)。")

    @property
    def is_iterative(self) -> bool:
        # 覆寫父類屬性，告訴 Trainer 這是一次性擬合的模型
        return False

    def _fit_faiss(self):
        """
        在數據攢齊後，執行一次性的 Faiss 擬合來訓練 OPQ 旋轉矩陣和 PQ 碼本。
        """
        if self.fitted:
            return

        logging.info("OPQ 数据已攒齐，开始一次性 Faiss 拟合...")
        total_start_time = time.time()
        
        embeddings_np = torch.cat(self._embedding_buffer, dim=0).cpu().numpy().astype('float32')
        self._embedding_buffer = []

        # 為了訓練，從數據中隨機抽樣一部分即可
        max_train_samples = 256 * 1000 # 例如，最多使用 25.6 萬個樣本
        train_size = min(len(embeddings_np), max_train_samples)
        train_indices = np.random.choice(len(embeddings_np), size=train_size, replace=False)
        train_vectors = embeddings_np[train_indices]
        logging.info(f"從 {len(embeddings_np)} 個向量中採樣 {train_size} 個用於訓練。")

        # --- 步驟 1: 訓練 OPQ 旋轉矩陣 ---
        logging.info(f"步驟 1: 訓練 OPQ 旋轉矩陣 (num_levels={self.num_levels})... (此步驟可能耗時較長)")
        start_time = time.time()
        # OPQMatrix 的 M 參數就是 num_levels
        opq_matrix = faiss.OPQMatrix(self.input_size, self.num_levels)
        opq_matrix.train(train_vectors)
        self.opq_matrix = opq_matrix
        logging.info(f"步驟 1 完成，耗時 {time.time() - start_time:.2f} 秒。")

        # --- 步驟 2: 訓練 Product Quantizer ---
        logging.info("步驟 2: 應用旋轉矩陣並訓練 PQ 碼本...")
        start_time = time.time()
        # 對訓練數據應用旋轉
        rotated_train_vectors = self.opq_matrix.apply_py(train_vectors)

        # 核心：創建並訓練一個 ProductQuantizer
        n_codebook_bits = int(math.log2(self.codebook_size))
        self.pq = faiss.ProductQuantizer(self.input_size, self.num_levels, n_codebook_bits)
        self.pq.train(rotated_train_vectors)
        logging.info(f"步驟 2 完成，耗時 {time.time() - start_time:.2f} 秒。")
        
        self.fitted = True
        logging.info(f"OPQ 擬合完成，總耗時 {time.time() - total_start_time:.2f} 秒。")

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
                raise ValueError("OPQ 模型需要 'total_item_count' 在 config 中被設置。")

        current_count = sum(len(b) for b in self._embedding_buffer)
        
        if current_count >= self._total_item_count:
            self._fit_faiss()

        # Trainer 看到 loss 為 None 就不會執行 backward
        return (None, None, None)

    def compute_loss(self, forward_outputs, batch_data) -> dict:
        """OPQ 是一次性擬合的，沒有可迭代優化的損失函數。"""
        return {'loss_total': 0.0}

    @torch.no_grad()
    def get_codes(self, batch_data: torch.Tensor) -> torch.Tensor:
        """
        使用已擬合好的 OPQ 矩陣和 PQ 碼本，對輸入的 batch 進行量化。
        """
        if not self.fitted:
            logging.warning("OPQ 在 get_codes 時仍未擬合，將執行緊急擬合。")
            self._fit_faiss()
            if not self.fitted:
                 raise RuntimeError("OPQ 緊急擬合失敗，無法生成 codes。")
        
        # 將輸入數據轉為 CPU 上的 float32 numpy 陣列
        batch_np = batch_data.cpu().numpy().astype('float32')
        
        # ✅ 關鍵邏輯修正：
        # 1. 先對輸入向量應用訓練好的旋轉矩陣
        rotated_batch = self.opq_matrix.apply_py(batch_np)
        
        # 2. 再對旋轉後的向量計算 PQ 碼
        codes_np = self.pq.compute_codes(rotated_batch)
        
        # 將 numpy uint8 陣列轉換為 PyTorch Long Tensor，並移回原始設備
        return torch.from_numpy(codes_np.astype(np.int64)).long().to(batch_data.device)