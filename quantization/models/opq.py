# /tokenlization_stage/models/opq.py (兼容 Trainer 的版本)

import torch
from torch import nn
import numpy as np
import faiss
import os
import json
import logging
from sklearn.model_selection import train_test_split
import math

from .abstract_vq import AbstractVQ

class OPQ(AbstractVQ):
    def __init__(self, config: dict, input_size: int):
        """
        初始化 OPQ 量化器。
        现在它不读取文件，只准备好接收数据。
        """
        super().__init__(config)
        self.config = config
        self.input_size = input_size

        model_params = config['opq']['model_params']
        self.n_codebook = model_params['n_codebook']
        self.codebook_size = model_params['codebook_size']
        self.opq_use_gpu = model_params['opq_use_gpu']
        self.opq_gpu_id = model_params['opq_gpu_id']
        self.faiss_omp_num_threads = model_params['faiss_omp_num_threads']

        # --- 核心改动 1: 初始化内部状态 ---
        self.fitted = False              # 标记是否已经拟合过
        self.index = None                # 存储训练好的 faiss index
        self._embedding_buffer = []      # ✨ 用来攒数据的缓冲区
        self._total_item_count = None    # ✨ 用来记录数据集的总大小

        logging.info("OPQ 量化器已初始化 (等待数据攒齐进行拟合)。")

    @property
    def is_iterative(self) -> bool:
        # 覆寫父類屬性，告訴 Trainer 我是一次性擬合的！
        return False

    def _fit_faiss(self):
        """
        这是真正执行 Faiss 拟合的私有方法。
        只有在数据攒够后才会被调用。
        """
        if self.fitted:
            return

        logging.info("OPQ 数据已攒齐，开始一次性 Faiss 拟合...")
        # 将缓冲区中的所有 tensor 合并成一个大的 numpy 数组
        embeddings_np = torch.cat(self._embedding_buffer, dim=0).cpu().numpy()
        self._embedding_buffer = [] # 释放内存

        # --- 以下是你原来 forward 方法中的 Faiss 核心逻辑 ---
        # 1. 划分训练/测试掩码
        indices = list(range(len(embeddings_np)))
        train_indices, _ = train_test_split(indices, test_size=0.05, random_state=42)
        mask = np.zeros(len(embeddings_np), dtype=bool)
        mask[train_indices] = True

        # 2. 构建并训练 faiss 索引
        n_codebook_bits = int(math.log2(self.codebook_size))
        index_factory_str = f'OPQ{self.n_codebook},IVF1,PQ{self.n_codebook}x{n_codebook_bits}'
        index = faiss.index_factory(self.input_size, index_factory_str, faiss.METRIC_INNER_PRODUCT)
        
        # ... (此处省略与你原版完全相同的 GPU 设置、index.train, index.add 逻辑) ...
        index.train(embeddings_np[mask])
        index.add(embeddings_np)
        # ----------------------------------------------------------------------
        
        self.index = index
        self.fitted = True
        logging.info(f"OPQ 拟合完成。Index 中包含 {self.index.ntotal} 个向量。")

    def forward(self, batch_data: torch.Tensor) -> tuple:
        """
        forward 的职责变成了“攒数据”。
        Trainer 会在每个 epoch 的每一批数据上调用它。
        """
        # 如果已经拟合过了，直接跳过，提高后续 epoch 的速度
        if self.fitted:
            return (None, None, None)

        # 把送来的 batch 存入缓冲区
        self._embedding_buffer.append(batch_data.detach().cpu())

        # --- 核心改动 2: 检查数据是否攒够 ---
        # 第一次调用时，我们需要从 Trainer 那里获取数据集总大小
        # 这是一个小小的技巧，假设 Trainer 会在 config 中注入这个信息
        if self._total_item_count is None:
            # 这个值需要由 main.py 或 trainer.py 传入 config
            self._total_item_count = self.config.get('total_item_count', -1)
            if self._total_item_count == -1:
                raise ValueError("OPQ 模型需要 'total_item_count' 在 config 中被设置。")

        # 计算当前已收集的数据量
        current_count = sum(len(b) for b in self._embedding_buffer)
        
        # 如果数据攒够了，就触发真正的拟合
        if current_count >= self._total_item_count:
            self._fit_faiss()

        # 为了兼容 Trainer 的返回值，返回空值
        # Trainer 看到 loss 是 None 就不会执行 backward
        return (None, None, None)

    def compute_loss(self, forward_outputs, batch_data) -> dict:
        """OPQ 没有可优化的损失函数。"""
        return {'loss_total': 0.0}

    @torch.no_grad()
    def get_codes(self, batch_data: torch.Tensor) -> torch.Tensor:
        """
        使用已经拟合好的 index 来对传入的 batch 进行量化。
        这个方法在 predict 阶段被调用。
        """
        if not self.fitted:
            # 如果在 predict 时还没拟合（例如 epochs=0），紧急拟合一次
            logging.warning("OPQ 在 get_codes 时仍未拟合，将执行紧急拟合。请确保 fit() 至少被完整地执行过一次。")
            self._fit_faiss()
            if not self.fitted:
                 raise RuntimeError("OPQ 紧急拟合失败，无法生成 codes。")

        batch_np = batch_data.cpu().numpy()
        
        # Faiss search 返回 (distances, indices)，我们只需要 indices (也就是 codes)
        _, codes = self.index.search(batch_np, 1)
        
        # 这里的 codes 是 PQ 码，需要像你原来一样转换为语义 ID
        # 为了演示，我们先假设它已经是最终形态
        # logging.warning("OPQ get_codes() 需要实现从 PQ 码到语义 ID 的转换。")
        # 这是一个 placeholder，你需要把你原来的转换逻辑放进来
        final_codes = np.random.randint(0, self.codebook_size, size=(len(batch_np), self.n_codebook))

        return torch.from_numpy(final_codes).long().to(batch_data.device)