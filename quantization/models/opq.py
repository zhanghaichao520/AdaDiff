import torch
from torch.nn import functional as F
from torch import nn
import numpy as np
import faiss
import os
import json
import logging
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from dataset import EmbeddingDataset
import math

class OPQ(nn.Module):
    def __init__(self,
        config: dict,
        emb_path: str,
        codebook_dir: str,):
        """
        初始化 OPQ 量化器
        提供 OPQ 特定的初始化逻辑
        """
        super().__init__()
        
        model_params = config['opq']['model_params']
        self.n_codebook = model_params['n_codebook']
        self.codebook_size = model_params['codebook_size']
        self.opq_use_gpu = model_params['opq_use_gpu']
        self.opq_gpu_id = model_params['opq_gpu_id']
        self.faiss_omp_num_threads = model_params['faiss_omp_num_threads']
        self.test_size = model_params['test_size']
        self.random_state = model_params['random_state']

        self.embedding = np.load(emb_path)
        self.codebook_dir = codebook_dir
        self.n_codebook_bits = self._get_codebook_bits(self.codebook_size)
        self.index_factory = f'OPQ{self.n_codebook},IVF1,PQ{self.n_codebook}x{self.n_codebook_bits}'

        self.mask = self._get_items_for_training(emb_path)
        self.n_digit = self.n_digit()

        self.json_path = os.path.join(codebook_dir, f"{config['dataset_name']}.codebook.json")
        self.pt_path = os.path.join(codebook_dir, f"{config['dataset_name']}.codebook.pt")
        
        logging.info("OPQ 量化器已初始化。")

    def _get_codebook_bits(self, n_codebook):
        """  
        根据码本的规模，计算码本的位数。
        """
        x = math.log2(n_codebook)
        assert x.is_integer() and x >= 0, "Invalid value for n_codebook"

        return int(x)

    
    def forward(self):
        logging.info("OPQ 开始训练...")
        # 是否使用gpu进行faiss训练
        if self.opq_use_gpu:
            res = faiss.StandardGpuResources()
            res.setTempMemory(1024 * 1024 * 512)
            co = faiss.GpuClonerOptions()
            co.useFloat16 = self.n_digit >= 56

        faiss.omp_set_num_threads(self.faiss_omp_num_threads)

        # 构建索引工厂
        index = faiss.index_factory(
            self.embedding.shape[1],
            self.index_factory,
            faiss.METRIC_INNER_PRODUCT
        )

        # 从 CPU 转移到 GPU
        if self.opq_use_gpu:
            index = faiss.index_cpu_to_gpu(res, self.opq_gpu_id, index, co)
        
        # 训练索引
        index.train(self.embedding[self.mask])
        index.add(self.embedding)
        logging.info("OPQ 训练完成，开始生成码本...")

        # 如果使用 GPU，转回 CPU
        if self.opq_use_gpu:
            index = faiss.index_gpu_to_cpu(index)
        
        # 倒排索引
        ivf_index = faiss.downcast_index(index.index)
        invlists = faiss.extract_index_ivf(ivf_index).invlists
        ls = invlists.list_size(0)
        pq_codes = faiss.rev_swig_ptr(invlists.get_codes(0), ls * invlists.code_size)
        pq_codes = pq_codes.reshape(-1, invlists.code_size) # 每行是一个PQ码 num_item*（code_size*code位数） 二进制
        logging.info(f"生成的PQ码维度: {pq_codes.shape}")
        
        # 将二进制PQ码转换为语义ID
        faiss_sem_ids = []
        n_bytes = pq_codes.shape[1]
        for u8code in pq_codes:
            bs = faiss.BitstringReader(faiss.swig_ptr(u8code), n_bytes)
            code = []
            for i in range(self.n_digit):
                code.append(bs.read(self.n_codebook_bits))
            faiss_sem_ids.append(code)
        pq_codes = np.array(faiss_sem_ids)
        logging.info(f"PQ码转换为语义ID维度: {pq_codes.shape}")

        item_ids2sem_ids = {}
        for i in range(pq_codes.shape[0]):
            item_ids2sem_ids[i] = pq_codes[i].tolist()

        # 保存语义ID           
        logging.info(f'保存物品语义ID到 {self.json_path}...')
        with open(self.json_path, 'w') as f:
             json.dump({str(i): item_ids2sem_ids[i] for i in range(len(item_ids2sem_ids))}, f, indent=2)    
        # 将字典转换为 NumPy 数组
        sem_ids_array = np.array([item_ids2sem_ids[i] for i in range(len(item_ids2sem_ids))])
        # 转置数组并保存
        torch.save(torch.from_numpy(sem_ids_array.T).contiguous().long(), self.pt_path)

        logging.info(f"物品语义ID已保存: JSON -> {self.json_path}, PT -> {self.pt_path}")

    def _get_items_for_training(self, emb_path: str):
        """
        标记用于OPQ训练的数据
        """
        full_dataset = EmbeddingDataset(emb_path)
        indices = list(range(len(full_dataset)))
        train_indices, val_indices = self._split_train_val(indices)
        num_items = len(full_dataset)
        # 初始化全 False
        is_train = np.zeros(num_items, dtype=bool) 
        # 标记训练数据
        for idx in train_indices:
            is_train[int(idx)] = True

        return is_train
    
    def _split_train_val(self, indices):
        """
        划分训练集和验证集
        """
        train_indices, val_indices = train_test_split(indices, test_size=0.05, random_state=42)

        return train_indices, val_indices
    
    def n_digit(self):
        """
        返回tokenizer的digit
        """
        return self.n_codebook
    


