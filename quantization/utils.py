import json
import os
import pickle
import importlib
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import trange
import sys

# quantizers/utils.py

import os
import logging
from datetime import datetime
from collections import defaultdict
import numpy as np

def get_model(model_name: str):
    """
    为平级结构设计的模型工厂函数。
    它会从 models/{model_name}.py 文件中加载模型类。
    
    Args:
        model_name (str): 模型名称 (例如 'rqvae')，来自命令行参数。

    Returns:
        torch.nn.Module: 加载到的模型类 (例如 RQVAE class)。
    """
    try:
        # 1. 动态构建模块路径，例如 'models.rqvae'
        module_path = f'models.{model_name}'
        
        # 2. 导入这个具体的模块 (即 models/rqvae.py 文件)
        model_module = importlib.import_module(module_path)
        
        # 3. 约定类名为模型名的大写形式，例如 'RQVAE'
        class_name = model_name.upper()
        model_class = getattr(model_module, class_name)
        
    except (ImportError, AttributeError) as e:
        # 如果找不到模块或类，给出清晰的错误提示
        print(f"ERROR: 尝试加载模型 '{model_name}' 时失败。 异常: {e}")
        raise ValueError(
            f'Model "{model_name}" not found. '
            f'请检查以下几点：\n'
            f'1. 在 "models/" 文件夹中是否存在一个名为 "{model_name}.py" 的文件。\n'
            f'2. 在该文件中，是否定义了一个名为 "{class_name}" 的类。'
        )
        
    return model_class
def setup_paths(args):
    """根据输入参数构建所有需要的路径"""
    input_embedding_filename = f"{args.dataset_name}.emb-{args.embedding_suffix}.npy"
    embedding_path = os.path.join(args.data_base_path, args.dataset_name, input_embedding_filename)
    
    # 输出目录现在包含量化器名称和特征后缀，实现完全隔离
    output_base_dir = f"{args.model_name}/{args.embedding_suffix}"
    log_dir = os.path.join(args.log_base_path, args.dataset_name, output_base_dir)
    ckpt_dir = os.path.join(args.ckpt_base_path, args.dataset_name, output_base_dir)
    codebook_dir = os.path.join(args.codebook_base_path, args.dataset_name)

    for d in [log_dir, ckpt_dir, codebook_dir]:
        os.makedirs(d, exist_ok=True)
        
    print("--- 自动构建路径 (模块化版) ---")
    print(f"输入特征: {embedding_path}")
    print(f"配置文件: {args.config_path}")
    print(f"日志目录: {log_dir}")
    print(f"模型目录: {ckpt_dir}")
    print(f"码本目录: {codebook_dir}")
    print("------------------------------------\n")

    return embedding_path, log_dir, ckpt_dir, codebook_dir

def setup_logging(log_dir):
    """配置日志记录器"""
    log_filename = os.path.join(log_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s',
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
    )
    logging.info("Logging setup complete.")

def build_dedup_layer(base_codes_np: np.ndarray, vocab_size: int):
    """
    为基础码本添加一个去重层。
    对基础码完全相同的条目，在各自簇内部分配 0..k-1 的ID。
    这是一个通用逻辑，可以被任何产生分层码本的模型复用。
    """
    logging.info("构建去重层...")
    N = base_codes_np.shape[0]
    groups = defaultdict(list)
    for idx, key in enumerate(map(tuple, base_codes_np)):
        groups[key].append(idx)

    dedup_layer = np.zeros((N, 1), dtype=np.int64)
    max_dup, overflow_count = 0, 0
    for idx_list in groups.values():
        k = len(idx_list)
        max_dup = max(max_dup, k)
        if k > vocab_size:
            logging.warning(f"一个簇内重复数 {k} > 码本大小 {vocab_size}。去重ID将取模，可能导致碰撞。")
            local_ids = np.arange(k, dtype=np.int64) % vocab_size
            overflow_count += 1
        else:
            local_ids = np.arange(k, dtype=np.int64)
        dedup_layer[np.array(idx_list), 0] = local_ids
    
    logging.info(f"去重层构建完成。最大簇内重复数: {max_dup}。发生取模的簇数量: {overflow_count}。")
    return dedup_layer

def calc_cos_sim(model, data, config):
    if len(data.shape) > 2:
        data = data[:, 0, :]
    ids = model.get_codes(data).cpu().numpy()
    max_item_calculate = 1000
    cos_sim_array = np.zeros(config["num_levels"])

    for n_prefix in range(1, config["num_levels"] + 1):
        unique_prefix = np.unique(ids[:, :n_prefix], axis=0)
        this_level_cos_sim_within_cluster = []

        for this_level_prefix in unique_prefix:
            mask = (ids[:, :n_prefix] == this_level_prefix).all(axis=1)
            this_cluster = data[mask].cpu()
            this_cluster_num = this_cluster.shape[0]

            if this_cluster_num > 1:
                indice = torch.randperm(this_cluster_num)[:max_item_calculate]
                cos_sim = F.cosine_similarity(
                    this_cluster[indice, :, None],
                    this_cluster.t()[None, :, indice]
                )
                cos_sim_sum = torch.tril(cos_sim, diagonal=-1).sum()
                normalization_factor = (this_cluster_num - 1) * this_cluster_num / 2
                this_level_cos_sim_within_cluster.append(
                    cos_sim_sum.item() / normalization_factor
                )

        if this_level_cos_sim_within_cluster:
            cos_sim_array[n_prefix - 1] = np.mean(this_level_cos_sim_within_cluster)

    return cos_sim_array


def process_embeddings(config, device, id2meta_file=None, embedding_save_path=None):
    category = config["dataset"]["name"]
    type = config["dataset"]["type"]
    final_output_path = os.path.join("cache", type, category, "processed", "final_pca_embeddings.npy")

    if not os.path.exists(final_output_path):
        raise FileNotFoundError(f"Embedding file not found: {final_output_path}")

    np_array = np.load(final_output_path)
    tensor = torch.from_numpy(np_array).to(device, dtype=torch.float32)
    print(f"[QUANTIZATION] Loaded embeddings from '{final_output_path}', shape={tensor.shape}, dtype={tensor.dtype}")
    return tensor


def set_weight_decay(optimizer, weight_decay):
    for param_group in optimizer.param_groups:
        param_group["weight_decay"] = weight_decay

def build_codebook_path(codebook_base_path: str, dataset_name: str, model_name: str) -> str:
    """
    生成码本保存/读取的规范路径:
      {codebook_base_path}/{dataset_name}/{dataset_name}.{model_name.lower()}.codebook.npy
    例如:
      ../datasets/Beauty/Beauty.rqvae.codebook.npy
    """
    ds = str(dataset_name)
    model_tag = str(model_name).lower()
    dir_path = os.path.join(codebook_base_path, ds)
    os.makedirs(dir_path, exist_ok=True)
    filename = f"{ds}.{model_tag}.codebook.npy"
    return os.path.join(dir_path, filename)