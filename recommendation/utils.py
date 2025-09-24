# utils.py
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import numpy as np
import random
import torch
import logging

VALID_QUANT = {"rkmeans", "rvq", "rqvae"}

def load_rqvae_yaml(path: str) -> dict:
    """加载并验证 RQ-VAE 的 YAML 配置文件"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"[Config] 未找到 RQ-VAE 配置文件: {path}")
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    if 'rqvae' not in cfg or 'model_params' not in cfg['rqvae']:
        raise ValueError(f"[Config] {path} 缺少 rqvae.model_params 节点")
    mp = cfg['rqvae']['model_params']
    return {
        'codebook_size': int(mp['codebook_size']),
        'num_levels': int(mp['num_levels']),
        'device': cfg.get('common', {}).get('device', None)
    }

def _ensure_parent_dir(file_path: str):
    """确保文件所在目录存在"""
    parent = os.path.dirname(os.path.abspath(file_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

def _auto_paths(dataset_name: str, quant_method: str) -> dict:
    """基于数据集名与量化方法自动拼接路径"""
    name = dataset_name
    name_lower = name.lower()
    q = quant_method.lower()
    if q not in VALID_QUANT:
        raise ValueError(f"[Config] 不支持的量化方法: {q}，可选：{sorted(VALID_QUANT)}")

    # 数据根目录
    dataset_dir = f"../datasets/{name}"

    # codebook 统一命名：<Name>.<quant>.codebook.npy
    code_path = os.path.join(dataset_dir, f"{name}.{q}.codebook.npy")

    # RQ-VAE 的默认配置路径（仅 rqvae 使用）
    rqvae_cfg = "../quantizers/rqvae/configs/rqvae_config.yaml"
    # 兼容你之前用的老路径（若默认不存在则尝试此路径）
    legacy_rqvae_cfg = "/home/wj/peiyu/GenRec/MM-RQVAE/quantization/configs/rqvae_config.yaml"
    if q == "rqvae":
        if not os.path.isfile(rqvae_cfg) and os.path.isfile(legacy_rqvae_cfg):
            rqvae_cfg = legacy_rqvae_cfg

    # 日志/ckpt：包含 quant 方法名，便于区分
    log_path = f"./logs/tiger_{name_lower}_{q}.log"
    save_path = f"./ckpt/tiger_{name_lower}_{q}.pth"

    # 数据文件（train/valid/test）
    train_json = os.path.join(dataset_dir, f"{name}.train.jsonl")
    valid_json = os.path.join(dataset_dir, f"{name}.valid.jsonl")
    test_json  = os.path.join(dataset_dir, f"{name}.test.jsonl")

    return {
        "dataset_path": dataset_dir,
        "code_path": code_path,
        "rqvae_config_path": rqvae_cfg if q == "rqvae" else None,
        "log_path": log_path,
        "save_path": save_path,
        "train_json": train_json,
        "valid_json": valid_json,
        "test_json": test_json,
        "quant_method": q,
    }

def setup_config(args):
    """
    根据 dataset_name + quant_method 自动拼接路径，
    推导词表相关参数，并做文件存在性与格式校验。
    """
    config = vars(args)
    name = config['dataset_name']
    quant = config['quant_method'].lower()

    # === 自动路径 ===
    auto = _auto_paths(name, quant)
    config.update(auto)

    # 创建日志/ckpt目录（若不存在）
    _ensure_parent_dir(config['log_path'])
    _ensure_parent_dir(config['save_path'])

    # === RQ-VAE 配置加载（仅 rqvae）===
    if quant == "rqvae":
        if not config['rqvae_config_path'] or not os.path.isfile(config['rqvae_config_path']):
            raise FileNotFoundError(
                f"[Config] rqvae_config_path 不存在：{config['rqvae_config_path']}\n"
                f"请检查默认路径 ../quantizers/rqvae/configs/rqvae_config.yaml 或 legacy 路径"
            )
        rq = load_rqvae_yaml(config['rqvae_config_path'])
    else:
        # 非 rqvae：给出一个“假结构”只为推导 K/L。按需改为常量。
        rq = {
            'codebook_size': 256,  # 你的 RKMeans/RVQ 的每层 codebook size
            'num_levels': 3,       # 残差层数
            'device': None
        }

    K, L = int(rq['codebook_size']), int(rq['num_levels'])
    if rq['device'] is not None:
        config['device'] = rq['device']

    # === 读取 codebook 校验 ===
    if not os.path.isfile(config['code_path']):
        raise FileNotFoundError(
            f"[FATAL] 未找到 codebook: {config['code_path']} "
            f"(期望命名: {name}.{quant}.codebook.npy，注意大小写与量化方法名)"
        )
    codes_arr = np.load(config['code_path'], allow_pickle=True)
    codes_mat = np.vstack(codes_arr) if codes_arr.dtype == object else codes_arr
    if not np.issubdtype(codes_mat.dtype, np.integer):
        raise ValueError("[FATAL] Codebook 文件应为整数 token（离散代码），实际非整型")
    if codes_mat.ndim != 2 or codes_mat.shape[1] != 4:
        raise ValueError(f"[FATAL] 期望 codebook 形状为 (N, 4)，实际为 {codes_mat.shape}")

    # 基于第4列（dup列）计算去重词表大小
    dup_max = int(codes_mat[:, 3].max()) if codes_mat.size > 0 else 0
    dup_vocab_size = dup_max + 1

    # 词表与基数（前三列为 L 层，每层大小为 K；第4列为 dup digit）
    vocab_sizes = [K] * min(L, 3)  # 你的下游目前用3位语义 + 1位dup；如需兼容L!=3可在模型里改
    if len(vocab_sizes) != 3:
        # 简单兜底：如果 L 不是 3，就取前三层；更严格可直接 raise 并提示修改下游模型 code_len
        vocab_sizes = [K, K, K]
    vocab_sizes.append(dup_vocab_size)

    bases = [0, K, 2 * K, 3 * K]
    max_token_id = sum(vocab_sizes)
    eos_id = max_token_id + 1
    vocab_size = eos_id + 1 + 4  # 预留

    # 自动数据路径（main 可直接用）
    config['train_json'] = auto['train_json']
    config['valid_json'] = auto['valid_json']
    config['test_json']  = auto['test_json']

    config.update({
        'code_len': 4,
        'codebook_size': K,
        'num_levels': L,
        'dup_vocab_size': dup_vocab_size,
        'vocab_sizes': vocab_sizes,
        'bases': bases,
        'pad_token_id': 0,
        'eos_token_id': eos_id,
        'vocab_size': max(vocab_size, int(config.get('vocab_size', 0) or 0))
    })
    return config

def setup_logging(log_path: str):
    """配置日志记录器"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    _ensure_parent_dir(log_path)
    fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)

def set_seed(seed: int):
    """设置全局随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
