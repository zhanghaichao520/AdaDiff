# utils.py
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import numpy as np
import random
import torch
import logging
from pathlib import Path
import importlib

VALID_QUANT_METHODS = {"rkmeans", "rvq", "rqvae"}

def _ensure_dir_exists(dir_path: Path):
    """确保目录存在"""
    if dir_path:
        dir_path.mkdir(parents=True, exist_ok=True)

def _load_rqvae_details(path: str) -> dict:
    """从 RQ-VAE 配置文件中加载必要的参数"""
    if not Path(path).is_file():
        raise FileNotFoundError(f"[Config] 未找到 RQ-VAE 配置文件: {path}")
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    if 'rqvae' not in cfg or 'model_params' not in cfg['rqvae']:
        raise ValueError(f"[Config] {path} 缺少 rqvae.model_params 节点")
    mp = cfg['rqvae']['model_params']
    return {
        'codebook_size': int(mp['codebook_size']),
        'num_levels': int(mp['num_levels']),
    }

def load_and_process_config(model_name: str, dataset_name: str, quant_method: str) -> dict:
    """
    通用配置加载器。
    1. 加载模型的基础 YAML 文件 (e.g., configs/TIGER.yaml)。
    2. 使用命令行参数 (dataset, quant_method) 填充路径模板。
    3. 推导词表大小和相关参数。
    4. 校验文件和路径。
    """
    # === 1. 加载模型的基础 YAML 文件 ===
    config_path = Path(f"configs/{model_name}.yaml")
    if not config_path.is_file():
        raise FileNotFoundError(f"模型配置文件未找到: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['model_name'] = model_name
    config['dataset_name'] = dataset_name
    config['quant_method'] = quant_method.lower()
    
    if config['quant_method'] not in VALID_QUANT_METHODS:
        raise ValueError(f"不支持的量化方法: {quant_method}。可选: {VALID_QUANT_METHODS}")

    # === 2. 格式化和派生路径 ===
    paths = config['paths']
    format_args = {'dataset_name': dataset_name, 'quant_method': config['quant_method']}
    
    dataset_root = Path(paths['dataset_root'].format(**format_args))
    output_root = Path(paths['output_root'].format(**format_args))

    config['code_path'] = paths['codebook_template'].format(dataset_root=dataset_root, **format_args)
    config['log_path'] = output_root / "training.log"
    config['save_path'] = output_root / "best_model.pth"
    
    config['train_json'] = dataset_root / f"{dataset_name}.train.jsonl"
    config['valid_json'] = dataset_root / f"{dataset_name}.valid.jsonl"
    config['test_json'] = dataset_root / f"{dataset_name}.test.jsonl"

    _ensure_dir_exists(output_root)

    # === 3. 加载量化参数并推导词表 ===
    if config['quant_method'] == "rqvae":
        rq_details = _load_rqvae_details(paths['rqvae_config_path'])
    else:
        # 从模型配置中获取非 RQ-VAE 方法的默认值
        rq_details = config['quantization_defaults']
        
    K, L = int(rq_details['codebook_size']), int(rq_details['num_levels'])
    config['codebook_size'] = K
    config['num_levels'] = L

    # === 4. 读取 codebook 并进行校验 ===
    if not Path(config['code_path']).is_file():
        raise FileNotFoundError(f"[FATAL] 未找到 codebook: {config['code_path']}")
        
    codes_arr = np.load(config['code_path'], allow_pickle=True)
    codes_mat = np.vstack(codes_arr) if codes_arr.dtype == object else codes_arr
    
    if not np.issubdtype(codes_mat.dtype, np.integer):
        raise ValueError("[FATAL] Codebook 文件应为整数 token，实际为非整型")
    if codes_mat.ndim != 2 or codes_mat.shape[1] != 4: # 假设 code 长度固定为 4
        raise ValueError(f"[FATAL] 期望 codebook 形状为 (N, 4)，实际为 {codes_mat.shape}")

    # === 5. 计算最终词表参数 ===
    # 基于第4列（dup列）计算去重词表大小
    dup_max = int(codes_mat[:, 3].max()) if codes_mat.size > 0 else 0
    dup_vocab_size = dup_max + 1
    config['dup_vocab_size'] = dup_vocab_size

    # 词表与基数 (前3列为语义层, 第4列为dup层)
    config['code_len'] = 4
    config['vocab_sizes'] = [K, K, K, dup_vocab_size]
    config['bases'] = [0, K, 2 * K, 3 * K]

    max_token_id = sum(config['vocab_sizes'])
    eos_id = max_token_id + 1
    
    # 将最终的词表相关 ID 添加到 config 中
    config['token_params'] = {
        'pad_token_id': 0,
        'eos_token_id': eos_id,
        # 总词表大小 = 最大 token ID + EOS + PAD + 一些预留位
        'vocab_size': eos_id + 5 
    }
    
    # 打印最终配置以供调试
    # import json
    # print(json.dumps(config, indent=2, default=str))

    return config

# --- 日志和随机种子函数保持不变 ---
def setup_logging(log_path: Path):
    """配置日志记录器"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(str(log_path), mode='a', encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)

def get_model_class(model_name: str):
    """
    增强版：根据模型名称字符串，动态地从 'models' 目录及其所有子目录中
    搜索并导入对应的模型类。

    Args:
        model_name (str): 模型的名称 (e.g., "TIGER")。
                          约定：文件名和类名都应与 model_name 一致 (TIGER.py, class TIGER)。
    """
    models_root_dir = "models"
    model_file_name = f"{model_name}.py"
    model_module_path = None

    # os.walk 会遍历指定目录下的所有文件夹和文件
    for root, dirs, files in os.walk(models_root_dir):
        if model_file_name in files:
            # 找到了文件！现在构建 Python 的导入路径
            # 例如, root = "models/encoder_decoder"
            
            # 1. 将文件系统路径 ('/') 替换为 Python 导入路径 ('.')
            # "models/encoder_decoder" -> "models.encoder_decoder"
            base_path = root.replace(os.sep, '.')
            
            # 2. 拼接成最终的模块路径
            # "models.encoder_decoder.TIGER"
            model_module_path = f"{base_path}.{model_name}"
            break # 找到后立刻停止搜索

    # 如果遍历完都没找到文件
    if not model_module_path:
        raise ImportError(
            f"错误：无法在 '{models_root_dir}' 目录或其任何子目录中找到模型文件 '{model_file_name}'。\n"
            f"请检查你的文件结构和 --model 参数是否正确。"
        )

    try:
        # 使用动态构建的路径来导入模块
        logging.info(f"Found model module at: {model_module_path}")
        model_module = importlib.import_module(model_module_path)
        # 从模块中获取与模型同名的类
        model_class = getattr(model_module, model_name)
        return model_class
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"错误：成功找到文件，但在从 '{model_module_path}' 导入类 '{model_name}' 时失败。\n"
            f"请确保你的 Python 文件内 class 的名称 ({model_name}) 与文件名和 --model 参数完全一致。\n"
            f"原始错误: {e}"
        )

def set_seed(seed: int):
    """设置全局随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)