# 文件路径: instruction_utils.py
import yaml
import logging
import os
from pathlib import Path
from collections.abc import Mapping
import argparse # 用于后续合并命令行参数
import sys

logger = logging.getLogger(__name__)

def _ensure_dir_exists(dir_path: Path):
    """确保目录存在"""
    if dir_path:
        dir_path.mkdir(parents=True, exist_ok=True)

def _recursive_update(base_dict: dict, new_dict: dict) -> dict:
    """递归更新字典 (来自你的 utils.py)"""
    for key, value in new_dict.items():
        if isinstance(value, Mapping) and key in base_dict and isinstance(base_dict[key], Mapping):
            base_dict[key] = _recursive_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict

def load_instruction_config(dataset_name: str, model_name_or_path: str, args: argparse.Namespace) -> dict:
    """
    加载指令微调的基础配置，并根据参数进行处理。
    ✅ 支持直接传入模型路径/Hub ID。
    """
    # 1. 加载基础配置 (不变)
    base_config_path = Path("configs/base.yaml")
    if not base_config_path.is_file():
        raise FileNotFoundError(f"基础配置文件未找到: {base_config_path}")
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. 存储核心参数 (不变)
    config['dataset_name'] = dataset_name
    # config['base_model_alias'] # 我们稍后决定这个

    # 3. ✅ 智能解析模型路径和别名
    defined_models = config.get('models', {})
    
    if model_name_or_path in defined_models:
        # 情况 1: 用户传入的是预定义的别名
        config['base_model_alias'] = model_name_or_path
        config['base_model_path'] = defined_models[model_name_or_path]
        logger.info(f"使用预定义模型别名 '{model_name_or_path}' -> 路径: {config['base_model_path']}")
    else:
        # 情况 2: 用户传入的是路径或 Hub ID
        config['base_model_path'] = model_name_or_path
        # 为输出路径生成一个别名 (例如取路径最后一部分，并替换 '/')
        derived_alias = Path(model_name_or_path).name.replace('.', '_') # 避免路径中有 '.' 导致问题
        config['base_model_alias'] = derived_alias
        logger.info(f"直接使用模型路径/ID: {model_name_or_path}")
        logger.info(f"  > 推断用于输出路径的别名: {derived_alias}")
        
    # 4. 解析 Tokenizer 路径 (逻辑不变，使用 base_model_path)
    if config.get('token_params', {}).get('tokenizer_path') is None:
        config['token_params']['tokenizer_path'] = config['base_model_path']
        logger.info(f"Tokenizer 路径未指定，将使用模型路径: {config['base_model_path']}")
    else:
        logger.info(f"使用指定的 Tokenizer 路径: {config['token_params']['tokenizer_path']}")

    # 5. 构建数据和输出路径 (使用 base_model_alias)
    paths = config['paths']
    format_args = {
        'dataset_name': dataset_name,
        'base_model_alias': config['base_model_alias'] # 使用确定好的别名
    }
    dataset_root = Path(paths['dataset_root'].format(**format_args))
    output_root = Path(paths['output_root'].format(**format_args))

    # --- 数据文件路径推断 (保持不变或根据需要修改) ---
    # 你需要确定如何从命令行或配置获取 quant_method 和 k
    quant_method = "rqvae" # 示例: 需要改为从 args 或 config 获取
    k = 10              # 示例: 需要改为从 args 或 config 获取
    data_suffix = f"{quant_method}.train.top{k}.jsonl" 

    config['train_jsonl'] = dataset_root / "prompts_topk" / f"{dataset_name}.{data_suffix.replace('train', 'train')}"
    config['valid_jsonl'] = dataset_root / "prompts_topk" / f"{dataset_name}.{data_suffix.replace('train', 'valid')}"
    # --- ---

    config['output_dir'] = output_root / "peft_adapter"
    config['log_path'] = output_root / "training.log"
    _ensure_dir_exists(output_root)
    _ensure_dir_exists(config['output_dir'])

    # 6. 合并命令行参数 (逻辑不变)
    logger.info("合并命令行参数以覆盖配置...")
    config_update = {}
    args_dict = vars(args)
    # ... (set_nested 和 arg_to_config_map 逻辑不变) ...
    def set_nested(d, keys, value): # 确保定义了 set_nested
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value
        
    arg_to_config_map = { # 确保定义了映射
        'max_seq_len': ['token_params', 'max_seq_len'],
        'lora_r': ['peft_params', 'lora_r'],
        'lora_alpha': ['peft_params', 'lora_alpha'],
        'lora_dropout': ['peft_params', 'lora_dropout'],
        'epochs': ['training_params', 'epochs'],
        'batch_size': ['training_params', 'batch_size'],
        'eval_batch_size': ['training_params', 'eval_batch_size'],
        'lr': ['training_params', 'lr'],
        'warmup_ratio': ['training_params', 'warmup_ratio'],
        'weight_decay': ['training_params', 'weight_decay'],
        'gradient_accumulation_steps': ['training_params', 'gradient_accumulation_steps'],
        'device': ['device']
    }
    for arg_key, config_keys in arg_to_config_map.items():
        if arg_key in args_dict and args_dict[arg_key] is not None:
             set_nested(config, config_keys, args_dict[arg_key])
             logger.info(f"  > 使用命令行参数覆盖 '{'.'.join(config_keys)}': {args_dict[arg_key]}")

    return config

def setup_instruction_logging(log_path: Path):
    """配置日志记录器 (类似你的 utils.py)"""
    logger_root = logging.getLogger() # 获取 root logger
    logger_root.setLevel(logging.INFO)
    if logger_root.hasHandlers():
        logger_root.handlers.clear() # 清除已有处理器
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(message)s') # 添加 logger 名称

    # 控制台输出
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger_root.addHandler(sh)

    # 文件输出
    fh = logging.FileHandler(str(log_path), mode='a', encoding='utf-8')
    fh.setFormatter(fmt)
    logger_root.addHandler(fh)