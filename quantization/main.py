# /quantization/main.py (最终修正版 - 带配置自动路由)

import argparse
import yaml
import numpy as np
import torch
import logging
import os
import sys
import importlib

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import utils
from trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description="通用量化器训练脚本")
    
    parser.add_argument('--model_name', type=str, required=True, choices=['rqvae','vqvae', 'rkmeans','opq'], help='要使用的量化器模型名称。')
    parser.add_argument('--dataset_name', type=str, required=True, help='数据集名称 (e.g., Baby)')
    parser.add_argument('--embedding_modality', type=str, default='text', help='模态类型 (text / multi)')
    parser.add_argument('--embedding_model', type=str, required=True, help='嵌入来源模型名称 (e.g., sentence-t5-base 或 text-embedding-3-large)')
    parser.add_argument('--config_path', type=str, default=None, help='配置文件路径。如果未提供，将根据模型名称自动查找。')
    parser.add_argument('--data_base_path', type=str, default='../datasets', help='数据集根目录')
    parser.add_argument('--log_base_path', type=str, default='../logs/quantization', help='日志根目录')
    parser.add_argument('--ckpt_base_path', type=str, default='../ckpt/quantization', help='模型根目录')
    parser.add_argument('--codebook_base_path', type=str, default='../datasets', help='码本根目录')
    
    args = parser.parse_args()

    # 1. 设置路径和日志
    embedding_path, log_dir, ckpt_dir, codebook_dir = utils.setup_paths(args)
    utils.setup_logging(log_dir)

    # 2. 加载数据和配置
    logging.info(f"开始任务: model={args.model_name}, dataset={args.dataset_name}")

    # --- 核心改动 2：添加自动路由逻辑 ---
    if args.config_path is None:
        # 如果用户未指定 config_path，则根据模型名称自动构建路径
        config_path = f"./configs/{args.model_name}_config.yaml"
        logging.info(f"未指定配置文件路径，将自动使用: {config_path}")
    else:
        # 如果用户指定了，则使用用户提供的路径
        config_path = args.config_path
        logging.info(f"使用指定的配置文件路径: {config_path}")
    
    if not os.path.exists(config_path):
        logging.error(f"配置文件未找到: {config_path}，程序终止。")
        return
        
    logging.info(f"加载配置文件: {config_path}")
    with open(config_path, 'r') as f: config = yaml.safe_load(f)
    # --- 改动结束 ---
    
    config['model_name'] = args.model_name
    config['dataset_name'] = args.dataset_name
    
    logging.info(f"加载特征文件: {embedding_path}")
    if not os.path.exists(embedding_path):
        logging.error(f"特征文件未找到，程序终止。"); return
    item_embeddings = np.load(embedding_path)
    logging.info(f"特征加载完成, 维度: {item_embeddings.shape}")
    config['total_item_count'] = len(item_embeddings)
    
    device = torch.device(config['common'].get('device', 'cuda:0'))
    logging.info(f"使用设备: {device}")
    
    # 动态加载模型
    logging.info(f"正在从 utils.py 动态加载模型: {args.model_name}...")
    ModelClass = utils.get_model(args.model_name)
    logging.info(f"成功加载模型类: {ModelClass.__name__}")
    
    # 初始化并执行流程
    model = ModelClass(config=config, input_size=item_embeddings.shape[1]).to(device)
    trainer = Trainer(config=config, model=model, device=device)
    best_model_path = trainer.fit(embeddings_path=embedding_path, ckpt_dir=ckpt_dir)
    
    if os.path.exists(best_model_path):
        logging.info(f"加载最佳模型进行码本生成: {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        logging.warning("未找到最佳模型，将使用训练结束时的模型状态。")
    
    trainer.predict(embeddings_path=embedding_path, codebook_dir=codebook_dir)
    
    logging.info("\n--- 所有任务完成 ---")

if __name__ == '__main__':
    main()