# /quantization/main.py

import argparse
import yaml
import numpy as np
import torch
import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import utils

def main():
    parser = argparse.ArgumentParser(description="模块化量化器训练脚本")
    parser.add_argument('--quantizer_name', type=str, required=True, choices=['rqvae'], help='要使用的量化器模型名称。')
    parser.add_argument('--dataset_name', type=str, required=True, help='数据集名称 (e.g., Beauty)')
    parser.add_argument('--embedding_suffix', type=str, required=True, help='嵌入文件名后缀 (e.g., td-pca512)')
    parser.add_argument('--config_path', type=str, default='./configs/rqvae_config.yaml', help='配置文件路径')
    parser.add_argument('--data_base_path', type=str, default='../datasets', help='数据集根目录')
    parser.add_argument('--log_base_path', type=str, default='../logs/quantizers', help='日志根目录')
    parser.add_argument('--ckpt_base_path', type=str, default='../ckpt/quantizers', help='模型根目录')
    parser.add_argument('--codebook_base_path', type=str, default='../datasets', help='码本根目录')
    args = parser.parse_args()

    # 1. 设置路径和日志
    embedding_path, log_dir, ckpt_dir, codebook_dir = utils.setup_paths(args)
    utils.setup_logging(log_dir)

    # 2. 加载数据和配置
    logging.info("加载数据和配置...")
    with open(args.config_path, 'r') as f: config = yaml.safe_load(f)
    if not os.path.exists(embedding_path):
        logging.error(f"特征文件未找到: {embedding_path}"); return
    item_embeddings = np.load(embedding_path)
    device = torch.device(config['common'].get('device', 'cuda:0'))
    
    # --- 3. 动态模型分派 (现在非常简洁) ---
    if args.quantizer_name == 'rqvae':
        # 从rqvae模块导入其执行入口
        from rqvae.train import run_pipeline as run_rqvae_pipeline
        # 将任务全权委托给rqvae模块
        run_rqvae_pipeline(item_embeddings, device, config['rqvae'], ckpt_dir, codebook_dir)

    # elif args.quantizer_name == 'vqvae':
    #     from vqvae.train import run_pipeline as run_vqvae_pipeline
    #     run_vqvae_pipeline(...)
    
    else:
        raise ValueError(f"未知的量化器名称: {args.quantizer_name}")

    logging.info("\n--- 所有任务完成 ---")

if __name__ == '__main__':
    main()