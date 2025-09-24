# main.py

import argparse
import logging
import torch
import torch.optim as optim
import os

from models import TIGER
from dataset import GenRecDataset
from dataloader import GenRecDataLoader
from trainer import train_one_epoch, evaluate
from utils import setup_config, setup_logging, set_seed


def main():
    # 1) 参数：只需两个关键参数
    parser = argparse.ArgumentParser(description="TIGER configuration")
    parser.add_argument('--dataset_name', type=str, required=True, help='数据集名称 (例如: Beauty)')
    parser.add_argument('--quant_method', type=str, required=True, choices=['rkmeans', 'rvq', 'rqvae'],
                        help='量化方法 (rkmeans / rvq / rqvae)')

    # 其他训练相关参数保持不变
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--infer_size', type=int, default=96)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_decoder_layers', type=int, default=4)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--d_ff', type=int, default=1024)
    parser.add_argument('--num_heads', type=int, default=6)
    parser.add_argument('--d_kv', type=int, default=64)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--vocab_size', type=int, default=1025)
    parser.add_argument('--feed_forward_proj', type=str, default='relu')
    parser.add_argument('--max_len', type=int, default=20)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluation'])
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--early_stop', type=int, default=10)
    parser.add_argument('--topk_list', type=list, default=[5,10,20])
    parser.add_argument('--beam_size', type=int, default=30)
    args = parser.parse_args()

    # 2) 自动配置（含路径/KL/词表推导/存在性校验）
    config = setup_config(args)

    # 3) 日志与随机种子
    setup_logging(config['log_path'])
    set_seed(config['seed'])
    logging.info(f"Configuration: {config}")

    # 4) 设备
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # 5) 数据集（直接用 utils 里自动拼好的 jsonl 路径）
    train_dataset = GenRecDataset(
        dataset_path=config['train_json'],
        code_path=config['code_path'], mode='train', max_len=config['max_len'],
        vocab_sizes=config['vocab_sizes'], bases=config['bases']
    )
    validation_dataset = GenRecDataset(
        dataset_path=config['valid_json'],
        code_path=config['code_path'], mode='evaluation', max_len=config['max_len'],
        vocab_sizes=config['vocab_sizes'], bases=config['bases']
    )
    test_dataset = GenRecDataset(
        dataset_path=config['test_json'],
        code_path=config['code_path'], mode='evaluation', max_len=config['max_len'],
        vocab_sizes=config['vocab_sizes'], bases=config['bases']
    )
    train_loader = GenRecDataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    validation_loader = GenRecDataLoader(validation_dataset, batch_size=config['infer_size'], shuffle=False)
    test_loader = GenRecDataLoader(test_dataset, batch_size=config['infer_size'], shuffle=False)

    # 6) 模型与优化器
    model = TIGER(config)
    model.model.resize_token_embeddings(config['vocab_size'])
    model.to(device)
    logging.info(model.n_parameters)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    # 7) 训练-评估循环
    best_ndcg = 0.0
    early_stop_counter = 0
    for epoch in range(config['num_epochs']):
        logging.info(f"Epoch {epoch + 1}/{config['num_epochs']}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        logging.info(f"Training loss: {train_loss:.4f}")

        val_recalls, val_ndcgs = evaluate(
            model, validation_loader, config['topk_list'],
            config['beam_size'], config['code_len'], device
        )
        logging.info(f"Validation Recalls: {val_recalls}")
        logging.info(f"Validation NDCGs: {val_ndcgs}")

        current_ndcg = val_ndcgs.get('NDCG@20', 0)
        if current_ndcg > best_ndcg:
            best_ndcg = current_ndcg
            early_stop_counter = 0
            logging.info(f"New best NDCG@20 on validation: {best_ndcg:.4f}")

            test_recalls, test_ndcgs = evaluate(
                model, test_loader, config['topk_list'],
                config['beam_size'], config['code_len'], device
            )
            logging.info(f"Test Recalls: {test_recalls}")
            logging.info(f"Test NDCGs: {test_ndcgs}")

            torch.save(model.state_dict(), config['save_path'])
            logging.info(f"Best model saved to {config['save_path']}")
        else:
            early_stop_counter += 1
            logging.info(f"No improvement in NDCG@20. Early stop counter: {early_stop_counter}/{config['early_stop']}")
            if early_stop_counter >= config['early_stop']:
                logging.info("Early stopping triggered.")
                break


if __name__ == "__main__":
    main()
