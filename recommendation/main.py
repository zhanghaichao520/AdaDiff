# main.py

import argparse
import logging
import torch
import torch.optim as optim
import os
import pprint

from dataset import GenRecDataset
from dataloader import GenRecDataLoader
from trainer import train_one_epoch, evaluate
from utils import load_and_process_config, setup_logging, set_seed, get_model_class


def main():
    # 1) 参数：现在只需要模型、数据集和量化方法
    parser = argparse.ArgumentParser(description="GenRec Universal Training Pipeline")
    parser.add_argument('--model', type=str, required=True, help='模型名称 (e.g., TIGER)')
    parser.add_argument('--dataset', type=str, required=True, help='数据集名称 (e.g., Beauty)')
    parser.add_argument('--quant_method', type=str, required=True, choices=['rkmeans', 'rvq', 'rqvae', 'opq', 'pq'],
                        help='量化方法 (rkmeans / rvq / rqvae)')
    args = parser.parse_args()

    # 2) 加载并处理所有配置
    config = load_and_process_config(args.model, args.dataset, args.quant_method)

    # 3) 初始化
    setup_logging(config['log_path'])
    set_seed(config['training_params']['seed'])
    logging.info(f"Configuration loaded for {args.model} on {args.dataset} with {args.quant_method}.")

    logging.info("=" * 50)
    logging.info("--- Final Configuration ---")
    # 使用 pprint.pformat 將字典格式化為一個易讀的字串
    config_str = pprint.pformat(config)
    logging.info("\n" + config_str)
    logging.info("=" * 50)

    # 4) 设备
    device = torch.device(config['training_params']['device'] if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    num_workers = config['training_params'].get('num_workers', 4)

    # 6) 模型与优化器
    # 6) ✨ 模型與優化器 (動態載入) ✨
    logging.info(f"Dynamically loading model: {args.model}")
    # 透過輔助函數，用字串獲取模型 Class
    ModelClass = get_model_class(args.model)
    # 實例化模型
    model = ModelClass(config)
    
    # 調整詞嵌入層大小 (使用 self.t5)
    model.to(device)

        # 5) 数据集（直接用 utils 里自动拼好的 jsonl 路径）
    train_dataset = GenRecDataset(
        dataset_path=config['train_json'],
        code_path=config['code_path'], mode='train', max_len=config['model_params']['max_len'],
        vocab_sizes=config['vocab_sizes'], bases=config['bases']
    )
    validation_dataset = GenRecDataset(
        dataset_path=config['valid_json'],
        code_path=config['code_path'], mode='evaluation', max_len=config['model_params']['max_len'],
        vocab_sizes=config['vocab_sizes'], bases=config['bases']
    )
    test_dataset = GenRecDataset(
        dataset_path=config['test_json'],
        code_path=config['code_path'], mode='evaluation', max_len=config['model_params']['max_len'],
        vocab_sizes=config['vocab_sizes'], bases=config['bases']
    )
    train_loader = GenRecDataLoader(train_dataset, model=model, batch_size=config['training_params']['batch_size'], shuffle=True, num_workers=num_workers)
    validation_loader = GenRecDataLoader(validation_dataset, model=model, batch_size=config['evaluation_params']['batch_size'], shuffle=False, num_workers=num_workers)
    test_loader = GenRecDataLoader(test_dataset, model=model, batch_size=config['evaluation_params']['batch_size'], shuffle=False, num_workers=num_workers)

    
    # ✨ 3. 新增區塊：打印模型參數數量和詳細架構 ✨
    logging.info("=" * 50)
    logging.info("--- Model Details ---")
    logging.info(model.n_parameters) # 打印參數數量
    logging.info("--- Model Architecture ---")
    logging.info(model) # 打印模型架構
    logging.info("=" * 50)

    optimizer = optim.Adam(model.parameters(), lr=config['training_params']['lr'])


    # 7) 訓練-評估循環
    best_ndcg = 0.0
    early_stop_counter = 0
    best_epoch = 0
    best_val_results = None
    best_test_results = None

    for epoch in range(config['training_params']['num_epochs']):
        logging.info(f"--- Epoch {epoch + 1}/{config['training_params']['num_epochs']} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        logging.info(f"Training loss: {train_loss:.4f}")

        # ✨ 改动 1: evaluate 现在返回一个包含所有指标的字典 ✨
        val_results = evaluate(
            model, 
            validation_loader, 
            config['evaluation_params']['topk_list'], 
            device
        )
        # 统一打印所有验证集结果
        logging.info(f"Validation Results: {val_results}")

        # ✨ 改动 2: 从结果字典中获取 NDCG@20 ✨
        current_ndcg = val_results.get('NDCG@20', 0.0)
        
        if current_ndcg > best_ndcg:
            best_ndcg = current_ndcg
            early_stop_counter = 0
            logging.info(f"🚀 New best NDCG@20 on validation: {best_ndcg:.4f}")

            # ✨ 改动 3: 对测试集也同样处理 ✨
            test_results = evaluate(
                model, 
                test_loader, 
                config['evaluation_params']['topk_list'], 
                device
            )
            logging.info(f"Test Results: {test_results}")

            # 更新最佳结果
            best_epoch = epoch + 1
            best_val_results = val_results
            best_test_results = test_results

            torch.save(model.state_dict(), config['save_path'])
            logging.info(f"Best model saved to {config['save_path']}")
        else:
            early_stop_counter += 1
            logging.info(f"No improvement in NDCG@20. Early stop counter: {early_stop_counter}/{config['training_params']['early_stop']}")
            if early_stop_counter >= config['training_params']['early_stop']:
                logging.info("Early stopping triggered.")
                break
    
    # ✨ 改动 4: 更新最终的打印逻辑 ✨
    logging.info("="*50)
    logging.info("🏁 Training Finished!")
    if best_test_results:
        logging.info(f"🏆 Best performance found at Epoch {best_epoch}")
        logging.info(f"  - Best Validation Results: {best_val_results}")
        logging.info(f"  - Corresponding Test Results: {best_test_results}")
        logging.info(f"  - Best model checkpoint saved at: {config['save_path']}")
    else:
        logging.info("No improvement was observed during training. No model was saved.")
    logging.info("="*50)
    # ---------------------------------------


if __name__ == "__main__":
    main()
