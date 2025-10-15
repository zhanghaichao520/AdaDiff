# main.py

import argparse
import logging
import torch
import torch.optim as optim
import os

from dataset import GenRecDataset
from dataloader import GenRecDataLoader
from trainer import train_one_epoch, evaluate
from utils import load_and_process_config, setup_logging, set_seed, get_model_class


def main():
    # 1) 参数：现在只需要模型、数据集和量化方法
    parser = argparse.ArgumentParser(description="GenRec Universal Training Pipeline")
    parser.add_argument('--model', type=str, required=True, help='模型名称 (e.g., TIGER)')
    parser.add_argument('--dataset', type=str, required=True, help='数据集名称 (e.g., Beauty)')
    parser.add_argument('--quant_method', type=str, required=True, choices=['rkmeans', 'rvq', 'rqvae'],
                        help='量化方法 (rkmeans / rvq / rqvae)')
    args = parser.parse_args()

    # 2) 加载并处理所有配置
    config = load_and_process_config(args.model, args.dataset, args.quant_method)

    # 3) 初始化
    setup_logging(config['log_path'])
    set_seed(config['training_params']['seed'])
    logging.info(f"Configuration loaded for {args.model} on {args.dataset} with {args.quant_method}.")

    # 4) 设备
    device = torch.device(config['training_params']['device'] if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

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
    train_loader = GenRecDataLoader(train_dataset, batch_size=config['training_params']['batch_size'], shuffle=True)
    validation_loader = GenRecDataLoader(validation_dataset, batch_size=config['training_params']['infer_size'], shuffle=False)
    test_loader = GenRecDataLoader(test_dataset, batch_size=config['training_params']['infer_size'], shuffle=False)

    # 6) 模型与优化器
    # 6) ✨ 模型與優化器 (動態載入) ✨
    logging.info(f"Dynamically loading model: {args.model}")
    # 透過輔助函數，用字串獲取模型 Class
    ModelClass = get_model_class(args.model)
    # 實例化模型
    model = ModelClass(config)
    
    # 調整詞嵌入層大小 (使用 self.t5)
    model.t5.resize_token_embeddings(config['token_params']['vocab_size'])
    model.to(device)
    logging.info(model.n_parameters)
    optimizer = optim.Adam(model.parameters(), lr=config['training_params']['lr'])


    # 7) 訓練-評估循環
    best_ndcg = 0.0
    early_stop_counter = 0
    
    # --- 新增：用於儲存最佳結果的變數 ---
    best_epoch = 0
    best_val_results = None
    best_test_results = None
    # ------------------------------------

    for epoch in range(config['training_params']['num_epochs']):
        logging.info(f"--- Epoch {epoch + 1}/{config['training_params']['num_epochs']} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        logging.info(f"Training loss: {train_loss:.4f}")

        val_recalls, val_ndcgs = evaluate(
            model, validation_loader, config['evaluation_params']['topk_list'],
            config['evaluation_params']['beam_size'], config['code_len'], device
        )
        logging.info(f"Validation Recalls: {val_recalls}")
        logging.info(f"Validation NDCGs: {val_ndcgs}")

        current_ndcg = val_ndcgs.get('NDCG@20', 0)
        if current_ndcg > best_ndcg:
            best_ndcg = current_ndcg
            early_stop_counter = 0
            logging.info(f"🚀 New best NDCG@20 on validation: {best_ndcg:.4f}")

            test_recalls, test_ndcgs = evaluate(
                model, test_loader, config['evaluation_params']['topk_list'],
                config['evaluation_params']['beam_size'], config['code_len'], device
            )
            logging.info(f"Test Recalls: {test_recalls}")
            logging.info(f"Test NDCGs: {test_ndcgs}")

            # --- 新增：更新最佳結果 ---
            best_epoch = epoch + 1
            best_val_results = {'recalls': val_recalls, 'ndcgs': val_ndcgs}
            best_test_results = {'recalls': test_recalls, 'ndcgs': test_ndcgs}
            # --------------------------

            torch.save(model.state_dict(), config['save_path'])
            logging.info(f"Best model saved to {config['save_path']}")
        else:
            early_stop_counter += 1
            logging.info(f"No improvement in NDCG@20. Early stop counter: {early_stop_counter}/{config['training_params']['early_stop']}")
            if early_stop_counter >= config['training_params']['early_stop']:
                logging.info("Early stopping triggered.")
                break
    
    # --- 新增：在訓練結束後打印最終總結 ---
    logging.info("="*50)
    logging.info("🏁 Training Finished!")
    if best_test_results:
        logging.info(f"🏆 Best performance found at Epoch {best_epoch}")
        logging.info(f"  - Best Validation Results: Recalls={best_val_results['recalls']}, NDCGs={best_val_results['ndcgs']}")
        logging.info(f"  - Corresponding Test Results: Recalls={best_test_results['recalls']}, NDCGs={best_test_results['ndcgs']}")
        logging.info(f"  - Best model checkpoint saved at: {config['save_path']}")
    else:
        logging.info("No improvement was observed during training. No model was saved.")
    logging.info("="*50)
    # ---------------------------------------


if __name__ == "__main__":
    main()
