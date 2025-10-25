import argparse
import logging
import torch
import torch.optim as optim
import os
import pprint

# ✅ 1. 從 torch.utils.data 直接導入 DataLoader
from torch.utils.data import DataLoader 
from dataset import GenRecDataset, item2code
# from dataloader import GenRecDataLoader  # <-- 已刪除
from tokenizer import get_tokenizer       
from trainer import train_one_epoch, evaluate
from utils import load_and_process_config, setup_logging, set_seed, get_model_class

def main():
    # === 1. 解析命令列參數 ===
    parser = argparse.ArgumentParser(description="GenRec Universal Training Pipeline")
    parser.add_argument('--model', type=str, required=True, help='模型名稱 (e.g., TIGER, GPT2, RPG)')
    parser.add_argument('--dataset', type=str, required=True, help='数据集名稱 (e.g., Beauty)')
    parser.add_argument('--quant_method', type=str, required=True, choices=['rkmeans', 'rvq', 'rqvae', 'opq', 'pq', 'vqvae'],
                        help='量化方法')
    args = parser.parse_args()

    # === 2. 載入並處理設定檔 ===
    config = load_and_process_config(args.model, args.dataset, args.quant_method)

    # === 3. 初始化 (日誌, 隨機種子) ===
    setup_logging(config['log_path'])
    set_seed(config['training_params']['seed'])
    logging.info(f"Configuration loaded for {args.model} on {args.dataset} with {args.quant_method}.")
    logging.info("=" * 50)
    config_str = pprint.pformat(config)
    logging.info("\n" + config_str)
    logging.info("=" * 50)

    # === 4. 設定設備 ===
    device = torch.device(config['training_params']['device'] if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    num_workers = config['training_params'].get('num_workers', 4)

    # === 5. 創建模型與優化器 ===
    logging.info(f"Dynamically loading model: {args.model}")
    ModelClass = get_model_class(args.model)
    model = ModelClass(config)
    model.to(device)
    logging.info(model.n_parameters)
    logging.info("=" * 50)
    optimizer = optim.Adam(model.parameters(), lr=float(config['training_params']['lr']))

    # === 6. 載入 item_to_code 映射 ===
    logging.info("Loading item to code mapping...")
    item_to_code_map, _ = item2code(
        config['code_path'],
        config['vocab_sizes'],
        config['bases']
    )
    logging.info(f"Item to code map loaded. Total items mapped: {len(item_to_code_map)}")

    # === 7. 初始化模型專屬的 Tokenizer ===
    logging.info(f"Initializing tokenizer for model: {args.model}")
    tokenizer_collate_fn = get_tokenizer(
        model_name=args.model,
        config=config,
        item_to_code_map=item_to_code_map
    )
    logging.info("Tokenizer initialized.")

    # === 8. 創建數據集與 DataLoader ===
    logging.info("Creating Datasets...")
    train_dataset = GenRecDataset(config=config, mode='train')
    validation_dataset = GenRecDataset(config=config, mode='valid')
    test_dataset = GenRecDataset(config=config, mode='test')

    logging.info("Creating DataLoaders...")
    
    # ✅ 2. 準備通用的 DataLoader 參數
    is_gpu_training = (torch.cuda.is_available() and num_workers > 0)
    loader_kwargs = {
        "num_workers": num_workers,
        "collate_fn": tokenizer_collate_fn, # 傳入 tokenizer
        "pin_memory": is_gpu_training,
        "persistent_workers": is_gpu_training if num_workers > 0 else False
    }

    # ✅ 3. 直接使用 PyTorch 官方的 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training_params']['batch_size'],
        shuffle=True, 
        **loader_kwargs
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config['evaluation_params']['batch_size'],
        shuffle=False, 
        **loader_kwargs
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['evaluation_params']['batch_size'],
        shuffle=False, 
        **loader_kwargs
    )

    # === 9. 训练-评估循环 (已修改) ===
    best_ndcg = 0.0
    early_stop_counter = 0
    best_epoch = 0
    best_val_results = None
    best_test_results = None
    
    # 从配置中获取评估间隔
    eval_interval = config['training_params'].get('eval_interval', 1) # 默认为 1 (慢速模式)
    logging.info(f"Evaluation interval set to: {eval_interval} epoch(s)")

    for epoch in range(config['training_params']['num_epochs']):
        epoch_num = epoch + 1 # 当前 epoch 编号 (从 1 开始)
        logging.info(f"--- Epoch {epoch_num}/{config['training_params']['num_epochs']} ---")
        
        # --- 训练 ---
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        logging.info(f"Training loss: {train_loss:.4f}")

        # --- 评估 (根据 eval_interval) ---
        # 检查是否到达评估的 epoch
        if epoch_num % eval_interval == 0:
            logging.info(f"--- Evaluating at Epoch {epoch_num} ---")
            val_results = evaluate(
                model,
                validation_loader,
                config['evaluation_params']['topk_list'],
                device
            )
            logging.info(f"Validation Results: {val_results}")

            current_ndcg = val_results.get('NDCG@10', val_results.get('NDCG@20', 0.0))

            # --- 检查性能提升和 Early Stopping ---
            if current_ndcg > best_ndcg:
                best_ndcg = current_ndcg
                early_stop_counter = 0 # 重置计数器
                logging.info(f"🚀 New best NDCG on validation: {best_ndcg:.4f} at Epoch {epoch_num}")

                # --- 只有在验证集性能提升时，才评估测试集 ---
                test_results = evaluate(
                    model,
                    test_loader,
                    config['evaluation_params']['topk_list'],
                    device
                )
                logging.info(f"Test Results: {test_results}")

                # 更新最佳结果记录
                best_epoch = epoch_num
                best_val_results = val_results
                best_test_results = test_results

                # 保存最佳模型
                torch.save(model.state_dict(), config['save_path'])
                logging.info(f"Best model saved to {config['save_path']}")
            
            else:
                # 验证集性能没有提升
                early_stop_counter += eval_interval # <--- 注意：每次检查时增加 interval 的值
                logging.info(f"No improvement since Epoch {best_epoch}. Early stop counter: {early_stop_counter}/{config['training_params']['early_stop'] * eval_interval}")
                # <--- 修改 Early Stopping 条件：当累计未提升的 epoch 数（考虑了 interval）超过阈值时停止
                if early_stop_counter >= config['training_params']['early_stop'] * eval_interval:
                    logging.info("Early stopping triggered.")
                    break
        else:
             # 如果不是评估 epoch，只打印训练损失信息
             logging.info(f"Skipping evaluation for Epoch {epoch_num}.")

    # === 10. 訓練結束總結 ===
    logging.info("="*50)
    logging.info("🏁 Training Finished!")
    if best_test_results:
        logging.info(f"🏆 Best performance found at Epoch {best_epoch}")
        logging.info(f"  - Best Validation Results: {best_val_results}")
        logging.info(f"  - Corresponding Test Results: {best_test_results}")
    else:
        logging.info("No improvement was observed.")
    logging.info("="*50)


if __name__ == "__main__":
    main()