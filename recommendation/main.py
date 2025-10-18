# 檔案路徑: recommendation/main.py (最終推薦版)

import argparse
import logging
import torch
import torch.optim as optim
import os
import pprint

# ✅ 從 dataset 導入的是簡化後的 Dataset
from dataset import GenRecDataset, item2code
from dataloader import GenRecDataLoader
from trainer import train_one_epoch, evaluate
# ✅ 從 utils 導入 item2code
from utils import load_and_process_config, setup_logging, set_seed, get_model_class

def main():
    # === 1. 解析命令列參數 ===
    parser = argparse.ArgumentParser(description="GenRec Universal Training Pipeline")
    parser.add_argument('--model', type=str, required=True, help='模型名稱 (e.g., TIGER, RPG)')
    parser.add_argument('--dataset', type=str, required=True, help='数据集名稱 (e.g., Beauty)')
    parser.add_argument('--quant_method', type=str, required=True, choices=['rkmeans', 'rvq', 'rqvae', 'opq', 'pq'],
                        help='量化方法')
    args = parser.parse_args()

    # === 2. 載入並處理設定檔 ===
    config = load_and_process_config(args.model, args.dataset, args.quant_method)

    # === 3. 初始化 (日誌, 隨機種子) ===
    setup_logging(config['log_path'])
    set_seed(config['training_params']['seed'])
    logging.info(f"Configuration loaded for {args.model} on {args.dataset} with {args.quant_method}.")

    # 打印最終生效的設定
    logging.info("=" * 50)
    logging.info("--- Final Configuration ---")
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

    # 打印模型資訊
    logging.info("=" * 50)
    logging.info("--- Model Details ---")
    logging.info(model.n_parameters)
    logging.info("--- Model Architecture ---")
    logging.info(model)
    logging.info("=" * 50)

    optimizer = optim.Adam(model.parameters(), lr=float(config['training_params']['lr']))

    # === 6. ✅ 關鍵改動：載入 item_to_code 映射 ===
    logging.info("Loading item to code mapping...")
    item_to_code_map, _ = item2code(
        config['code_path'],
        config['vocab_sizes'],
        config['bases']
    )
    logging.info(f"Item to code map loaded. Total items mapped: {len(item_to_code_map)}")
    # 打印一個範例，確保載入正確
    example_item_id = next(iter(item_to_code_map.keys()), None)
    if example_item_id is not None:
         logging.info(f"Example mapping for item {example_item_id}: {item_to_code_map[example_item_id]}")
    else:
         logging.warning("Item to code map appears to be empty!")


    # === 7. 創建數據集與 DataLoader ===
    # 使用簡化後的 Dataset 初始化
    train_dataset = GenRecDataset(config=config, mode='train')
    validation_dataset = GenRecDataset(config=config, mode='valid')
    test_dataset = GenRecDataset(config=config, mode='test')

    pad_token_id = config['token_params']['pad_token_id']
    code_len = config['code_len']

    # DataLoader 現在接收 item_to_code_map 和 code_len
    train_loader = GenRecDataLoader(
        train_dataset,
        model=model,
        item_to_code_map=item_to_code_map,
        batch_size=config['training_params']['batch_size'],
        shuffle=True, num_workers=num_workers,
        pad_token_id=pad_token_id, code_len=code_len
    )
    validation_loader = GenRecDataLoader(
        validation_dataset,
        model=model,
        item_to_code_map=item_to_code_map,
        batch_size=config['evaluation_params']['batch_size'],
        shuffle=False, num_workers=num_workers,
        pad_token_id=pad_token_id, code_len=code_len
    )
    test_loader = GenRecDataLoader(
        test_dataset,
        model=model,
        item_to_code_map=item_to_code_map,
        batch_size=config['evaluation_params']['batch_size'],
        shuffle=False, num_workers=num_workers,
        pad_token_id=pad_token_id, code_len=code_len
    )

    # === 8. 訓練-評估循環 ===
    best_ndcg = 0.0
    early_stop_counter = 0
    best_epoch = 0
    best_val_results = None
    best_test_results = None

    for epoch in range(config['training_params']['num_epochs']):
        logging.info(f"--- Epoch {epoch + 1}/{config['training_params']['num_epochs']} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        logging.info(f"Training loss: {train_loss:.4f}")

        # 評估邏輯保持不變 (假設 evaluate 返回單一字典)
        val_results = evaluate(
            model,
            validation_loader,
            config['evaluation_params']['topk_list'],
            device
        )
        logging.info(f"Validation Results: {val_results}")

        # 使用 NDCG@20 作為早停指標
        current_ndcg = val_results.get('NDCG@20', 0.0)

        if current_ndcg > best_ndcg:
            best_ndcg = current_ndcg
            early_stop_counter = 0
            logging.info(f"🚀 New best NDCG@20 on validation: {best_ndcg:.4f}")

            test_results = evaluate(
                model,
                test_loader,
                config['evaluation_params']['topk_list'],
                device
            )
            logging.info(f"Test Results: {test_results}")

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

    # === 9. 訓練結束總結 ===
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


if __name__ == "__main__":
    main()