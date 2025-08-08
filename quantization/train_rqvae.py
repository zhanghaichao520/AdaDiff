import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import yaml
import argparse
import logging
from datetime import datetime

from rqvae.rqvae import RQVAE

# --- 核心训练函数 (已修复) ---
def train_epoch(model, dataloader, optimizer, beta, device, flag_eval=False):
    model.eval() if flag_eval else model.train()
    total_loss, total_rec_loss, total_commit_loss = 0.0, 0.0, 0.0
    for batch in dataloader:
        x_batch = batch[0]
        # 修复：将输入数据移动到传入的 device 上
        x_batch = x_batch.to(device)
        
        if not flag_eval: optimizer.zero_grad()
        with torch.set_grad_enabled(not flag_eval):
            recon_x, commitment_loss, _ = model(x_batch)
            reconstruction_mse_loss = F.mse_loss(recon_x, x_batch, reduction="mean")
            loss = reconstruction_mse_loss + beta * commitment_loss
        if not flag_eval:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item()
        total_rec_loss += reconstruction_mse_loss.item()
        total_commit_loss += commitment_loss.item()
    return total_loss / len(dataloader), total_rec_loss / len(dataloader), total_commit_loss / len(dataloader)

def train_rqvae(model, embeddings, device, config, output_dir):
    model.to(device)
    batch_size, num_epochs, lr, weight_decay = config["batch_size"], config["epochs"], config["lr"], config.get("weight_decay", 0.0)
    optimizer = getattr(torch.optim, config["optimizer"])(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_data, val_data = train_test_split(embeddings, test_size=0.05, random_state=42)
    train_loader = DataLoader(TensorDataset(torch.Tensor(train_data)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.Tensor(val_data)), batch_size=batch_size, shuffle=False)
    best_val_loss = float('inf')
    for epoch in tqdm(range(num_epochs), desc="Training RQ-VAE"):
        # 修复：在调用 train_epoch 时传入 device
        train_loss, _, _ = train_epoch(model, train_loader, optimizer, config["beta"], device)
        if (epoch + 1) % 100 == 0:
            # 修复：在调用 train_epoch 时传入 device
            val_loss, val_rec_loss, val_commit_loss = train_epoch(model, val_loader, None, config["beta"], device, flag_eval=True)
            logging.info(f"[VALIDATION] Epoch {epoch+1:04d} | Val Loss: {val_loss:.4f} | Recon Loss: {val_rec_loss:.4f} | Commit Loss: {val_commit_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(output_dir, "rqvae_best_model.pth")
                torch.save(model.state_dict(), save_path)
                logging.info(f"[MODEL] Best model saved to: {save_path}")
    print("[TRAINING] Training complete.")

def generate_codebook(model, embeddings, device, config, output_dir):
    logging.info("[CODEBOOK] Generating Codebook from all items...")
    model.to(device)
    model.eval()
    all_codes_list = []
    dataloader = DataLoader(TensorDataset(torch.Tensor(embeddings)), batch_size=config["batch_size"], shuffle=False)
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating Codes"):
            codes = model.get_codes(batch[0].to(device)).cpu().numpy()
            all_codes_list.append(codes)
    all_codes_np = np.vstack(all_codes_list)
    item_to_codes = {str(item_id): codes.tolist() for item_id, codes in enumerate(all_codes_np)}
    codebook_path = os.path.join(output_dir, "codebook.json")
    with open(codebook_path, 'w') as f: json.dump(item_to_codes, f)
    logging.info(f"[CODEBOOK] Codebook successfully saved to: {codebook_path}")
    return codebook_path

# --- 主程序入口 ---

def main():
    # --- 1. 解析命令行参数 (MODIFIED) ---
    parser = argparse.ArgumentParser(description="从融合后的 .npy 文件训练 RQ-VAE 模型。")
    parser.add_argument('--dataset_name', type=str, required=True, help='数据集名称 (例如: Beauty, Sports)')
    parser.add_argument('--fusion_method', type=str, required=True, choices=['concat', 'clip-align', 'projection', 'cross-attention'], help='用于加载对应特征的融合方法')
    parser.add_argument('--base_data_path', type=str, default='../datasets', help='包含所有数据集文件夹的根目录')
    parser.add_argument('--base_output_path', type=str, default='../datasets', help='所有训练产出物的根目录')
    parser.add_argument('--config_path', type=str, default='./rqvae_config.yaml', help='指向 rqvae_config.yaml 的路径')
    args = parser.parse_args()

    # --- 2. 自动构建路径 (NEW) ---
    input_embedding_filename = f"{args.dataset_name}.emb-fused-{args.fusion_method}.npy"
    embedding_path = os.path.join(args.base_data_path, args.dataset_name, input_embedding_filename)
    
    # 为每个实验创建独立的输出文件夹
    output_dir = os.path.join(args.base_output_path, f"{args.dataset_name}", f"{args.fusion_method}", "rqvae")
    os.makedirs(output_dir, exist_ok=True)
    
    print("--- 自动构建路径 ---")
    print(f"输入特征路径: {embedding_path}")
    print(f"配置文件路径: {args.config_path}")
    print(f"输出结果目录: {output_dir}")
    print("---------------------\n")

    # --- 3. 检查文件并设置日志 ---
    for path in [embedding_path, args.config_path]:
        if not os.path.exists(path):
            print(f"错误: 找不到输入文件 {path}")
            return
            
    log_filename = os.path.join(output_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_filename), logging.StreamHandler()])

    # --- 4. 加载配置和数据 ---
    logging.info(f"加载配置文件: {args.config_path}")
    with open(args.config_path, 'r') as f: config = yaml.safe_load(f)
    model_config = config["RQ-VAE"]

    logging.info(f"加载特征文件: {embedding_path}")
    item_embeddings = np.load(embedding_path)
    logging.info(f"特征加载完成. Shape: {item_embeddings.shape}")
    
    device = torch.device(config.get('training', {}).get('device', 'cuda:0'))
    logging.info(f"使用设备: {device}")

    # --- 5. 智能设定 input_dim 并初始化模型 ---
    input_dim_from_data = item_embeddings.shape[1]
    if model_config.get('input_dim') != input_dim_from_data:
        logging.warning(f"配置文件中的 input_dim ({model_config.get('input_dim')}) 与数据维度 ({input_dim_from_data}) 不符。将自动使用数据维度。")
    
    rqvae = RQVAE(
        input_size=input_dim_from_data,
        hidden_sizes=model_config["hidden_dim"],
        latent_size=model_config["latent_dim"],
        num_levels=model_config["num_layers"],
        codebook_size=model_config["code_book_size"],
        dropout=model_config["dropout"],
        beta=model_config["beta"]
    )
    logging.info("RQ-VAE 模型初始化完成。")

    # --- 6. 训练和生成 ---
    logging.info(f"开始训练 RQ-VAE 模型...")
    train_rqvae(rqvae, item_embeddings, device, model_config, output_dir)
    
    best_model_path = os.path.join(output_dir, "rqvae_best_model.pth")
    if os.path.exists(best_model_path):
        logging.info(f"加载最佳模型进行最终 Codebook 生成: {best_model_path}")
        rqvae.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        logging.warning("未找到最佳模型，将使用训练结束时的模型状态。")

    codebook_path = generate_codebook(rqvae, item_embeddings, device, model_config, output_dir)

    logging.info(f"\n--- 训练全过程结束 ---")
    logging.info(f"最佳模型保存在: {best_model_path}")
    logging.info(f"最终码本保存在: {codebook_path}")

if __name__ == '__main__':
    main()