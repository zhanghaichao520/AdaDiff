import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import yaml
import argparse
import logging
from datetime import datetime
import torch.nn as nn
# 假设您的 RQVAE 模型定义在一个可以引用的地方
# 如果 RQVAE 类就在同一个文件夹，可以这样写:
from rqvae import RQVAE
# 这里为了让脚本独立，我们把 RQVAE 的简化定义放在这里
# 请确保这里的定义和您实际使用的模型一致

def train_epoch(model, dataloader, optimizer, beta, flag_eval=False):
    model.eval() if flag_eval else model.train()
    total_loss, total_rec_loss, total_commit_loss = 0.0, 0.0, 0.0

    for batch in dataloader:
        x_batch = batch[0]
        if not flag_eval:
            optimizer.zero_grad()

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
    batch_size = config["batch_size"]
    num_epochs = config["epochs"]
    lr = config["lr"]
    weight_decay = config.get("weight_decay", 0.0)

    optimizer = getattr(torch.optim, config["optimizer"])(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 数据集分割
    train_data, val_data = train_test_split(embeddings, test_size=0.05, random_state=42)
    train_tensor = torch.Tensor(train_data).to(device)
    val_tensor = torch.Tensor(val_data).to(device)
    
    train_dataset = TensorDataset(train_tensor)
    val_dataset = TensorDataset(val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_loss = float('inf')
    
    for epoch in tqdm(range(num_epochs), desc="Training RQ-VAE"):
        train_loss, _, _ = train_epoch(model, train_loader, optimizer, config["beta"])

        if (epoch + 1) % 100 == 0: # 每100个epoch评估一次
            val_loss, val_rec_loss, val_commit_loss = train_epoch(model, val_loader, None, config["beta"], flag_eval=True)
            logging.info(f"[VALIDATION] Epoch {epoch+1:04d} | Val Loss: {val_loss:.4f} | Recon Loss: {val_rec_loss:.4f} | Commit Loss: {val_commit_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_name = "rqvae_best_model.pth"
                save_path = os.path.join(output_dir, model_name)
                torch.save(model.state_dict(), save_path)
                logging.info(f"[MODEL] Best model saved to: {save_path}")

    print("[TRAINING] Training complete.")


def generate_codebook(model, embeddings, device, config, output_dir):
    logging.info("[CODEBOOK] Generating Codebook from all items...")

    model.to(device)
    model.eval()
    
    all_codes_list = []
    dataset = TensorDataset(torch.Tensor(embeddings))
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating Codes"):
            x_batch = batch[0].to(device)
            codes = model.get_codes(x_batch).cpu().numpy()
            all_codes_list.append(codes)

    all_codes_np = np.vstack(all_codes_list)
    logging.info(f"[CODEBOOK] Successfully generated all codes with shape: {all_codes_np.shape}")

    # item_id 从 0 开始
    item_to_codes = {
        str(item_id): codes.tolist()
        for item_id, codes in enumerate(all_codes_np)
    }

    codebook_path = os.path.join(output_dir, "codebook.json")
    with open(codebook_path, 'w') as f:
        json.dump(item_to_codes, f)
    logging.info(f"[CODEBOOK] Codebook successfully saved to: {codebook_path}")
    return codebook_path

# --- 主程序入口 ---

def main():
    parser = argparse.ArgumentParser(description="从 .npy 文件训练 RQ-VAE 模型。")
    parser.add_argument('--embedding_path', type=str, required=True, help='指向 .npy 格式的特征文件路径')
    parser.add_argument('--config_path', type=str, required=True, help='指向 rqvae_config.yaml 配置文件的路径')
    parser.add_argument('--output_dir', type=str, required=True, help='保存模型、日志和 Codebook 的输出目录')
    args = parser.parse_args()

    # --- 1. 创建输出目录和日志 ---
    os.makedirs(args.output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(args.output_dir, f"training_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
    )

    # --- 2. 加载配置和数据 ---
    logging.info(f"加载配置文件: {args.config_path}")
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    model_config = config["RQ-VAE"]

    logging.info(f"加载特征文件: {args.embedding_path}")
    item_embeddings = np.load(args.embedding_path)
    logging.info(f"特征加载完成. Shape: {item_embeddings.shape}")
    
    device = torch.device(config.get('training', {}).get('device', 'cuda:0'))
    logging.info(f"使用设备: {device}")

    # --- 3. 智能设定 input_dim 并初始化模型 ---
    # 自动从数据推断输入维度，而不是依赖配置文件，这样更稳健！
    input_dim_from_data = item_embeddings.shape[1]
    if model_config.get('input_dim') != input_dim_from_data:
        logging.warning(f"配置文件中的 input_dim ({model_config.get('input_dim')}) 与数据维度 ({input_dim_from_data}) 不符。")
        logging.warning(f"将自动使用从数据中推断的维度: {input_dim_from_data}")
    
    # 替换为您自己项目中 RQVAE 类的真实导入和调用
    # from your_project.rqvae import RQVAE
    rqvae = RQVAE(
        input_dim=input_dim_from_data, # 使用从数据中得到的真实维度
        hidden_dims=model_config["hidden_dim"],
        latent_dim=model_config["latent_dim"],
        num_layers=model_config["num_layers"],
        code_book_size=model_config["code_book_size"],
        dropout=model_config["dropout"],
        beta=model_config["beta"]
    )
    logging.info("RQ-VAE 模型初始化完成。")

    # --- 4. 训练模型 ---
    logging.info(f"开始训练 RQ-VAE 模型...")
    train_rqvae(rqvae, item_embeddings, device, model_config, args.output_dir)

    # --- 5. 生成最终 Codebook ---
    # 加载训练过程中保存的最佳模型
    best_model_path = os.path.join(args.output_dir, "rqvae_best_model.pth")
    if os.path.exists(best_model_path):
        logging.info(f"加载最佳模型进行最终 Codebook 生成: {best_model_path}")
        rqvae.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        logging.warning("未找到最佳模型，将使用训练结束时的模型状态。")

    codebook_path = generate_codebook(rqvae, item_embeddings, device, model_config, args.output_dir)

    logging.info(f"\n--- 训练全过程结束 ---")
    logging.info(f"最佳模型保存在: {best_model_path}")
    logging.info(f"最终码本保存在: {codebook_path}")

if __name__ == '__main__':
    main()