import os
import sys
import json
from collections import defaultdict
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

# --- 核心训练函数 ---
def train_epoch(model, dataloader, optimizer, beta, device, flag_eval=False):
    model.eval() if flag_eval else model.train()
    total_loss, total_rec_loss, total_commit_loss = 0.0, 0.0, 0.0
    for batch in dataloader:
        x_batch = batch[0]
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
        train_loss, _, _ = train_epoch(model, train_loader, optimizer, config["beta"], device)
        if (epoch + 1) % 100 == 0:
            val_loss, val_rec_loss, val_commit_loss = train_epoch(model, val_loader, None, config["beta"], device, flag_eval=True)
            logging.info(f"[VALIDATION] Epoch {epoch+1:04d} | Val Loss: {val_loss:.4f} | Recon Loss: {val_rec_loss:.4f} | Commit Loss: {val_commit_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(output_dir, "rqvae_best_model.pth")
                torch.save(model.state_dict(), save_path)
                logging.info(f"[MODEL] Best model saved to: {save_path}")
    print("[TRAINING] Training complete.")

def _build_one_suffix_local_rank(base_sids_np: np.ndarray):
    keys = [tuple(row.tolist()) for row in base_sids_np]
    groups = defaultdict(list)
    for idx, k in enumerate(keys):
        groups[k].append(idx)

    local_rank = np.zeros(len(base_sids_np), dtype=np.int64)
    max_dup = 1
    for g in groups.values():
        for r, i in enumerate(g):
            local_rank[i] = r  # 同簇内 0..k-1
        if len(g) > max_dup: max_dup = len(g)
    return local_rank.reshape(-1, 1), max_dup

def generate_codebook(model, embeddings, device, config, output_dir):
    """
    生成最终码本：
      - 基础层来自 RQVAE：形如 (N, D_base)
      - 追加去重层（第 D_base+1 层）：对“基础层完全相同”的条目，在各自簇内编号 0..k-1
      - 保存为 .pt 时转置为 (D_final, N) —— 供 TIGER 模型使用
    """
    import numpy as np
    from collections import defaultdict

    logging.info("[CODEBOOK] Generating Codebook from all items...")
    model.to(device)
    model.eval()

    vocab = int(config["code_book_size"])           # 例如 2048
    n_base = int(config["num_layers"])              # 例如 3
    bs = int(config["batch_size"])

    all_codes = []
    dataloader = DataLoader(
        TensorDataset(torch.tensor(embeddings, dtype=torch.float32)),
        batch_size=bs, shuffle=False
    )

    with torch.no_grad():
        for (x,) in tqdm(dataloader, desc="Generating Codes"):
            codes = model.get_codes(x.to(device)).detach().cpu().numpy().astype(np.int64)  # (B, D_base)
            if codes.shape[1] != n_base:
                raise ValueError(f"RQVAE返回的层数{codes.shape[1]}与config.num_layers={n_base}不一致")
            all_codes.append(codes)

    base_sids_np = np.vstack(all_codes)  # (N, D_base)
    if base_sids_np.size == 0:
        raise ValueError("空的码本：没有生成到任何 codes")

    # 基础层取值检查（必须在 [0, vocab)）
    if base_sids_np.min() < 0 or base_sids_np.max() >= vocab:
        raise ValueError(f"基础层存在越界值，代码范围应在 [0,{vocab})，实际最小={base_sids_np.min()} 最大={base_sids_np.max()}")

    N = base_sids_np.shape[0]
    logging.info(f"[CODEBOOK] 基础层 N={n_base}, Shape={base_sids_np.shape}")

    # —— 以“基础层完全相同”为 key 分簇，并在簇内分配 0..k-1 的去重ID（保证 < vocab）——
    groups = defaultdict(list)  # key: tuple(base_codes) -> [indices...]
    for idx, key in enumerate(map(tuple, base_sids_np)):
        groups[key].append(idx)

    dedup = np.zeros((N, 1), dtype=np.int64)
    max_dup = 0
    overflow = 0
    for idx_list in groups.values():
        k = len(idx_list)
        max_dup = max(max_dup, k)
        if k > vocab:
            # 理论上不该发生；保险处理：取模并告警
            logging.warning(f"[CODEBOOK] 出现簇大小 {k} > vocab {vocab}，去重层将取模处理")
            local = np.arange(k, dtype=np.int64) % vocab
            overflow += 1
        else:
            local = np.arange(k, dtype=np.int64)
        dedup[np.array(idx_list), 0] = local

    final_codes_np = np.concatenate([base_sids_np, dedup], axis=1).astype(np.int32)  # (N, D_final)
    D_final = final_codes_np.shape[1]
    logging.info(f"[CODEBOOK] 最终层数 N+1={D_final}, Shape={final_codes_np.shape}, max_dup_in_cluster={max_dup}, vocab(共享)={vocab}")
    if overflow > 0:
        logging.warning(f"[CODEBOOK] 有 {overflow} 个簇发生了取模，建议增大 vocab 或检查碰撞率")

    # —— 保存 JSON（N 行，排查方便）——
    json_codebook_path = os.path.join(output_dir, "codebook.json")
    with open(json_codebook_path, 'w') as f:
        json.dump({str(i): final_codes_np[i].tolist() for i in range(N)}, f)
    logging.info(f"[CODEBOOK] JSON -> {json_codebook_path}")

    # —— 保存 PT：**转置**成 (D_final, N)，供 TIGER 侧直接加载使用 —— 关键！
    tensor_codebook_path = os.path.join(output_dir, "codebook.pt")
    torch.save(torch.from_numpy(final_codes_np.T).contiguous().long(), tensor_codebook_path)
    logging.info(f"[CODEBOOK] PT   -> {tensor_codebook_path}")

    return json_codebook_path, tensor_codebook_path


# --- 主程序入口 ---
def main():
    parser = argparse.ArgumentParser(description="从融合后的 .npy 文件训练 RQ-VAE 模型。")
    parser.add_argument('--dataset_name', type=str, required=True, help='数据集名称 (例如: Beauty, Sports)')
    parser.add_argument('--fusion_method', type=str, required=True, choices=['concat', 'clip-align', 'projection', 'cross-attention'], help='用于加载对应特征的融合方法')
    parser.add_argument('--data_base_path', type=str, default='../datasets', help='包含所有数据集文件夹的根目录')
    parser.add_argument('--log_base_path', type=str, default='../logs', help='日志文件保存的根目录')
    parser.add_argument('--ckpt_base_path', type=str, default='../ckpt', help='模型检查点保存的根目录')
    parser.add_argument('--codebook_base_path', type=str, default='../datasets', help='码本文件保存的根目录')
    parser.add_argument('--config_path', type=str, default='./rqvae_config.yaml', help='指向 rqvae_config.yaml 的路径')
    args = parser.parse_args()

    # --- 自动构建路径 ---
    input_embedding_filename = f"{args.dataset_name}.emb-fused-{args.fusion_method}.npy"
    embedding_path = os.path.join(args.data_base_path, args.dataset_name, input_embedding_filename)
    
    # 定义新的日志和检查点目录
    log_dir = os.path.join(args.log_base_path, args.dataset_name)
    ckpt_dir = os.path.join(args.ckpt_base_path, args.dataset_name)
    codebook_dir = os.path.join(args.codebook_base_path, args.dataset_name)

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(codebook_dir, exist_ok=True)
    
    print("--- 自动构建路径 ---")
    print(f"输入特征路径: {embedding_path}")
    print(f"配置文件路径: {args.config_path}")
    print(f"日志目录: {log_dir}")
    print(f"检查点目录: {ckpt_dir}")
    print(f"码本目录: {codebook_dir}")
    print("---------------------\n")

    # --- 检查文件并设置日志 ---
    for path in [embedding_path, args.config_path]:
        if not os.path.exists(path):
            print(f"错误: 找不到输入文件 {path}")
            return
            
    log_filename = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_filename), logging.StreamHandler()])

    # --- 加载配置和数据 ---
    logging.info(f"加载配置文件: {args.config_path}")
    with open(args.config_path, 'r') as f: config = yaml.safe_load(f)
    model_config = config["RQ-VAE"]

    logging.info(f"加载特征文件: {embedding_path}")
    item_embeddings = np.load(embedding_path)
    logging.info(f"特征加载完成. Shape: {item_embeddings.shape}")
    
    device = torch.device(config.get('training', {}).get('device', 'cuda:0'))
    logging.info(f"使用设备: {device}")

    # --- 智能设定 input_dim 并初始化模型 ---
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

    # --- 训练和生成 ---
    logging.info(f"开始训练 RQ-VAE 模型...")
    train_rqvae(rqvae, item_embeddings, device, model_config, ckpt_dir)
    
    best_model_path = os.path.join(ckpt_dir, "rqvae_best_model.pth")
    if os.path.exists(best_model_path):
        logging.info(f"加载最佳模型进行最终 Codebook 生成: {best_model_path}")
        rqvae.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        logging.warning("未找到最佳模型，将使用训练结束时的模型状态。")

    json_codebook_path, tensor_codebook_path = generate_codebook(rqvae, item_embeddings, device, model_config, codebook_dir)

    logging.info(f"\n--- 训练全过程结束 ---")
    logging.info(f"最佳模型保存在: {best_model_path}")
    logging.info(f"最终 JSON 码本保存在: {json_codebook_path}")
    logging.info(f"最终 PyTorch 码本保存在: {tensor_codebook_path}")

if __name__ == '__main__':
    main()