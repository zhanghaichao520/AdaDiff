# /quantization/rqvae/train.py

import os
import sys
import json
import logging
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 从同级目录导入模型定义和通用工具
from .rqvae import RQVAE
from quantization import utils

def _train_epoch(model, dataloader, optimizer, beta, device, is_eval=False):
    """
    (内部函数) 单个epoch的训练/评估。这是RQ-VAE特有的逻辑。
    """
    model.eval() if is_eval else model.train()
    total_loss, total_rec_loss, total_commit_loss = 0.0, 0.0, 0.0
    
    for (x_batch,) in dataloader:
        x_batch = x_batch.to(device)
        if not is_eval: optimizer.zero_grad()
        
        with torch.set_grad_enabled(not is_eval):
            recon_x, commitment_loss, _ = model(x_batch)
            reconstruction_loss = F.mse_loss(recon_x, x_batch, reduction="mean")
            loss = reconstruction_loss + beta * commitment_loss
        
        if not is_eval:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
        total_loss += loss.item()
        total_rec_loss += reconstruction_loss.item()
        total_commit_loss += commitment_loss.item()
        
    return tuple(map(lambda x: x / len(dataloader), [total_loss, total_rec_loss, total_commit_loss]))


def run_pipeline(embeddings, device, config, ckpt_dir, codebook_dir):
    """
    (公开接口) 执行完整的RQ-VAE训练和码本生成流程。
    这个函数将被顶层的 main.py 调用。
    """
    logging.info("================== RQ-VAE Pipeline-开始 ==================")
    
    # 1. 初始化模型
    input_dim = embeddings.shape[1]
    model = RQVAE(
        input_size=input_dim,
        **config['model_params']
    ).to(device)
    logging.info("RQ-VAE 模型初始化完成。")

    # 2. 训练模型
    logging.info("开始训练 RQ-VAE 模型...")
    train_cfg = config['training_params']
    batch_size, num_epochs, lr, beta = train_cfg['batch_size'], train_cfg['epochs'], train_cfg['lr'], train_cfg['beta']
    optimizer = getattr(torch.optim, train_cfg['optimizer'])(model.parameters(), lr=lr, weight_decay=train_cfg.get('weight_decay', 0.0))
    
    train_data, val_data = train_test_split(embeddings, test_size=0.05, random_state=42)
    train_loader = DataLoader(TensorDataset(torch.Tensor(train_data)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.Tensor(val_data)), batch_size=batch_size, shuffle=False)
    
    best_val_loss = float('inf')
    best_model_path = os.path.join(ckpt_dir, "best_model.pth")

    for epoch in tqdm(range(num_epochs), desc="训练 RQ-VAE"):
        _train_epoch(model, train_loader, optimizer, beta, device)
        if (epoch + 1) % 100 == 0:
            val_loss, val_rec, val_commit = _train_epoch(model, val_loader, None, beta, device, is_eval=True)
            logging.info(f"[VAL] Ep {epoch+1:04d} | Loss: {val_loss:.4f} (Rec: {val_rec:.4f}, Commit: {val_commit:.4f})")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                logging.info(f"模型已保存到: {best_model_path}")

    logging.info("训练完成。")

    # 3. 加载最佳模型
    if os.path.exists(best_model_path):
        logging.info(f"加载最佳模型进行码本生成: {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        logging.warning("未找到最佳模型，将使用训练结束时的模型状态。")
    model.eval()

    # 4. 生成码本
    logging.info("开始生成码本...")
    model_cfg = config['model_params']
    vocab_size = model_cfg['codebook_size']

    all_codes = []
    dataloader = DataLoader(TensorDataset(torch.tensor(embeddings, dtype=torch.float32)), batch_size=batch_size)
    with torch.no_grad():
        for (x_batch,) in tqdm(dataloader, desc="生成基础码"):
            codes = model.get_codes(x_batch.to(device)).detach().cpu().numpy().astype(np.int64)
            all_codes.append(codes)

    base_sids_np = np.vstack(all_codes)
    dedup_layer = utils.build_dedup_layer(base_sids_np, vocab_size)
    final_codes_np = np.concatenate([base_sids_np, dedup_layer], axis=1).astype(np.int32)
    logging.info(f"最终码本维度: {final_codes_np.shape}")
    
    # 保存码本
    json_path = os.path.join(codebook_dir, "codebook.json")
    with open(json_path, 'w') as f:
        json.dump({str(i): final_codes_np[i].tolist() for i in range(len(final_codes_np))}, f, indent=2)
    pt_path = os.path.join(codebook_dir, "codebook.pt")
    torch.save(torch.from_numpy(final_codes_np.T).contiguous().long(), pt_path)
    logging.info(f"码本已保存: JSON -> {json_path}, PT -> {pt_path}")