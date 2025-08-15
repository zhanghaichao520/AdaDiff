# /tokenlization_stage/trainer.py

import os
import json
import logging
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np

from dataset import EmbeddingDataset
import evaluate
import utils

class Trainer:
    def __init__(self, config: dict, model: torch.nn.Module, device: torch.device):
        self.config = config
        self.model = model
        self.device = device
        self.model_name = config['model_name']
        self.model_config = config[self.model_name]
        self.train_params = self.model_config['training_params']

    def fit(self, embeddings_path: str, ckpt_dir: str):
        logging.info(f"--- 开始训练模型: {self.model_name} ---")
        full_dataset = EmbeddingDataset(embeddings_path)
        
        indices = list(range(len(full_dataset)))
        train_indices, val_indices = train_test_split(indices, test_size=0.05, random_state=42)
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)

        train_loader = DataLoader(train_dataset, batch_size=self.train_params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.train_params['batch_size'])
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.train_params['lr'])
        best_val_loss = float('inf')
        best_model_path = os.path.join(ckpt_dir, "best_model.pth")
        
        num_epochs = self.train_params.get('epochs', 100) # 从配置获取epochs
        
        for epoch in tqdm(range(num_epochs), desc=f"训练 {self.model_name}"):
            self.model.train()
            for batch_data in train_loader:
                batch_data = batch_data.to(self.device)
                optimizer.zero_grad()
                forward_outputs = self.model(batch_data)
                loss_dict = self.model.compute_loss(forward_outputs, batch_data)
                loss = loss_dict['loss_total']
                loss.backward()
                optimizer.step()
            
            # --- 验证循环 (已增强) ---
            eval_interval = self.train_params.get('eval_interval', 100)
            if (epoch + 1) % eval_interval == 0:
                self.model.eval()
                # --- 核心改动 1：初始化用于累加详细损失的变量 ---
                total_val_loss, total_val_recon, total_val_latent = 0, 0, 0
                
                with torch.no_grad():
                    for batch_data in val_loader:
                        batch_data = batch_data.to(self.device)
                        forward_outputs = self.model(batch_data)
                        loss_dict = self.model.compute_loss(forward_outputs, batch_data)
                        
                        # --- 核心改动 1：累加所有损失项 ---
                        total_val_loss += loss_dict['loss_total'].item()
                        if 'loss_recon' in loss_dict:
                            total_val_recon += loss_dict['loss_recon'].item()
                        if 'loss_latent' in loss_dict:
                            total_val_latent += loss_dict['loss_latent'].item()

                # 计算平均值
                avg_val_loss = total_val_loss / len(val_loader)
                avg_val_recon = total_val_recon / len(val_loader)
                avg_val_latent = total_val_latent / len(val_loader)
                
                # 打印增强后的日志
                logging.info(f"[VAL] Epoch {epoch+1:04d} | Total Loss: {avg_val_loss:.4f} (Recon: {avg_val_recon:.4f}, Latent: {avg_val_latent:.4f})")

                # --- 核心改动 2：加入余弦相似度计算 ---
                # 从验证集中取一个batch的数据用于计算
                val_sample_data = next(iter(val_loader)).to(self.device)
                cos_sim_array = utils.calc_cos_sim(self.model, val_sample_data, self.model_config['model_params'])
                logging.info(f"[VAL] Cosine Similarity per level: {np.round(cos_sim_array, 4)}")
                # --- 改动结束 ---

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(self.model.state_dict(), best_model_path)
                    logging.info(f"模型已保存到: {best_model_path}")
        
        logging.info("训练完成。")
        return best_model_path

    def predict(self, embeddings_path: str, codebook_dir: str):
        logging.info(f"--- 开始用 {self.model_name} 生成码本 ---")
        self.model.eval()
        full_dataset = EmbeddingDataset(embeddings_path)
        dataloader = DataLoader(full_dataset, batch_size=self.train_params['batch_size'])
        
        all_codes = []
        with torch.no_grad():
            for batch_data in tqdm(dataloader, desc="生成基础码"):
                batch_data = batch_data.to(self.device)
                codes = self.model.get_codes(batch_data).detach().cpu().numpy().astype(np.int64)
                all_codes.append(codes)

        base_sids_np = np.vstack(all_codes)
        vocab_size = self.model_config['model_params']['codebook_size']
        dedup_layer = utils.build_dedup_layer(base_sids_np, vocab_size)
        final_codes_np = np.concatenate([base_sids_np, dedup_layer], axis=1).astype(np.int32)
        logging.info(f"最终码本维度: {final_codes_np.shape}")
        
        json_path = os.path.join(codebook_dir, f"{self.config['dataset_name']}.codebook.json")
        pt_path = os.path.join(codebook_dir, f"{self.config['dataset_name']}.codebook.pt")
        
        # 保存逻辑...
        with open(json_path, 'w') as f:
             json.dump({str(i): final_codes_np[i].tolist() for i in range(len(final_codes_np))}, f, indent=2)
        torch.save(torch.from_numpy(final_codes_np.T).contiguous().long(), pt_path)
        logging.info(f"码本已保存: JSON -> {json_path}, PT -> {pt_path}")