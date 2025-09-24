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

                                # --- 核心改动 3：更多指标 ---
                with torch.no_grad():
                    # 获取一个 batch 的离散编码
                    val_codes = self.model.get_codes(val_sample_data).detach().cpu().numpy()
                    num_levels = val_codes.shape[1]

                    # 1. 每层活跃 code 数量
                    active_counts = [len(np.unique(val_codes[:, i])) for i in range(num_levels)]
                    logging.info(f"[VAL] Active codes per level: {active_counts}")

                    # 2. 每层 code 使用分布熵（衡量利用均匀度）
                    usage_entropy = []
                    for i in range(num_levels):
                        unique, counts = np.unique(val_codes[:, i], return_counts=True)
                        probs = counts / counts.sum()
                        entropy = -np.sum(probs * np.log2(probs + 1e-9))
                        usage_entropy.append(round(float(entropy), 4))
                    logging.info(f"[VAL] Code usage entropy per level: {usage_entropy}")

                    # 3. 碰撞率（完全相同的 code 行数 / 总数）
                    unique_rows = np.unique(val_codes, axis=0).shape[0]
                    collision_rate = 1 - unique_rows / val_codes.shape[0]
                    logging.info(f"[VAL] Collision rate (batch): {collision_rate:.4f}")
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

        # 1) 数据加载
        full_dataset = EmbeddingDataset(embeddings_path)
        dataloader = DataLoader(full_dataset, batch_size=self.train_params['batch_size'])

        # 2) 推理得到基础 codes
        all_codes = []
        with torch.no_grad():
            for batch_data in tqdm(dataloader, desc="生成基础码"):
                batch_data = batch_data.to(self.device)
                codes = self.model.get_codes(batch_data).detach().cpu().numpy().astype(np.int64)
                all_codes.append(codes)

        base_sids_np = np.vstack(all_codes)

        # 3) 构建去重层并拼最终 codebook
        vocab_size = self.model_config['model_params']['codebook_size']
        dedup_layer = utils.build_dedup_layer(base_sids_np, vocab_size)
        final_codes_np = np.concatenate([base_sids_np, dedup_layer], axis=1).astype(np.int32)
        logging.info(f"最终码本维度: {final_codes_np.shape}")

        # 4) 规范化保存路径：{dataset}/{dataset}.{model}.codebook.{npy,json}
        dataset_name = str(self.config['dataset_name'])
        model_tag = str(self.config['model_name']).lower()

        os.makedirs(codebook_dir, exist_ok=True)  # 兜底确保目录存在
        std_prefix = os.path.join(codebook_dir, f"{dataset_name}.{model_tag}.codebook")
        json_path = f"{std_prefix}.json"
        npy_path  = f"{std_prefix}.npy"

        # 5) 生成 JSON 词表（可用于可视化/调试）
        prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]  # 前几层前缀
        dedup_prefix = "<x_{}>"  # 去重层前缀
        json_dict = {}
        for i, row in enumerate(final_codes_np):
            tokens = [prefix[j].format(code) for j, code in enumerate(row[:-1])]
            tokens.append(dedup_prefix.format(row[-1]))
            json_dict[str(i)] = "".join(tokens)

        with open(json_path, 'w') as f:
            json.dump(json_dict, f, indent=2)

        np.save(npy_path, final_codes_np)

        logging.info(f"码本已保存（标准命名）: JSON -> {json_path}, NPY -> {npy_path}")
