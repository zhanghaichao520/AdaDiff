# /quantization/trainer.py (嚴格遵守你的設置，並加入智能調度)

import os
import json
import logging
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

# 假設你的 dataset.py 和 utils.py 都在可導入的路徑
from dataset import EmbeddingDataset
import utils

class Trainer:
    """
    🌐 Universal Trainer for Quantization Models
    Supports both supervised (RQ-VAE) and unsupervised (RKMEANS, OPQ) paradigms.
    """

    def __init__(self, config: dict, model: torch.nn.Module, device: torch.device):
        self.config = config
        self.model = model.to(device)
        self.device = device
        self.model_name = config.get("model_name", model.__class__.__name__)
        self.model_cfg = config.get(self.model_name, {})
        self.train_cfg = self.model_cfg.get("training_params", {})
        self.logger = logging.getLogger(f"Trainer[{self.model_name}]")

    # ====================================================
    # 🔹 1. 通用訓練邏輯 (智能調度入口)
    # ====================================================
    def fit(self, embeddings_path, ckpt_dir):
        """
        通用的 fit 方法，現在是一個智能調度器。
        它會根據模型名稱，自動選擇正確的訓練流程。
        """
        # ✅ 核心改動：在這裡進行簡單、直觀的判斷和分派
        # 未來如果新增 PQ 模型，只需在此列表中加入 'pq' 即可
        if self.model_name in ['opq']:
            return self._fit_one_shot(embeddings_path, ckpt_dir)
        else:
            return self._fit_iterative(embeddings_path, ckpt_dir)

    # ====================================================
    # 🔹 1a. 內部方法：迭代式訓練 (你原來的 fit 邏輯)
    # ====================================================
    def _fit_iterative(self, embeddings_path, ckpt_dir):
        """處理需要迭代訓練的模型 (如 VQ-VAE, RKMEANS)。"""
        self.logger.info(f"檢測到迭代式模型，開始訓練循環...")

        dataset = EmbeddingDataset(embeddings_path)
        # ✅ 增加了驗證集來做模型選擇，更科學
        train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.05, random_state=42)
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=self.train_cfg.get("batch_size", 1024), shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=self.train_cfg.get("batch_size", 1024))
        
        # ✅ 只有在模型有可訓練參數時才創建優化器
        params_to_optimize = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        optimizer = torch.optim.AdamW(params_to_optimize, lr=self.train_cfg.get("lr", 1e-4)) if params_to_optimize else None

        best_loss, best_epoch = float("inf"), 0
        num_epochs = self.train_cfg.get("epochs", 100)
        best_path = os.path.join(ckpt_dir, f"{self.model_name}_best.pth")

        pbar = tqdm(range(num_epochs), desc=f"Training {self.model_name}", ncols=120)
        for epoch in pbar:
            self.model.train()
            epoch_loss_sum = {}
            for batch in train_loader:
                batch = batch.to(self.device)
                outputs = self.model(batch)
                loss_dict = self.model.compute_loss(outputs, batch)
                loss_total = loss_dict.get("loss_total", 0)

                # 僅對可微且有優化器的模型執行 backward
                if optimizer and hasattr(loss_total, 'requires_grad') and loss_total.requires_grad:
                    optimizer.zero_grad()
                    loss_total.backward()
                    optimizer.step()
                
                for key, val in loss_dict.items():
                    epoch_loss_sum[key] = epoch_loss_sum.get(key, 0.0) + float(val.item())

            avg_losses = {k: v / len(train_loader) for k, v in epoch_loss_sum.items()}

            # ✅ 執行驗證
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    outputs = self.model(batch)
                    val_loss += self.model.compute_loss(outputs, batch).get('loss_total', 0)
            avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
            
            postfix_str = f"train_loss={avg_losses.get('loss_total', 0):.4f}, val_loss={avg_val_loss:.4f}, best_val_loss={best_loss:.4f}"
            pbar.set_postfix_str(postfix_str)

            # ✅ 根據驗證集損失保存最優模型
            if avg_val_loss < best_loss:
                best_loss, best_epoch = avg_val_loss, epoch + 1
                if optimizer: # 只有可訓練的模型才需要保存
                    torch.save(self.model.state_dict(), best_path)

        pbar.close()
        self.logger.info("=" * 100)
        self.logger.info(f"🏁 迭代式訓練完成 [{self.model_name}]")
        self.logger.info(f"📉 最佳驗證集 Loss: {best_loss:.6f} (在 Epoch {best_epoch})")
        if optimizer: self.logger.info(f"💾 最佳模型已保存至: {best_path}")
        self.logger.info("=" * 100)
        return best_path

    # ====================================================
    # 🔹 1b. 內部方法：一次性擬合
    # ====================================================
    def _fit_one_shot(self, embeddings_path: str, ckpt_dir: str) -> str:
        """處理一次性擬合的模型 (如 OPQ)。"""
        self.logger.info(f"檢測到 one-shot 模型，開始一次性擬合...")
        self.model.train()
        
        full_dataset = EmbeddingDataset(embeddings_path)
        full_data_tensor = full_dataset.embeddings.to(self.device)
        
        # 直接將全部數據喂給模型的 forward 來觸發擬合
        # 假設 one-shot 模型的 forward 會處理這個邏輯
        self.model(full_data_tensor)
        
        best_path = os.path.join(ckpt_dir, f"{self.model_name}_fitted.pth")
        torch.save({}, best_path) # 保存一個空字典作為完成信號
        
        self.logger.info("=" * 100)
        self.logger.info(f"🏁 One-shot 擬合完成 [{self.model_name}]")
        self.logger.info(f"💾 擬合完成信號已保存至: {best_path}")
        self.logger.info("=" * 100)
        return best_path

    # ====================================================
    # 🔹 2. 通用碼本生成邏輯 (你的原版 predict，完全不變)
    # ====================================================
    def predict(self, embeddings_path: str, codebook_dir: str):
        self.logger.info(f"开始生成码本 ({self.model_name}) ...")
        self.model.eval()

        dataset = EmbeddingDataset(embeddings_path)
        # 增加 batch_size 以加速推論
        loader = DataLoader(dataset, batch_size=self.train_cfg.get("batch_size", 2048) * 2)
        all_codes = []

        with torch.no_grad():
            for batch in tqdm(loader, desc="编码中"):
                batch = batch.to(self.device)
                if hasattr(self.model, "get_codes"):
                    codes = self.model.get_codes(batch)
                elif hasattr(self.model, "encode"):
                    codes = self.model.encode(batch)
                else:
                    raise ValueError(f"{self.model_name} 缺少 get_codes/encode 方法")
                all_codes.append(codes.detach().cpu().numpy().astype(np.int64))

        base_codes = np.vstack(all_codes)
        self.logger.info(f"基礎碼本生成完畢，形狀: {base_codes.shape}")

        # === ✅ 關鍵改動：根據 config 決定是否添加去重層 ===
        model_params = self.model_cfg.get("model_params", {})
        # 使用 .get('has_dup_layer', True) 確保如果 config 中沒有這個鍵，預設行為是添加去重層
        if model_params.get('has_dup_layer', True):
            self.logger.info("配置中 'has_dup_layer' 為 True 或未設置，將構建去重層。")
            vocab_size = model_params.get("codebook_size", 1024)
            dedup = utils.build_dedup_layer(base_codes, vocab_size)
            final_codes = np.concatenate([base_codes, dedup], axis=1)
        else:
            self.logger.info("配置中 'has_dup_layer' 設置為 False，將不構建去重層。")
            final_codes = base_codes
        # =======================================================

        os.makedirs(codebook_dir, exist_ok=True)
        dataset_name = self.config["dataset_name"]
        model_tag = self.model_name.lower()
        # 檔名格式: {dataset_name}.{model_name}.codebook
        prefix = os.path.join(codebook_dir, f"{dataset_name}.{model_tag}.codebook")

        np.save(f"{prefix}.npy", final_codes)
        
        # 保存 JSON 格式 (可選)
        json_path = f"{prefix}.json"
        json_dict = {str(i): " ".join([f"<L{l}_{v}>" for l, v in enumerate(row)]) for i, row in enumerate(final_codes)}
        with open(json_path, "w") as f: json.dump(json_dict, f, indent=2)

        self.logger.info(f"✅ 码本保存完成，最終形狀: {final_codes.shape}，已保存至: {prefix}.(npy/json)")
        return final_codes