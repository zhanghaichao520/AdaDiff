# /tokenlization_stage/models/r_kmeans.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Literal

from .abstract_vq import AbstractVQ

# -------------------- 距离函数 --------------------
def _pairwise_euclidean2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a: [B, D], b: [K, D] -> [B, K]
    return (a.pow(2).sum(-1, keepdim=True) - 2 * a @ b.t() + b.pow(2).sum(-1)).clamp_min_(0.0)

def _pairwise_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return 1.0 - (a @ b.t()).clamp(-1, 1)

# -------------------- 主类 --------------------
class RKMEANS(AbstractVQ):
    """
    Residual K-Means in 'Hydra-style':
    - residual quantization with L levels
    - layer-wise training (train_layer_wise=True): 逐层学习，上一层收敛后再开下一层
    - KMeans++ init with init buffer (init_buffer_size)
    - optional residual normalization (normalize_residuals=True)
    - mini-batch KMeans SGD-like updates (lr=mb_lr), or EMA 更新
    兼容你的 main/trainer 接口：
      __init__(config, input_size)
      forward(xs) -> (x_recon, recon_loss, codes)
      compute_loss(forward_outputs, batch) -> {'loss_total': ...}
    """

    def __init__(self, config: dict, input_size: Optional[int] = None):
        super().__init__(config)

        # 读取配置（兼容 rkmeans / r_kmeans / 顶层）
        node = (config.get("rkmeans") or config.get("r_kmeans") or {})
        mcfg = node.get("model_params", config.get("model_params", {}))
        tcfg = node.get("training_params", config.get("training_params", {}))

        # 模型参数
        self.code_dim: int = int(input_size if input_size is not None else mcfg.get("code_dim", 64))
        self.num_levels: int = int(mcfg.get("num_levels", mcfg.get("n_layers", 3)))
        self.codebook_size: int = int(mcfg.get("codebook_size", mcfg.get("codebook_width", 256)))
        self.normalize_residuals: bool = bool(mcfg.get("normalize_residuals", True))
        self.track_residuals: bool = bool(mcfg.get("track_residuals", False))  # 仅日志用途
        metric: Literal["euclidean", "cosine"] = mcfg.get("distance", "euclidean")
        self.metric = metric if metric in ("euclidean", "cosine") else "euclidean"
        self._dist_fn = _pairwise_euclidean2 if self.metric == "euclidean" else _pairwise_cosine

        # 训练参数
        self.train_layer_wise: bool = bool(mcfg.get("train_layer_wise", True))
        self.init_buffer_size: int = int(mcfg.get("init_buffer_size", 3072))
        self.update_mode: Literal["sgd_like", "ema"] = tcfg.get("update_mode", "sgd_like")
        self.mb_lr: float = float(tcfg.get("mb_lr", 0.5))          # mini-batch 等效 SGD 学习率
        self.ema_decay: float = float(tcfg.get("ema_decay", 0.9))  # EMA 衰减

        # 每层训练步数（近似“收敛标准”），达到就切换到下一层
        self.level_train_steps: int = int(tcfg.get("level_train_steps", 200))
        # 空簇重启
        self.reseed_empty: bool = bool(tcfg.get("reseed_empty", True))

        # 码本 & 统计
        self.codebooks = nn.Parameter(torch.randn(self.num_levels, self.codebook_size, self.code_dim))
        self.register_buffer("cluster_counts", torch.zeros(self.num_levels, self.codebook_size))  # 使用统计

        # 初始化缓存（KMeans++ 用）
        self.register_buffer("initialized_levels", torch.zeros(self.num_levels, dtype=torch.bool))
        self._init_buffer = []            # list of tensors (residuals)
        self._init_buffer_count = 0

        # 逐层训练状态
        self.current_level: int = 0
        self.level_steps: int = 0  # 当前层累计训练步数（按 batch 计）

        # usage 正则（可选）：更均匀地使用码字（减少碰撞/未用）
        self.use_usage_reg: bool = bool(tcfg.get("usage_reg", False))
        # 分层权重（例如 [0.005, 0.02, 0.04]）
        self.usage_reg_weights = tcfg.get("usage_reg_weights", None)
        self.usage_reg_weight: float = float(tcfg.get("usage_reg_weight", 0.01))
        self._last_usage_hist = None  # forward 缓存

    # -------------------- 工具函数 --------------------
    @torch.no_grad()
    def _maybe_collect_init(self, residual: torch.Tensor):
        """收集初始化缓冲（只在当前层未初始化时生效）"""
        if self.initialized_levels[self.current_level]:
            return
        need = self.init_buffer_size - self._init_buffer_count
        if need <= 0:
            return
        take = min(need, residual.size(0))
        self._init_buffer.append(residual[:take].detach().clone())
        self._init_buffer_count += take

    @torch.no_grad()
    def _kmeanspp_init(self, level: int, data: torch.Tensor):
        """对指定层用 KMeans++ 初始化，data 是该层的 residual 样本 [N, D]"""
        N, D = data.shape
        K = self.codebook_size
        C = torch.empty(K, D, device=data.device, dtype=data.dtype)

        # 第一个中心
        idx0 = torch.randint(0, N, (1,), device=data.device)
        C[0] = data[idx0]

        # 迭代选择其余中心
        for k in range(1, K):
            dist_to_C = self._dist_fn(data, C[:k])       # [N, k]
            min_d2 = dist_to_C.min(dim=1).values         # [N]
            if min_d2.sum() <= 1e-12:
                rand_idx = torch.randint(0, N, (K - k,), device=data.device)
                C[k:] = data[rand_idx]
                break
            probs = (min_d2 / (min_d2.sum() + 1e-12)).clamp_min_(0)
            nxt = torch.multinomial(probs, 1)
            C[k] = data[nxt]

        self.codebooks.data[level] = C
        self.initialized_levels[level] = True

    @torch.no_grad()
    def _layer_init_if_needed(self, residual: torch.Tensor):
        """当前层如果未初始化，等缓冲攒够后做 KMeans++ 初始化"""
        l = self.current_level
        if self.initialized_levels[l]:
            return
        self._maybe_collect_init(residual)
        if self._init_buffer_count >= self.init_buffer_size:
            buf = torch.cat(self._init_buffer, dim=0)[: self.init_buffer_size]
            self._kmeanspp_init(l, buf)
            # 用完清空缓冲
            self._init_buffer.clear()
            self._init_buffer_count = 0

    @torch.no_grad()
    def _mini_batch_update(self, level: int, batch: torch.Tensor, assign_idx: torch.Tensor):
        """
        基于硬分配的 mini-batch 更新（更贴近原始 MiniBatchKMeans）
        """
        K = self.codebook_size
        one_hot = F.one_hot(assign_idx, num_classes=K).to(batch.dtype)  # [B, K]
        counts = one_hot.sum(dim=0)                                     # [K]
        sums = one_hot.t() @ batch                                      # [K, D]

        used = counts > 0
        if used.any():
            batch_mean = sums[used] / counts[used].unsqueeze(1)         # [Ku, D]
            if self.update_mode == "ema":
                self.codebooks.data[level][used] = (
                    self.ema_decay * self.codebooks.data[level][used]
                    + (1 - self.ema_decay) * batch_mean
                )
            else:
                # SGD-like: C <- (1-η)C + η*batch_mean
                eta = self.mb_lr
                self.codebooks.data[level][used] = (
                    (1 - eta) * self.codebooks.data[level][used] + eta * batch_mean
                )
            self.cluster_counts[level, used] += counts[used]

        # 空簇重启
        if self.reseed_empty:
            empty = self.cluster_counts[level] == 0
            if empty.any():
                # 用 batch 的随机样本回填
                ridx = torch.randint(0, batch.size(0), (int(empty.sum()),), device=batch.device)
                self.codebooks.data[level][empty] = batch[ridx]
                # 给一个小的 count，避免再次被当作空簇
                self.cluster_counts[level][empty] = 1.0

        # 返回 batch 使用直方图（用于 usage 正则）
        hist = counts / (counts.sum() + 1e-12)
        return hist  # [K]

    # -------------------- 前向（训练/推理通用） --------------------
    def forward(self, xs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        xs: [B, D]
        返回:
          x_recon: [B, D]
          recon_loss: 标量 MSE
          codes: [B, L] （未训练到的层，用 0 填充）
        """
        B, D = xs.shape
        assert D == self.code_dim, f"Input dim {D} != code_dim {self.code_dim}"

        device = xs.device
        residual = xs
        if self.normalize_residuals:
            # 零均值、单位方差（按 batch）
            mean = residual.mean(dim=0, keepdim=True)
            std = residual.std(dim=0, keepdim=True).clamp_min_(1e-6)
            residual = (residual - mean) / std

        x_recon = torch.zeros_like(xs)
        codes = torch.zeros(B, self.num_levels, dtype=torch.long, device=device)
        usage_hists = []

        # 先“通过已训练层”进行量化（不更新）→ 得到当前层的 residual
        for l in range(self.num_levels):
            C = self.codebooks[l]  # [K, D]
            if not self.initialized_levels[l]:
                break
            dist = self._dist_fn(residual, C)   # [B, K]
            idx = dist.argmin(dim=1)            # 硬分配
            q = F.embedding(idx, C)
            x_recon = x_recon + q
            residual = residual - q
            codes[:, l] = idx

            # 已训练层不再更新，usage 仅用于日志
            hist = (F.one_hot(idx, num_classes=self.codebook_size).float().sum(dim=0))
            hist = (hist / (hist.sum() + 1e-12)).detach()
            usage_hists.append(hist)

        # 处理“当前正在训练的层”
        if self.train_layer_wise and (self.current_level < self.num_levels):
            l = self.current_level
            # 初始化不足则先攒 buffer
            self._layer_init_if_needed(residual)
            if self.initialized_levels[l]:
                C = self.codebooks[l]
                dist = self._dist_fn(residual, C)
                idx = dist.argmin(dim=1)
                q = F.embedding(idx, C)
                x_recon = x_recon + q
                residual = residual - q
                codes[:, l] = idx

                # 只更新当前层
                hist = self._mini_batch_update(l, residual + q, idx)  # 用量化前的该层输入 (= old residual)
                usage_hists.append(hist)

                # 累计步数并考虑“切层”
                self.level_steps += 1
                if self.level_steps >= self.level_train_steps:
                    # 切到下一层
                    self.current_level = min(self.current_level + 1, self.num_levels - 1)
                    self.level_steps = 0
                    # 清空下一层的初始化缓冲
                    self._init_buffer.clear()
                    self._init_buffer_count = 0

        # 未到达/未训练的后续层：codes 留 0 占位；x_recon 不叠加（等后续训练）

        # 重构误差（对当前已叠加的层）
        recon_loss = F.mse_loss(x_recon, xs)

        # 缓存 usage（可能用于 usage 正则）
        self._last_usage_hist = usage_hists if usage_hists else None

        return x_recon, recon_loss, codes

    @torch.no_grad()
    def get_codes(self, xs: torch.Tensor) -> torch.Tensor:
        # 推理阶段：用已训练的所有层逐层量化
        residual = xs
        if self.normalize_residuals:
            mean = residual.mean(dim=0, keepdim=True)
            std = residual.std(dim=0, keepdim=True).clamp_min_(1e-6)
            residual = (residual - mean) / std

        B = xs.size(0)
        device = xs.device
        codes = torch.zeros(B, self.num_levels, dtype=torch.long, device=device)

        for l in range(self.num_levels):
            C = self.codebooks[l]
            dist = self._dist_fn(residual, C)
            idx = dist.argmin(dim=1)
            codes[:, l] = idx
            q = F.embedding(idx, C)
            residual = residual - q

        return codes

    def compute_loss(self, forward_outputs, batch_data=None) -> dict:
        """
        兼容你的 Trainer 调用：loss_dict = model.compute_loss(forward_outputs, batch)
        forward_outputs: (x_recon, recon_loss, codes)
        """
        _, recon_loss, _ = forward_outputs
        loss = recon_loss

        # 可选：分层 usage 正则（提升码字使用熵，减少未用/碰撞）
        if self.use_usage_reg and (self._last_usage_hist is not None):
            if isinstance(self.usage_reg_weights, list):
                w_list = [float(w) for w in self.usage_reg_weights]
            else:
                w_list = None
            for l, hist in enumerate(self._last_usage_hist):
                p = hist.clamp_min(1e-12)
                usage_loss = (p * p.log()).sum()  # = -H 的负号
                w = (w_list[l] if (w_list is not None and l < len(w_list)) else self.usage_reg_weight)
                loss = loss + w * usage_loss

        return {
            "loss_total": loss,
            "loss_recon": recon_loss  # 新增：显式返回重构误差
        }

