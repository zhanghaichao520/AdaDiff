import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch import nn

# 假设 abstract_vq.py 在同一目录下或在 python 路径中
from .abstract_vq import AbstractVQ

# 
# ==============================================================================
# 辅助类 (MLP, VQEmbedding, RQBottleneck) 保持不变
# ==============================================================================
#

class GatedFusion(nn.Module):
    """自适应门控融合，用于多模态晚期融合"""
    def __init__(self, latent_size):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(latent_size * 2, latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, z_T, z_I):
        # 输出权重 w[:,0]、w[:,1] 分别对应 text/image
        w = self.gate(torch.cat([z_T, z_I], dim=-1))
        z_fused = w[:, 0:1] * z_T + w[:, 1:2] * z_I
        return z_fused, w


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, latent_size, dropout=0.0):
        super(MLP, self).__init__()
        self.mlp_blocks = nn.ModuleList()
        self.residuals = nn.ModuleList()
        self.in_dropout = nn.Dropout(p=dropout)
        self.out_projection = nn.Linear(hidden_sizes[-1], latent_size)
        hidden_sizes = [input_size] + hidden_sizes
        for idx, (input_size, output_size) in enumerate(
            zip(hidden_sizes[:-1], hidden_sizes[1:])
        ):
            self.mlp_blocks.append(
                nn.Sequential(
                    nn.Linear(input_size, output_size),
                    nn.LayerNorm(output_size),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                )
            )
            # add residual connections
            self.residuals.append(
                nn.Conv1d(
                    in_channels=1,
                    out_channels=output_size,
                    kernel_size=input_size,
                    bias=False,
                    stride=input_size,
                )
            )

    def forward(self, x):
        x = self.in_dropout(x)
        for i in range(len(self.mlp_blocks)):
            res = self.residuals[i](x.unsqueeze(1)).squeeze()
            x = self.mlp_blocks[i](x)
            x = x + res
        return self.out_projection(x)


class VQEmbedding(nn.Embedding):
    """VQ embedding module with ema update."""

    def __init__(
        self,
        n_embed,
        embed_dim,
        ema=True,
        decay=0.99,
        restart_unused_codes=True,
        eps=1e-5,
    ):
        super().__init__(n_embed + 1, embed_dim, padding_idx=n_embed)

        self.ema = ema
        self.decay = decay
        self.eps = eps
        self.restart_unused_codes = restart_unused_codes
        self.n_embed = n_embed

        if self.ema:
            _ = [p.requires_grad_(False) for p in self.parameters()]

            # padding index is not updated by EMA
            self.register_buffer("cluster_size_ema", torch.zeros(n_embed))
            self.register_buffer("embed_ema", self.weight[:-1, :].detach().clone())

    @torch.no_grad()
    def compute_distances(self, inputs):
        codebook_t = self.weight[:-1, :].t()

        (embed_dim, _) = codebook_t.shape
        inputs_shape = inputs.shape
        assert inputs_shape[-1] == embed_dim

        inputs_flat = inputs.reshape(-1, embed_dim)

        inputs_norm_sq = inputs_flat.pow(2.0).sum(dim=1, keepdim=True)
        codebook_t_norm_sq = codebook_t.pow(2.0).sum(dim=0, keepdim=True)
        distances = torch.addmm(
            inputs_norm_sq + codebook_t_norm_sq,
            inputs_flat,
            codebook_t,
            alpha=-2.0,
        )
        distances = distances.reshape(
            *inputs_shape[:-1], -1
        )  # [B, h, w, n_embed or n_embed+1]

        return distances

    @torch.no_grad()
    def find_nearest_embedding(self, inputs):
        distances = self.compute_distances(inputs)  # [B, h, w, n_embed or n_embed+1]
        embed_idxs = distances.argmin(dim=-1)  # use padding index or not

        return embed_idxs

    @torch.no_grad()
    def _tile_with_noise(self, x, target_n):
        B, embed_dim = x.shape
        n_repeats = (target_n + B - 1) // B
        std = x.new_ones(embed_dim) * 0.01 / np.sqrt(embed_dim)
        x = x.repeat(n_repeats, 1)
        x = x + torch.rand_like(x) * std
        return x

    @torch.no_grad()
    def _update_buffers(self, vectors, idxs):

        n_embed, embed_dim = self.weight.shape[0] - 1, self.weight.shape[-1]

        vectors = vectors.reshape(-1, embed_dim)
        idxs = idxs.reshape(-1)

        n_vectors = vectors.shape[0]
        n_total_embed = n_embed

        one_hot_idxs = vectors.new_zeros(n_total_embed, n_vectors)
        one_hot_idxs.scatter_(
            dim=0, index=idxs.unsqueeze(0), src=vectors.new_ones(1, n_vectors)
        )

        cluster_size = one_hot_idxs.sum(dim=1)
        vectors_sum_per_cluster = one_hot_idxs @ vectors

        if dist.is_initialized():
            dist.all_reduce(vectors_sum_per_cluster, op=dist.ReduceOp.SUM)
            dist.all_reduce(cluster_size, op=dist.ReduceOp.SUM)

        self.cluster_size_ema.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        self.embed_ema.mul_(self.decay).add_(
            vectors_sum_per_cluster, alpha=1 - self.decay
        )

        if self.restart_unused_codes:
            if n_vectors < n_embed:
                vectors = self._tile_with_noise(vectors, n_embed)
            n_vectors = vectors.shape[0]
            _vectors_random = vectors[torch.randperm(n_vectors, device=vectors.device)][
                :n_embed
            ]

            if dist.is_initialized():
                dist.broadcast(_vectors_random, 0)

            usage = (self.cluster_size_ema.view(-1, 1) >= 1).float()
            self.embed_ema.mul_(usage).add_(_vectors_random * (1 - usage))
            self.cluster_size_ema.mul_(usage.view(-1))
            self.cluster_size_ema.add_(
                torch.ones_like(self.cluster_size_ema) * (1 - usage).view(-1)
            )

    @torch.no_grad()
    def _update_embedding(self):

        n_embed = self.weight.shape[0] - 1
        n = self.cluster_size_ema.sum()
        normalized_cluster_size = (
            n * (self.cluster_size_ema + self.eps) / (n + n_embed * self.eps)
        )
        self.weight[:-1, :] = self.embed_ema / normalized_cluster_size.reshape(-1, 1)

    def forward(self, inputs):
        embed_idxs = self.find_nearest_embedding(inputs)
        if self.training:
            if self.ema:
                self._update_buffers(inputs, embed_idxs)

        embeds = self.embed(embed_idxs)

        if self.ema and self.training:
            self._update_embedding()

        return embeds, embed_idxs

    def embed(self, idxs):
        embeds = super().forward(idxs)
        return embeds


class RQBottleneck(nn.Module):
    """
    Quantization bottleneck via Residual Quantization.
    (保持不变)
    """

    def __init__(
        self,
        latent_shape,
        code_shape,
        decay=0.99,
        shared_codebook=False,
        restart_unused_codes=True,
        commitment_loss="cumsum",
    ):
        super().__init__()

        self.latent_shape = latent_shape
        self.code_shape = code_shape
        self.shared_codebook = shared_codebook
        self.restart_unused_codes = restart_unused_codes
        self.n_embed = [n_embed for n_embed in code_shape]
        self.decay = [decay for _ in range(len(self.code_shape))]

        if self.shared_codebook:
            codebook0 = VQEmbedding(
                self.n_embed[0],
                self.latent_shape,
                decay=self.decay[0],
                restart_unused_codes=restart_unused_codes,
            )
            self.codebooks = nn.ModuleList(
                [codebook0 for _ in range(self.code_shape[-1])]
            )
        else:
            codebooks = [
                VQEmbedding(
                    self.n_embed[idx],
                    latent_shape,
                    decay=self.decay[idx],
                    restart_unused_codes=restart_unused_codes,
                )
                for idx in range(len(self.code_shape))
            ]
            self.codebooks = nn.ModuleList(codebooks)

        self.commitment_loss = commitment_loss

    def quantize(self, x):
        r"""
        Return list of quantized features and the selected codewords by the residual quantization.
        The code is selected by the residuals between x and quantized features by the previous codebooks.
        """
        residual_feature = x.detach().clone()

        quant_list = []
        code_list = []
        aggregated_quants = torch.zeros_like(x)
        for i in range(len(self.code_shape)):
            quant, code = self.codebooks[i](residual_feature)

            residual_feature.sub_(quant)
            aggregated_quants.add_(quant)

            quant_list.append(aggregated_quants.clone())
            code_list.append(code.unsqueeze(-1))

        codes = torch.cat(code_list, dim=-1)
        return quant_list, codes

    def forward(self, x):
        quant_list, codes = self.quantize(x)

        commitment_loss = self.compute_commitment_loss(x, quant_list)
        quants_trunc = quant_list[-1]
        quants_trunc = x + (quants_trunc - x).detach()

        return quants_trunc, commitment_loss, codes

    def compute_commitment_loss(self, x, quant_list):
        r"""
        Compute the commitment loss for the residual quantization.
        The loss is iteratively computed by aggregating quantized features.
        """
        loss_list = []

        for idx, quant in enumerate(quant_list):
            partial_loss = (x - quant.detach()).pow(2.0).mean()
            loss_list.append(partial_loss)

        commitment_loss = torch.mean(torch.stack(loss_list))
        return commitment_loss

    @torch.no_grad()
    def embed_code(self, code):
        assert code.shape[1:] == self.code_shape

        code_slices = torch.chunk(code, chunks=code.shape[-1], dim=-1)

        if self.shared_codebook:
            embeds = [
                self.codebooks[0].embed(code_slice)
                for i, code_slice in enumerate(code_slices)
            ]
        else:
            embeds = [
                self.codebooks[i].embed(code_slice)
                for i, code_slice in enumerate(code_slices)
            ]

        embeds = torch.cat(embeds, dim=-2).sum(-2)

        return embeds

#
# ==============================================================================
# 核心修改：MM_RQVAE
# ==============================================================================
#

class MM_RQVAE(AbstractVQ):
    """
    Multi-Modal RQVAE (MM-RQVAE)
    
    使用两个独立的编码器（文本和图像）和一个共享的量化器，
    以及两个独立的解码器来重建两个模态。
    """
    def __init__(
        self,
        config: dict,
        input_size_text: int,
        input_size_image: int,
    ):
        # --- 继承自 AbstractVQ ---
        super().__init__(config)
        
        # --- 1. 从 config 解析参数 ---
        # ✅ Corrected key
        model_params = config['mm_rqvae']['model_params'] 
        hidden_sizes = model_params['hidden_sizes']
        latent_size = model_params['latent_size']
        num_levels = model_params['num_levels']
        codebook_size = model_params['codebook_size']
        dropout = model_params['dropout']

        # ✅ Corrected key
        train_params = config['mm_rqvae']['training_params'] 
        self.latent_loss_weight = train_params.get('latent_loss_weight', 0.25)
        self.loss_type = train_params.get('loss_type', 'mse')
        
        self.w_recon_T = train_params.get('w_recon_T', 1.0)
        self.w_recon_I = train_params.get('w_recon_I', 0.5)
        
        # --- 2. 构建多模态架构 ---
        # ... (rest of the __init__ remains the same)
        # --- 2. 构建多模态架构 ---
        
        # (新增) 两个独立的编码器
        self.encoder_T = MLP(input_size_text, hidden_sizes, latent_size, dropout=dropout)
        self.encoder_I = MLP(input_size_image, hidden_sizes, latent_size, dropout=dropout)
        self.fusion = GatedFusion(latent_size)

        
        # (新增) 两个独立的解码器
        rev_hidden_sizes = hidden_sizes.copy()
        rev_hidden_sizes.reverse()
        self.decoder_T = MLP(latent_size, rev_hidden_sizes, input_size_text, dropout=dropout)
        self.decoder_I = MLP(latent_size, rev_hidden_sizes, input_size_image, dropout=dropout)
        
        # (不变) 共享的量化器
        code_shape = [codebook_size] * num_levels
        self.quantizer = RQBottleneck(
            latent_shape=latent_size,
            code_shape=code_shape,
        )


    def forward(self, xs_T, xs_I):
        """
        MM-RQVAE 的前向传播
        
        Args:
            xs_T (torch.Tensor): 文本输入 (B, D_text)
            xs_I (torch.Tensor): 图像输入 (B, D_image)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 重建的 (out_T, out_I)
            torch.Tensor: 量化损失 quant_loss
            torch.Tensor: 离散码 code
        """
        # 1. 编码和融合
        z_e = self.encode(xs_T, xs_I)
        
        # 2. 量化
        z_q, quant_loss, code = self.quantizer(z_e)
        
        # 3. 解码
        out_T, out_I = self.decode(z_q)
        
        # 返回元组以便 compute_loss 解析
        return (out_T, out_I), quant_loss, code

    def encode(self, xs_T, xs_I):
        """
        (增强版) 编码阶段：
        - 分别编码两模态
        - 归一化
        - 门控融合
        - 加噪声稳定化
        """
        z_T = self.encoder_T(xs_T)
        z_I = self.encoder_I(xs_I)
        
        # LayerNorm + tanh 归一化
        z_T = torch.tanh(F.layer_norm(z_T, z_T.shape[-1:]))
        z_I = torch.tanh(F.layer_norm(z_I, z_I.shape[-1:]))

        # 门控融合（自适应加权）
        z_fused, weights = self.fusion(z_T, z_I)
        
        # 防止 collapse：加入轻微噪声扰动
        z_fused = z_fused + 0.01 * torch.randn_like(z_fused)
        
        return z_fused


    def decode(self, z_q):
        """
        (修改) 解码器步骤：从共享的 z_q 重建两个模态
        """
        out_T = self.decoder_T(z_q)
        out_I = self.decoder_I(z_q)
        return out_T, out_I

    @torch.no_grad()
    def get_codes(self, xs_T, xs_I):
        """
        (修改) 仅用于推理：获取多模态输入的离散码
        """
        z_e = self.encode(xs_T, xs_I)
        _, _, code = self.quantizer(z_e)
        return code

    def compute_loss(self, forward_outputs, **kwargs):
        """
        (增强版) 多模态损失计算：
        - 模态重建损失
        - 量化损失
        - 一致性对齐损失 (z_T vs z_I)
        """
        xs_T = kwargs.get('xs_T')
        xs_I = kwargs.get('xs_I')
        if xs_T is None or xs_I is None:
            raise ValueError("MM_RQVAE compute_loss 必须接收 'xs_T' 和 'xs_I'")

        (out_T, out_I), quant_loss, code = forward_outputs
        
        # 重建损失
        loss_fn = F.mse_loss if self.loss_type == "mse" else F.l1_loss
        loss_recon_T = loss_fn(out_T, xs_T, reduction="mean")
        loss_recon_I = loss_fn(out_I, xs_I, reduction="mean")
        loss_recon = (self.w_recon_T * loss_recon_T) + (self.w_recon_I * loss_recon_I)
        
        # ============================================================
        # 🔁 InfoNCE 式跨模态对齐（对称双向）
        # ============================================================
        z_T = self.encoder_T(xs_T)
        z_I = self.encoder_I(xs_I)

        # 归一化 latent 表示
        z_T_norm = F.normalize(z_T, dim=-1)
        z_I_norm = F.normalize(z_I, dim=-1)

        # 计算相似度矩阵 (B x B)
        logits_T2I = torch.matmul(z_T_norm, z_I_norm.T) / 0.07
        logits_I2T = logits_T2I.T  # 对称方向

        # 正样本索引
        labels = torch.arange(logits_T2I.size(0), device=z_T.device)

        # 双向 InfoNCE
        loss_T2I = F.cross_entropy(logits_T2I, labels)
        loss_I2T = F.cross_entropy(logits_I2T, labels)
        loss_align = 0.5 * (loss_T2I + loss_I2T)

        # （可选）引入分布正则项：防止塌陷
        sim_reg = (1 - torch.diag(logits_T2I).mean()) ** 2
        loss_align = loss_align + 0.01 * sim_reg

        # 总损失
        loss_total = (
            loss_recon
            + self.latent_loss_weight * quant_loss
            + 0.1 * loss_align
        )

        return {
            "loss_total": loss_total,
            "loss_recon": loss_recon,
            "loss_latent": quant_loss,
            "loss_recon_T": loss_recon_T,
            "loss_recon_I": loss_recon_I,
            "loss_align": loss_align,
        }


    @property
    def is_iterative(self) -> bool:
        """模型需要迭代训练 (VAE 范式)"""
        return True