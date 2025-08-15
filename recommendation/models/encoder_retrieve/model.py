import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model

from dataset import AbstractDataset
from model import AbstractModel
from tokenizer import AbstractTokenizer


class ResBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.zeros_(self.linear.weight)
        self.act = nn.SiLU()
    def forward(self, x):
        return x + self.act(self.linear(x))


class encoder_retrieve(AbstractModel):
    def __init__(self, config: dict, tokenizer: AbstractTokenizer):
        super(encoder_retrieve, self).__init__(config, None, tokenizer)

        self.rqvae_config = self.config['RQ-VAE']
        self.codebook_size = int(self.rqvae_config['code_book_size'])
        self.n_pred_head = int(self.tokenizer.n_digit)  # == D

        # 物品ID -> 码本token（全局ID）的查表：形状 [max_item_id+1, D]
        self.item_id2tokens = self.tokenizer.item_id2tokens_tensor

        gpt2config = GPT2Config(vocab_size=tokenizer.vocab_size, **config)
        self.gpt2 = GPT2Model(gpt2config)

        self.pred_heads = nn.Sequential(*[ResBlock(config['n_embd']) for _ in range(self.n_pred_head)])
        self.temperature = float(self.config['temperature'])
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.ignored_label)

        self._warned_input_type = False  # 仅首个batch打印一次输入类型

    @property
    def n_parameters(self) -> str:
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb_params = sum(p.numel() for p in self.gpt2.get_input_embeddings().parameters() if p.requires_grad)
        return (f'#Embedding parameters: {emb_params}\n'
                f'#Non-embedding parameters: {total_params - emb_params}\n'
                f'#Total trainable parameters: {total_params}\n')

    # ---------- 工具：把 input_ids 和 attention_mask 规整为 item 级别 ----------
    def _to_digit_tokens_and_mask(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        返回:
          digit_tokens: [B, T_item, D]（每个 item 的 D 个全局 token id）
          item_mask   : [B, T_item]   （item 级别的 mask）
        兼容两种输入：
          A) input_ids 是 item_id: 取表 self.item_id2tokens 再得到 digit_tokens；
          B) input_ids 是展平后的 token 序列: 直接按 D 还原为 [B, T_item, D]。
        """
        B, L = input_ids.shape
        D = self.n_pred_head
        table = self.item_id2tokens  # [max_item_id+1, D]

        # 判定：若 max(input_ids) >= 物品表行数，则认为输入已经是 token
        is_token_input = int(input_ids.max().item()) >= table.size(0)

        if not self._warned_input_type:
            mx = int(input_ids.max().item()); nrow = table.size(0)
            msg = "TOKEN input" if is_token_input else "ITEM-ID input"
            acc = self.config.get('accelerator', None)



            self._warned_input_type = True

        if is_token_input:
            assert (L % D) == 0, f"Flatten tokens length {L} not divisible by n_digit={D}"
            T_item = L // D
            digit_tokens = input_ids.view(B, T_item, D)                      # [B, T, D]
            item_mask = attention_mask.view(B, T_item, D).any(dim=2).long()  # 任一位非pad即有效
        else:
            # input_ids 是 item_id: 先查表得到每个 item 的 D 个 token
            T_item = L
            flat_ids = input_ids.reshape(-1)                                  # [B*L]
            digit_tokens = table.index_select(0, flat_ids).view(B, T_item, D) # [B, T, D]
            if attention_mask.shape[1] == L * D:
                item_mask = attention_mask.view(B, T_item, D).any(dim=2).long()
            else:
                item_mask = attention_mask  # 已经是 item 级别

        return digit_tokens, item_mask  # [B, T, D], [B, T]

    def forward(self, batch: dict, return_loss=True):
        input_ids: torch.Tensor = batch['input_ids']
        attn_mask: torch.Tensor = batch['attention_mask']

        # 统一成 item 级 token 与 mask
        digit_tokens, item_mask = self._to_digit_tokens_and_mask(input_ids, attn_mask)  # [B,T,D], [B,T]

        # token → embedding，按 D 求均值得到 item 表征
        tok_emb = self.gpt2.wte(digit_tokens)       # [B, T, D, H]
        item_emb = tok_emb.mean(dim=2)              # [B, T, H]

        # 过 GPT-2
        outputs = self.gpt2(inputs_embeds=item_emb, attention_mask=item_mask)

        # D 个预测头得到 position 表征（仍保留你原来的结构）
        states = outputs.last_hidden_state          # [B, T, H]
        heads_out = [self.pred_heads[i](states).unsqueeze(2) for i in range(self.n_pred_head)]
        final_states = torch.cat(heads_out, dim=2)  # [B, T, D, H]
        outputs.final_states = final_states

        if not return_loss:
            return outputs

        # ===== 关键修改：把 (T,D) 展平成 token 时间步，直接 DV 分类 =====
        D, V = self.n_pred_head, self.codebook_size
        B, T, _, H = final_states.shape

        # 归一化
        states_pos = F.normalize(final_states, dim=-1, eps=1e-6).reshape(B, T*D, H)   # [B, T*D, H]
        token_emb = self.gpt2.wte.weight[1:self.tokenizer.eos_token]                  # [D*V, H]
        token_emb = F.normalize(token_emb, dim=-1, eps=1e-6)

        # logits: 每个位置与整张 token 词表的相似度
        token_logits = (states_pos @ token_emb.T) / max(self.temperature, 1e-6)       # [B, T*D, D*V]

        # 目标：展平后的“下一 token”标签（与你 tokenizer 的构造一致）
        targets = batch['labels']                                                     # [B, T*D]
        # 忽略 EOS
        targets = targets.masked_fill(targets == self.tokenizer.eos_token, -100)
        # 全局 token 1..D*V → 索引 0..D*V-1（PAD/ignored 变成 -100）
        targets = targets - 1
        targets = targets.masked_fill(targets < 0, -100)

        # 计算 CE（展平）
        loss = self.loss_fct(token_logits.reshape(-1, D*V), targets.reshape(-1))
        # 可选：防 NaN 保险
        if torch.isnan(loss):
            # 打印一行用于排查（只在普通 print；不会卡住训练）
            print("[WARN] NaN loss: valid_ratio=",
                float((targets != -100).float().mean().item()))
            loss = torch.zeros((), device=token_logits.device, dtype=token_logits.dtype)

        outputs.loss = loss
        return outputs




    @torch.no_grad()
    def generate(self, batch, n_return_sequences=1):
        # 前向拿到 final_states: [B,T,D,H]
        out = self.forward(batch, return_loss=False)
        final_states = F.normalize(out.final_states, dim=-1)  # [B,T,D,H]

        # 用 item 级别 mask 找每个样本最后一个位置
        input_ids, attn_mask = batch['input_ids'], batch['attention_mask']
        digit_tokens, item_mask = self._to_digit_tokens_and_mask(input_ids, attn_mask)  # 复用函数
        last_pos = (item_mask.sum(dim=1) - 1).clamp(min=0)   # [B]
        B, T, D, H = final_states.shape
        idx = last_pos.view(B, 1, 1, 1).expand(-1, 1, D, H)  # [B,1,D,H]
        states = final_states.gather(dim=1, index=idx).squeeze(1)  # [B,D,H]

        # 逐 digit 的 logits: [B,D,V]
        V = self.codebook_size
        token_emb = self.gpt2.wte.weight[1:self.tokenizer.eos_token]
        token_emb = F.normalize(token_emb, dim=-1)
        token_chunks = torch.chunk(token_emb, D, dim=0)
        logits = [torch.matmul(states[:, i, :], token_chunks[i].T) / self.temperature for i in range(D)]
        logits = [F.log_softmax(l, dim=-1) for l in logits]  # D 个 [B,V]

        # 物品打分（避免巨型 expand）：按 digit gather 后求和
        # 候选物品 ID：0..n_items-1（若有稀疏集合，可换成 tokenizer.valid_item_ids）
        N = int(getattr(self.tokenizer, 'n_items', self.item_id2tokens.size(0)))
        code_idx = self.item_id2tokens[:N, :]  # [N,D] 全局 token id
        scores = []
        for i in range(D):
            idx_i = (code_idx[:, i] - (i * V + 1)).clamp(0, V - 1)   # [N] → digit 局部 0..V-1
            scores_i = logits[i].index_select(1, idx_i)              # [B,N]
            # 若某物品该digit无code（全0/非法），这里可额外做mask为 -inf（可选）
            scores.append(scores_i)
        item_scores = torch.stack(scores, dim=0).sum(0)              # [B,N]

        topk_pos = item_scores.topk(k=n_return_sequences, dim=-1).indices  # [B,K]
        return topk_pos  # 物品ID
