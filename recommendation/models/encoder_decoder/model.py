# models/encoder_decoder/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
from transformers import T5Config, T5Model

from model import AbstractModel
from tokenizer import AbstractTokenizer


class encoder_decoder(AbstractModel):
    """
    Encoder-Decoder (T5Model) + 分层 codebook 推理：
    - 训练：teacher forcing 的 token-level LM（labels 是展平 token，末位 EOS，pad=-100）
    - 推理：按 digit 做约束 Beam Search（band 限制 + 可选前缀合法性过滤），
           最后把 beam 的 D 位 code 精确还原为 item_id 返回 [B, K]
    """

    def __init__(self, config: dict, tokenizer: AbstractTokenizer):
        super().__init__(config, dataset=None, tokenizer=tokenizer)

        self.codebook_size = int(config['RQ-VAE']['code_book_size'])   # V
        self.n_digit = int(tokenizer.n_digit)                          # D
        self.vocab_size = int(tokenizer.vocab_size)                    # D*V + 2 (PAD=0, EOS=D*V+1)
        self.temperature = float(config.get('temperature', 0.07))

        d_model = int(config.get('n_embd', 448))
        n_layer = int(config.get('n_layer', 2))
        n_head = int(config.get('n_head', 8))
        dropout = float(config.get('dropout', 0.1))

        # 共享 token embedding & LM head（绑权重）
        self.wte = nn.Embedding(self.vocab_size, d_model)
        self.lm_head = nn.Linear(d_model, self.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

        # 学习到的 BOS / SEP 向量
        self.bos_embed = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.sep_embed = nn.Parameter(torch.randn(1, d_model) * 0.02)

        # 轻量 T5（同一模型内含 encoder/decoder）
        t5cfg = T5Config(
            d_model=d_model, d_ff=4 * d_model, num_layers=n_layer, num_heads=n_head,
            dropout_rate=dropout, is_encoder_decoder=True
        )
        self.t5 = T5Model(t5cfg)

        # item_id -> [D] 全局 token id 查表（评测时还原 item 用）
        self.item_id2tokens = tokenizer.item_id2tokens_tensor  # [N, D]

        self.eos_token = tokenizer.eos_token
        self.pad_token = 0
        self.ignored_label = tokenizer.ignored_label
        # 这里不再直接用整表 CE；在 forward 里做 band-CE
        # self.loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignored_label)
         # === beam 与前缀约束所需的缓存 ===
        self.V = self.codebook_size
        self.D = self.n_digit

        # [N, D] 全局 → local(0..V-1)
        self.codes_global = self.item_id2tokens.clone().long()               # [N, D]
        self.codes_local = ((self.codes_global - 1) % self.V).long()         # [N, D]
        self.N = self.codes_local.size(0)

        # 哪些 item 有效（一般全 True；若你想屏蔽冷门/无效条目可在外面改这个 mask）
        self.valid_items_mask = torch.ones(self.N, dtype=torch.bool)

        # 空前缀下一位可取的 local 值集合（unique）
        uniq0 = torch.unique(self.codes_local[:, 0])
        self.allowed_step0 = torch.zeros(self.V, dtype=torch.bool)
        self.allowed_step0[uniq0] = True

        # local-code -> item_id 的快速映射（Python dict，体量 N，放 CPU 即可）
        self.local_tuple_to_item = {tuple(row.tolist()): int(i)
                                    for i, row in enumerate(self.codes_local.cpu())}

        # beam 宽度（可从 config 里取，没有就默认 10）
        self.beam_width = int(self.config.get('top_k_for_generation',
                                              self.config.get('beam_width', 10)))


    # -------------------- 编码 / 解码（训练） --------------------
    def _encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        编码器输入：先做词嵌入；然后按每个 item 的 D 个 code 后面插入 1 个 SEP，
        同步扩展 attention_mask；最后送入 encoder。
        返回：encoder_hidden_state 以及扩展后的 encoder attention mask
        """
        B, L = input_ids.shape
        H = self.wte.embedding_dim
        D = self.n_digit

        enc_emb = self.wte(input_ids)  # [B, L, H]

        # 计算每条序列的 item 数（假定 L 是 D 的整数倍；若不是，向下取整使用前 L' = items*D 个）
        item_cnt = L // D
        if item_cnt > 0:
            # 截到整 item 的长度
            trim_len = item_cnt * D
            enc_emb = enc_emb[:, :trim_len, :]                      # [B, items*D, H]
            enc_msk = attention_mask[:, :trim_len]                  # [B, items*D]

            # [B, items, D, H]
            enc_emb = enc_emb.view(B, item_cnt, D, H)
            enc_msk_3d = enc_msk.view(B, item_cnt, D)

            # 构造 SEP 并拼接到每个 item 末尾
            sep = self.sep_embed.view(1, 1, 1, H).expand(B, item_cnt, 1, H)  # [B, items, 1, H]
            enc_emb = torch.cat([enc_emb, sep], dim=2).reshape(B, item_cnt*(D+1), H)

            # SEP 的 mask：复用该 item 最后一个 code 的 mask（对 padding item 将得到 0）
            sep_m = enc_msk_3d[:, :, [-1]]
            enc_msk = torch.cat([enc_msk_3d, sep_m], dim=2).reshape(B, item_cnt*(D+1))
        else:
            # 极端空序列兜底
            enc_msk = attention_mask

        out = self.t5.encoder(inputs_embeds=enc_emb, attention_mask=enc_msk)
        return out.last_hidden_state, enc_msk  # [B, Lenc', H], [B, Lenc']



    def _teacher_force_decode(self, enc_out, enc_mask, labels):
        """
        labels: [B, L]（next-token，含 -100/EOS）
        构造 decoder_inputs = [BOS] + embed(labels[:-1]) → 输出对齐 labels
        """
        B, L = labels.shape
        y = labels.clone()
        y[y == self.ignored_label] = self.pad_token
        y_emb = self.wte(y)  # [B, L, H]

        bos = self.bos_embed.unsqueeze(0).expand(B, 1, -1)
        dec_inputs = torch.cat([bos, y_emb[:, :-1, :]], dim=1)  # [B, L, H]

        dec_mask = torch.cat(
            [torch.ones(B, 1, device=labels.device, dtype=enc_mask.dtype),
            (labels[:, :-1] != self.ignored_label).to(enc_mask.dtype)], dim=1)  # [B, L]

        dec_out = self.t5.decoder(
            inputs_embeds=dec_inputs,
            attention_mask=dec_mask,
            encoder_hidden_states=enc_out,
            encoder_attention_mask=enc_mask,
            use_cache=False,
        ).last_hidden_state  # [B, L, H]

        logits = self.lm_head(dec_out)  # [B, L, vocab]
        return logits
    
    def _local_to_global(self, prefixes_local: torch.Tensor) -> torch.Tensor:
        """
        prefixes_local: [..., L]，每位 ∈[0..V-1]
        返回同形状的全局 token id：global = local + 1 + idx*V
        """
        if prefixes_local.numel() == 0:
            return prefixes_local
        L = prefixes_local.size(-1)
        device = prefixes_local.device
        offs = (torch.arange(L, device=device) * self.V + 1).view(*([1] * (prefixes_local.dim() - 1)), L)
        return prefixes_local + offs

    # -------------------- 前缀合法性（基于 item_id2tokens） --------------------
    def _allowed_next_mask(self, prefixes_local: torch.Tensor, step: int) -> torch.Tensor:
        """
        给定若干条前缀（local 码，长度=step），返回每条前缀在下一位可取的 local 值集合。
        prefixes_local: [M, step]（step=0 时可给 shape [M, 0] 的空张量）
        返回: Bool 掩码 [M, V]，True 表示该前缀下一位允许取到该 local 值（0..V-1）。
        """
        M = prefixes_local.size(0)
        device = prefixes_local.device

        codes_local = self.codes_local.to(device)           # [N, D]
        valid_items = self.valid_items_mask.to(device)      # [N]

        # step==0：所有 item 在第 0 位出现过的 local 值集合，直接广播
        if step == 0:
            allowed0 = self.allowed_step0.to(device)        # [V] bool
            return allowed0.unsqueeze(0).expand(M, -1)      # [M, V]

        # 与每条前缀匹配的 item（按 step 位前缀完全相等）
        # match: [M, N] (bool)
        match = (codes_local[:, :step].unsqueeze(0) == prefixes_local.unsqueeze(1)).all(dim=-1)
        match = match & valid_items.unsqueeze(0)

        # 取出所有 item 的下一位 local 值并 one-hot（注意必须是浮点，matmul 不支持 bool/int）
        next_local = codes_local[:, step].clamp(min=0, max=self.V - 1)           # [N]
        one_hot_vals = torch.nn.functional.one_hot(next_local, num_classes=self.V).to(torch.float32)  # [N, V]

        # 用浮点做矩阵乘法，>0 即表示该前缀下至少存在一个可达的下一位取值
        allowed_count = match.to(torch.float32) @ one_hot_vals                   # [M, V], float
        allowed = allowed_count > 0                                             # [M, V] bool
        return allowed

    def forward(self, batch: dict, return_loss: bool = True):
        inp: torch.Tensor = batch['input_ids']         # [B, Lenc]（展平 token）
        msk: torch.Tensor = batch['attention_mask']    # [B, Lenc]
        enc_out, enc_mask = self._encode(inp, msk)

        if not return_loss:
            from types import SimpleNamespace
            return SimpleNamespace(encoder_hidden_states=enc_out, attention_mask=enc_mask)

        labels: torch.Tensor = batch['labels']         # [B, Ldec]
        logits = self._teacher_force_decode(enc_out, enc_mask, labels)  # [B, L, vocab]

        # ====== 关键：band-CE（只在所属 digit 的 V 个类里做 CE；忽略 EOS 与 ignore_index）======
        B, L, Vtot = logits.shape
        V = self.codebook_size
        flat_logits = logits.reshape(-1, Vtot)        # [B*L, vocab]
        flat_labels = labels.reshape(-1)              # [B*L]

        valid = (flat_labels != self.ignored_label) & (flat_labels != self.eos_token)
        if valid.any():
            y = flat_labels[valid]                    # [N]
            # 当前标签属于哪个 digit（基于全局 token 1..D*V 的区间）
            digit_start = ((y - 1) // V) * V + 1      # [N] 每个样本对应 band 的起始全局 id
            local_y = (y - digit_start).clamp(min=0, max=V-1).long()  # [N] ∈[0..V-1]

            # 取出每个样本对应 band 的 V 列 logits
            # 构造列索引：每行是 [start, start+1, ..., start+V-1]
            base = torch.arange(V, device=flat_logits.device).unsqueeze(0).expand(y.size(0), -1)  # [N, V]
            band_cols = base + digit_start.unsqueeze(1)                                            # [N, V]

            band_logits = flat_logits[valid].gather(1, band_cols)  # [N, V]
            loss = F.cross_entropy(band_logits, local_y)
        else:
            loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        # ===========================================================================================

        from types import SimpleNamespace
        return SimpleNamespace(loss=loss)
    # -------------------- 推理：约束 Beam Search → item_id --------------------
    
    @torch.no_grad()
    def generate(self, batch, n_return_sequences: int = 1):
        """
        约束 Beam Search（band 限制 + 前缀合法性）生成 D 位 code，
        再把完整 local 码还原为 item_id，返回 [B, K]。
        """
        inp: torch.Tensor = batch['input_ids']
        msk: torch.Tensor = batch['attention_mask']
        device = inp.device
        B = inp.size(0)

        # 编码端（带 SEP）
        enc_out, enc_mask = self._encode(inp, msk)

        # 初始 beam（每个样本 1 条空前缀）
        beam_K = max(1, self.beam_width)
        prefixes = torch.empty(B, 1, 0, dtype=torch.long, device=device)  # [B, t, 0]
        cum_logp = torch.zeros(B, 1, device=device)                       # [B, t]

        H = self.wte.embedding_dim
        bos = self.bos_embed.unsqueeze(0)                                  # [1, H]

        for step in range(self.D):
            t = prefixes.size(1)  # 当前 beam 条数（每个样本）
            # 1) 构造 decoder 输入：B*t 条序列
            if step == 0:
                dec_emb = bos.expand(B * t, 1, H)                          # 只喂 BOS
            else:
                # local → global → embed，并在开头拼 BOS
                prev_glob = self._local_to_global(prefixes.view(B * t, step))          # [B*t, step]
                prev_emb = self.wte(prev_glob)                                           # [B*t, step, H]
                bos_emb = bos.expand(B * t, 1, H)
                dec_emb = torch.cat([bos_emb, prev_emb], dim=1)                          # [B*t, step+1, H]

            # 2) 过 decoder，拿到下一位 logits
            dec_out = self.t5.decoder(
                inputs_embeds=dec_emb,
                attention_mask=torch.ones(dec_emb.size(0), dec_emb.size(1), device=device, dtype=enc_mask.dtype),
                encoder_hidden_states=enc_out.repeat_interleave(t, dim=0),
                encoder_attention_mask=enc_mask.repeat_interleave(t, dim=0),
                use_cache=False,
            ).last_hidden_state
            last_h = dec_out[:, -1, :]                                  # [B*t, H]
            vocab_logits = self.lm_head(last_h)                         # [B*t, vocab]
            logp_full = F.log_softmax(vocab_logits / max(self.temperature, 1e-6), dim=-1)

            # 3) band 限制：只取该 digit 的 [start:end)
            start = step * self.V + 1
            end = (step + 1) * self.V + 1
            logp_band = logp_full[:, start:end]                          # [B*t, V]

            # 4) 前缀合法性：对每条 beam 给出下一位允许的 local 值
            allow = self._allowed_next_mask(
                prefixes.view(B * t, step) if step > 0 else prefixes.new_zeros(B * t, 0),
                step=step
            )                                                            # [B*t, V]
            logp_band = logp_band.masked_fill(~allow, float("-inf"))

            # 5) 把 cum_logp 累加到每个候选上，按样本选 top-k
            cand_logp = logp_band + cum_logp.view(B * t, 1).expand(-1, self.V)  # [B*t, V]
            cand_logp = cand_logp.view(B, t * self.V)                            # [B, t*V]
            topk_val, topk_idx = torch.topk(cand_logp, k=min(beam_K, t * self.V), dim=1)  # [B, k]

            # 6) 反推选中了哪条旧 beam + 哪个 local 值
            src_beam = (topk_idx // self.V)                                      # [B, k]
            next_local = (topk_idx % self.V).long()                              # [B, k]

            # 7) 组装新的前缀与分数
            if step == 0:
                new_prefixes = next_local.unsqueeze(-1)                          # [B, k, 1]
            else:
                take = torch.gather(
                    prefixes, dim=1,
                    index=src_beam.unsqueeze(-1).expand(B, topk_val.size(1), step)
                )                                                                # [B, k, step]
                new_prefixes = torch.cat([take, next_local.unsqueeze(-1)], dim=-1)  # [B, k, step+1]

            prefixes, cum_logp = new_prefixes, topk_val                          # 更新 beam

        # === 结束：把每条完整 local 序列映射成 item_id，按分数从大到小取前 K ===
        K = n_return_sequences
        out_ids = torch.full((B, K), -1, dtype=torch.long, device=device)

        # 从 CPU 的 dict 取（beam 数很小，CPU 查字典最稳健）
        for b in range(B):
            seqs = prefixes[b].detach().cpu()              # [t, D]
            scores = cum_logp[b].detach().cpu()            # [t]
            order = torch.argsort(scores, descending=True)
            filled = 0
            used = set()
            for idx in order.tolist():
                key = tuple(seqs[idx].tolist())
                item = self.local_tuple_to_item.get(key, None)
                if item is None or item in used:
                    continue
                out_ids[b, filled] = item
                used.add(item)
                filled += 1
                if filled >= K:
                    break

            # 兜底：若 beam 里有重复/无映射导致不够 K，用“全量打分”的老办法补齐
            if filled < K:
                # 用每位 logp 对所有 item 聚合（和你之前贪心版同样的做法）
                codes = self.codes_global.to(device)       # [N, D]（全局 token）
                scores_items = None
                # 为此需要各位 logp（beam search阶段最后一轮没有保存逐位 logp，这里简化为重新过一遍贪心位的 logp）
                # 简约实现：直接用最后一步 dec 的 logp_full 再跑一轮逐位（开销小，D 很小）
                # ——如果你已有逐位 logp，替换掉这里即可。
                # 为了简单，我们用生成好的 prefixes 里的第一条，重算逐位 logp 并聚合：
                seq_local = prefixes[b, 0]                 # [D]
                # 逐位重算 logp
                logps = []
                past = None
                cur = self.bos_embed.unsqueeze(0).to(device)   # [1, H]
                cur = cur.unsqueeze(1)                         # [1,1,H]
                for i in range(self.D):
                    dec = self.t5.decoder(
                        inputs_embeds=cur,
                        attention_mask=torch.ones(1, cur.size(1), device=device, dtype=enc_mask.dtype),
                        encoder_hidden_states=enc_out[b:b+1],
                        encoder_attention_mask=enc_mask[b:b+1],
                        use_cache=True,
                        past_key_values=past,
                    )
                    last_h = dec.last_hidden_state[:, -1, :]
                    past = dec.past_key_values
                    lp = F.log_softmax(self.lm_head(last_h) / max(self.temperature, 1e-6), dim=-1)[0]  # [vocab]
                    logps.append(lp)
                    gtok = (seq_local[i] + 1 + i * self.V).view(1)  # 选用 beam 第一条的 token 续上，避免额外搜索
                    cur = self.wte(gtok).unsqueeze(0)               # [1,1,H]

                scores_items = 0
                for i in range(self.D):
                    idx = codes[:, i].clamp(1, self.vocab_size - 1)     # [N]
                    s_i = logps[i].gather(0, idx)                        # [N]
                    scores_items = s_i if i == 0 else (scores_items + s_i)
                # 把未填满的位置补齐（排除已填过的）
                mask_fill = torch.ones(self.N, dtype=torch.bool, device=device)
                if used:
                    used_idx = torch.tensor(sorted(list(used)), device=device, dtype=torch.long)
                    mask_fill[used_idx] = False
                fill_idx = scores_items.masked_fill(~mask_fill, float('-inf')).topk(K - filled).indices
                out_ids[b, filled:K] = fill_idx

        return out_ids


