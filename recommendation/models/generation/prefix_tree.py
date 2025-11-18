# recommendation/models/generation/prefix_tree.py

from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import logging

logger = logging.getLogger(__name__)


class Trie:
    """
    Hash 前缀树版本（性能优化版）：

    - 内部用一个 dict: prefix(tuple[int]) -> List[allowed_next_token_id]
    - 支持前缀查询缓存，减少重复计算：
        - 同一个 prefix 在一次生成过程会被多次访问（尤其是 beam search）
        - 缓存可以显著减少 Python 层开销

    对外接口保持不变：
    - build_trie_from_codebook(...) 返回的就是这个 Trie
    - Trie.get_allowed_next_tokens(batch_id, input_ids) 可直接给
      transformers.generate(prefix_allowed_tokens_fn=...) 使用
    """

    def __init__(
        self,
        eos_token_id: Optional[int] = None,
        pad_token_id: int = 0,
        skip_bos: int = 1,
        use_cache: bool = True,
        max_cache_size: int = 100_000,
    ) -> None:
        """
        Args:
            eos_token_id: 生成结束用的 EOS id（TIGER 用 config['token_params']['eos_token_id']）
            pad_token_id: code 序列中的 padding id（通常是 0，会在构建时滤掉）
            skip_bos: 在 get_allowed_next_tokens 中，从 input_ids 前面跳过多少个 token。
                      对 TIGER 而言，decoder_start_token_id=0，所以一般为 1。
            use_cache: 是否启用查询缓存（推荐 True）
            max_cache_size: 缓存中允许的最大不同前缀数量，超过后自动清空缓存
        """
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.skip_bos = skip_bos

        # prefix(tuple[int]) -> List[int]
        self._table: Dict[Tuple[int, ...], List[int]] = {}

        # 查询缓存：prefix(tuple[int]) -> List[int]
        self._use_cache = use_cache
        self._cache: Dict[Tuple[int, ...], List[int]] = {}
        self._max_cache_size = max_cache_size

    # ----------------------------------------------------------------------
    # 构建阶段：从所有合法 code 序列里构建 prefix -> next_tokens 映射
    # ----------------------------------------------------------------------
    def bulk_insert(self, sequences: Iterable[Sequence[int]]) -> None:
        """
        直接从所有合法的 code 序列构建前缀表。

        Args:
            sequences: 形如 [[c1,c2,...,cL], ...]，每个都是一个完整的 code。
                       注意：这里的序列不包含 decoder_start_token_id，只是纯 code。
        """
        table: Dict[Tuple[int, ...], set[int]] = {}
        n_seq = 0

        for seq in sequences:
            # 转成 int，并去掉 pad
            seq = [int(t) for t in seq if t != self.pad_token_id]
            if not seq:
                continue

            n_seq += 1

            # extended: seq (+ eos)，保证完整 code 之后还能结束
            if self.eos_token_id is not None:
                extended = seq + [self.eos_token_id]
            else:
                extended = seq

            # 对 extended 的每个前缀 extended[:i]，记录下一个 token extended[i]
            # i 可以是 0，表示还没生成任何 code，这样 prefix=() 也会有一组起始 token
            for i in range(len(extended)):
                prefix = tuple(extended[:i])
                next_token = extended[i]

                if prefix not in table:
                    table[prefix] = set()
                table[prefix].add(next_token)

        # set -> list，节省一点内存
        self._table = {k: list(v) for k, v in table.items()}

        # 构建完清空缓存
        self._cache.clear()

        logger.info(
            f"[Trie(Hash+Cache)] Built from {n_seq} sequences, "
            f"unique prefixes: {len(self._table)}, "
            f"use_cache={self._use_cache}, max_cache_size={self._max_cache_size}"
        )

    # ----------------------------------------------------------------------
    # 查询阶段：给 HF generate / logits_processor 用的接口
    # ----------------------------------------------------------------------
    @staticmethod
    def _ids_to_prefix_tuple(ids, skip_bos: int) -> Tuple[int, ...]:
        """
        将当前 decoder input_ids 转成 prefix tuple：
        - ids: 1D tensor/list，如 [decoder_start, c1, c2, ...]
        - skip_bos: 跳过前面的 BOS / decoder_start token 数
        """
        # transformers 通常会给一个 1D LongTensor，这里统一成 Python list
        if hasattr(ids, "tolist"):
            ids = ids.tolist()

        # 只取 code 部分作为前缀（不再过滤 pad，因为 decoder 侧通常没 pad）
        # 这里假设 pad_token 还没出现在 decoder_prefix 里，若你后面确实会 pad，可以再加过滤。
        return tuple(int(t) for t in ids[skip_bos:])

    def _lookup(self, prefix: Tuple[int, ...]) -> List[int]:
        """
        内部查表 + 缓存逻辑。
        """
        # 1) 先查缓存
        if self._use_cache and prefix in self._cache:
            return self._cache[prefix]

        # 2) 查主表
        if prefix in self._table:
            allowed = self._table[prefix]
        else:
            # 不在主表：非法前缀，尽量只允许 eos 结束
            if self.eos_token_id is not None:
                allowed = [self.eos_token_id]
            else:
                # 实在不行就返回空（不推荐走到这里）
                allowed = []

        # 3) 写入缓存（如果开启）
        if self._use_cache:
            # 简单策略：缓存太大就清空一次
            if len(self._cache) >= self._max_cache_size:
                self._cache.clear()
            self._cache[prefix] = allowed

        return allowed

    def get_allowed_next_tokens(self, batch_id: int, input_ids) -> List[int]:
        """
        核心接口：给 Transformers generate(prefix_allowed_tokens_fn=...) 用。

        Args:
            batch_id: HF 的 batch 索引（这里用不到）
            input_ids: 当前 decoder 已经生成的 token 序列（1D）

        Returns:
            一组允许的下一个 token id 列表。
        """
        prefix = self._ids_to_prefix_tuple(input_ids, self.skip_bos)
        return self._lookup(prefix)


# ----------------------------------------------------------------------
# 工厂函数：保持原来的调用接口不变
# ----------------------------------------------------------------------
def build_trie_from_codebook(
    token_sequences: Iterable[Sequence[int]],
    eos_token_id: Optional[int] = None,
) -> Trie:
    """
    从 codebook 序列构建 Hash 前缀树 Trie（带缓存）。

    Args:
        token_sequences: 一般是 item_to_code_map.values()，每个是一个 code 序列
        eos_token_id: 生成结束 token id

    Returns:
        Trie 实例（Hash+Cache 版）
    """
    # 假设：
    # - code 中的 padding 是 0
    # - decoder_start_token_id 占用了第 0 个位置，真正的 code 从位置 1 开始
    trie = Trie(
        eos_token_id=eos_token_id,
        pad_token_id=0,
        skip_bos=1,
        use_cache=True,
        max_cache_size=100_000,
    )
    trie.bulk_insert(token_sequences)
    return trie
