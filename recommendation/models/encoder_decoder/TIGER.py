# models/tiger.py

from typing import Any, Dict

import torch
import transformers
from ..abstract_model import AbstractModel # 假設 abstract_model 在同級目錄

# 從 transformers 庫直接導入 T5 模型和配置
T5ForConditionalGeneration = transformers.T5ForConditionalGeneration
T5Config = transformers.T5Config


class TIGER(AbstractModel):
  """
  一個通用的、由配置驅動的 T5 模型封裝，用於 GenRec。
  """

  def __init__(self, config: Dict[str, Any]):
    """
    僅透過 config 字典初始化模型。
    所有必要的參數（模型結構、詞表大小等）都應包含在 config 中。

    Args:
        config (Dict[str, Any]): 包含所有超參數的字典。
    """
    # 注意：不再需要 dataset 和 tokenizer 參數
    super().__init__(config)
    
    # 1. 從 config 中提取模型結構參數和 token 相關參數
    model_params = config['model_params']
    token_params = config['token_params']

    # 2. 使用字典解包來創建 T5Config，更簡潔且可擴展
    t5config = T5Config(
        **model_params,        # 解包模型結構參數
        **token_params,        # 解包詞表、pad_id 等參數
        decoder_start_token_id=0 # T5 通常需要這個
    )

    # 實例化 T5 模型
    self.t5 = T5ForConditionalGeneration(config=t5config)

  @property
  def n_parameters(self) -> str:
    """計算並返回模型的可訓練參數數量。"""
    num_params = lambda ps: sum(p.numel() for p in ps if p.requires_grad)
    total_params = num_params(self.parameters())
    emb_params = num_params(self.t5.get_input_embeddings().parameters())
    return (
        f'# Embedding parameters: {emb_params:,}\n'
        f'# Non-embedding parameters: {total_params - emb_params:,}\n'
        f'# Total trainable parameters: {total_params:,}\n'
    )

  def forward(self, batch: Dict[str, torch.Tensor]) -> transformers.modeling_outputs.BaseModelOutput:
    """
    執行標準的前向傳播（用於訓練）。
    直接將 batch 字典解包後傳給底層的 T5 模型。
    """
    return self.t5(**batch)

  def generate(self, **kwargs: Any) -> torch.Tensor:
    """
    執行生成（用於推斷/評估）。
    這是 Hugging Face generate 方法的一個簡單封裝。
    所有生成所需的參數 (如 input_ids, num_beams) 都透過 kwargs 傳入。
    """
    # 3. 直接將所有關鍵字參數傳給底層的 generate 方法
    return self.t5.generate(**kwargs)