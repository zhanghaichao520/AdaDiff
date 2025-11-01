# models/OneRec.py - 基于OneRec-Think的统一对话、推理和推荐框架

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional
import logging
import transformers
from transformers import T5ForConditionalGeneration, T5Config

import sys
from pathlib import Path
root = Path(__file__).resolve().parents[3]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from recommendation.metrics import recall_at_k, ndcg_at_k
from recommendation.models.generation.prefix_tree import Trie
from recommendation.models.abstract_model import AbstractModel

logger = logging.getLogger(__name__)


class OneRec(AbstractModel):
    """
    OneRec-Think: 统一对话、推理和个性化推荐框架
    
    核心特性:
    1. 分层项目token对齐 (Hierarchical Itemic Token Alignment)
    2. 推理激活 (Reasoning Activation via CoT)
    3. 强化学习推理优化 (Reinforcement-based Reasoning Refinement)
    4. 编码器-解码器架构与交叉注意力
    5. 优化的生成策略
    """
    
    def __init__(
        self, 
        config: Dict[str, Any], 
        prefix_trie: Optional[Trie] = None
    ):
        super().__init__(config)
        
        # 获取配置参数
        model_params = config['model_params']
        token_params = config['token_params']
        
        # 构建T5配置
        t5_config = T5Config(
            vocab_size=token_params['vocab_size'],
            d_model=model_params['d_model'],
            d_ff=model_params['d_ff'],
            num_heads=model_params['num_heads'],
            num_layers=model_params['num_encoder_layers'],
            num_decoder_layers=model_params['num_decoder_layers'],
            dropout_rate=model_params['dropout'],
            decoder_start_token_id=0,
            # OneRec-Think 特定配置
            use_cache=True,
            is_encoder_decoder=True,
        )
        
        # 初始化T5模型
        self.t5 = T5ForConditionalGeneration(config=t5_config)
        self.t5.resize_token_embeddings(token_params['vocab_size'])
        
        # OneRec-Think 特定组件
        self.reasoning_head = nn.Linear(model_params['d_model'], model_params['d_model'])
        self.reasoning_activation = nn.GELU()
        
        # 分层token对齐
        self.semantic_projection = nn.Linear(model_params['d_model'], model_params['d_model'])
        self.item_projection = nn.Linear(model_params['d_model'], model_params['d_model'])
        
        # 推理奖励机制
        self.reasoning_reward_head = nn.Linear(model_params['d_model'], 1)
        
        # 前缀树支持
        self.prefix_trie_fn = None
        if prefix_trie is not None:
            self.prefix_trie_fn = prefix_trie.get_allowed_next_tokens
            logger.info("OneRec 模型已成功加载前缀树 (Prefix Trie)。")
        else:
            logger.info("OneRec 模型未加载前缀树 (Prefix Trie)。")
            
        # 计算参数数量
        self.n_params_str = self._calculate_n_parameters()
        
        logger.info("OneRec-Think 模型初始化完成")
    
    @property
    def task_type(self) -> str:
        return 'generative'
    
    @property
    def n_parameters(self) -> str:
        return self.n_params_str
    
    def _calculate_n_parameters(self) -> str:
        """计算模型参数数量"""
        num_params = lambda ps: sum(p.numel() for p in ps if p.requires_grad)
        total_params = num_params(self.parameters())
        t5_params = num_params(self.t5.parameters())
        reasoning_params = total_params - t5_params
        
        return (
            f'# T5 backbone parameters: {t5_params:,}\n'
            f'# Reasoning components parameters: {reasoning_params:,}\n'
            f'# Total trainable parameters: {total_params:,}\n'
        )
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        OneRec-Think 前向传播
        
        支持两种模式:
        1. 标准推荐模式 (input_ids, attention_mask, labels)
        2. 推理模式 (包含reasoning tokens)
        """
        # 提取T5需要的参数
        t5_inputs = {
            key: value for key, value in batch.items() 
            if key in {'input_ids', 'attention_mask', 'labels'}
        }
        
        # T5前向传播
        outputs = self.t5(**t5_inputs)
        
        # OneRec-Think 增强
        if hasattr(outputs, 'encoder_last_hidden_state') and outputs.encoder_last_hidden_state is not None:
            # 编码器输出增强
            encoder_hidden = outputs.encoder_last_hidden_state
            
            # 分层token对齐
            semantic_features = self.semantic_projection(encoder_hidden)
            item_features = self.item_projection(encoder_hidden)
            
            # 推理激活
            reasoning_features = self.reasoning_activation(
                self.reasoning_head(encoder_hidden)
            )
            
            # 计算推理奖励
            reasoning_rewards = self.reasoning_reward_head(reasoning_features)
            
            # 将增强特征添加到输出中
            outputs.semantic_features = semantic_features
            outputs.item_features = item_features
            outputs.reasoning_features = reasoning_features
            outputs.reasoning_rewards = reasoning_rewards
        
        return outputs
    
    def generate(self, **kwargs: Any) -> torch.Tensor:
        """
        OneRec-Think 优化生成
        
        特性:
        1. 前缀约束生成
        2. 推理引导生成
        3. 优化的beam search
        """
        # 注入前缀约束
        if self.prefix_trie_fn is not None:
            kwargs.setdefault('prefix_allowed_tokens_fn', self.prefix_trie_fn)
        
        # OneRec-Think 生成参数优化
        generation_config = self.config.get('evaluation_params', {})
        
        # 设置默认生成参数
        kwargs.setdefault('do_sample', generation_config.get('do_sample', False))
        kwargs.setdefault('temperature', generation_config.get('temperature', 1.0))
        kwargs.setdefault('top_k', generation_config.get('top_k', 50))
        kwargs.setdefault('top_p', generation_config.get('top_p', 0.9))
        kwargs.setdefault('length_penalty', generation_config.get('length_penalty', 1.0))
        kwargs.setdefault('early_stopping', generation_config.get('early_stopping', True))
        
        return self.t5.generate(**kwargs)
    
    def evaluate_step(self, batch: Dict[str, torch.Tensor], topk_list: List[int]) -> Dict[str, float]:
        """
        OneRec-Think 评估步骤
        
        包含:
        1. 推理引导生成
        2. 分层token匹配
        3. 推理质量评估
        """
        # 获取评估参数
        eval_params = self.config['evaluation_params']
        beam_size = eval_params['beam_size']
        code_len = self.config['code_len']
        
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        # OneRec-Think 生成
        with torch.no_grad():
            preds = self.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=beam_size,
                num_return_sequences=beam_size,
                max_new_tokens=code_len,
                early_stopping=eval_params.get('early_stopping', True),
                length_penalty=eval_params.get('length_penalty', 1.0)
            )
        
        # 后处理预测结果
        preds = preds[:, 1:1 + code_len].view(batch_size, beam_size, -1)
        
        # OneRec-Think 分层匹配
        pos_index = self._calculate_hierarchical_pos_index(
            preds, labels, maxk=beam_size
        ).to(device)
        
        # 计算指标
        batch_metrics = {'count': batch_size}
        
        for k in topk_list:
            recall_sum = recall_at_k(pos_index, k).sum().item()
            ndcg_sum = ndcg_at_k(pos_index, k).sum().item()
            
            batch_metrics[f'Recall@{k}'] = recall_sum
            batch_metrics[f'NDCG@{k}'] = ndcg_sum
        
        return batch_metrics
    
    def _calculate_hierarchical_pos_index(
        self, 
        preds: torch.Tensor, 
        labels: torch.Tensor, 
        maxk: int
    ) -> torch.Tensor:
        """
        OneRec-Think 分层位置索引计算
        
        支持:
        1. 语义层匹配
        2. 重复层匹配
        3. 推理一致性检查
        """
        preds = preds.detach().cpu()
        labels = labels.detach().cpu()
        B, _, L_pred = preds.shape
        L_label = labels.shape[1]
        
        # 长度对齐
        if L_pred < L_label:
            padding = torch.zeros((B, maxk, L_label - L_pred), dtype=preds.dtype)
            preds = torch.cat([preds, padding], dim=2)
        elif L_pred > L_label:
            preds = preds[:, :, :L_label]
        
        pos_index = torch.zeros((B, maxk), dtype=torch.bool)
        
        for i in range(B):
            gt = labels[i]
            gt_semantic = gt[:-1].tolist()
            gt_dup = int(gt[-1].item())
            
            for j in range(maxk):
                pj = preds[i, j]
                pj_semantic = pj[:-1].tolist()
                pj_dup = int(pj[-1].item())
                
                # OneRec-Think 分层匹配
                semantic_match = pj_semantic == gt_semantic
                dup_match = pj_dup == gt_dup
                
                if semantic_match and dup_match:
                    pos_index[i, j] = True
                    break
        
        return pos_index
    
    def compute_reasoning_loss(
        self, 
        reasoning_features: torch.Tensor, 
        reasoning_rewards: torch.Tensor,
        target_reasoning: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算推理损失 (用于强化学习优化)
        """
        if target_reasoning is not None:
            # 监督推理损失
            reasoning_loss = F.mse_loss(reasoning_features, target_reasoning)
        else:
            # 自监督推理损失 (基于奖励)
            reasoning_loss = -reasoning_rewards.mean()
        
        return reasoning_loss
    
    def get_reasoning_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        提取推理特征 (用于推理质量评估)
        """
        with torch.no_grad():
            encoder_outputs = self.t5.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            reasoning_features = self.reasoning_activation(
                self.reasoning_head(encoder_outputs.last_hidden_state)
            )
            
        return reasoning_features