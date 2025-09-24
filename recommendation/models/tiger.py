# models/tiger.py
import torch
import torch.nn as nn
from transformers import T5Config, T5ForConditionalGeneration
from typing import Dict, Any, Optional

class TIGER(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(TIGER, self).__init__()
        t5config = T5Config(
            num_layers=config['num_layers'],
            num_decoder_layers=config['num_decoder_layers'],
            d_model=config['d_model'],
            d_ff=config['d_ff'],
            num_heads=config['num_heads'],
            d_kv=config['d_kv'],
            dropout_rate=config['dropout_rate'],
            vocab_size=config['vocab_size'],
            pad_token_id=config['pad_token_id'],
            eos_token_id=config['eos_token_id'],
            decoder_start_token_id=config['pad_token_id'],
            feed_forward_proj=config['feed_forward_proj'],
        )
        self.model = T5ForConditionalGeneration(t5config)

    @property
    def n_parameters(self):
        num_params = lambda ps: sum(p.numel() for p in ps if p.requires_grad)
        total_params = num_params(self.parameters())
        emb_params = num_params(self.model.get_input_embeddings().parameters())
        return (
            f'#Embedding parameters: {emb_params}\n'
            f'#Non-embedding parameters: {total_params - emb_params}\n'
            f'#Total trainable parameters: {total_params}\n'
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs.loss, outputs.logits
    
    def generate(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, num_beams: int = 20, **kwargs):
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=num_beams,
            **kwargs
        )
