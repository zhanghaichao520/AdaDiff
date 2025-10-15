import torch.nn as nn


class AbstractModel(nn.Module):
    def __init__(self, config: dict):
        super(AbstractModel, self).__init__()
        self.config = config

    @property
    def n_parameters(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f'Total number of trainable parameters: {total_params}'

    def forward(self, *args, **kwargs):
        raise NotImplementedError('forward method must be implemented.')

    def generate(self, *args, **kwargs):
        raise NotImplementedError('generate method must be implemented.')


