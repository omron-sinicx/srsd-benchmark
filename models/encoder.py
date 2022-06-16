import torch
from torch import nn


def make_mlp(mlp_configs):
    module_list = list()
    for config in mlp_configs:
        module = nn.__dict__[config['type']](**config['kwargs'])
        module_list.append(module)
    return nn.Sequential(*module_list)


class DimMaxpool(nn.Module):
    def __init__(self, dim=-1, unsqueezes=True):
        super().__init__()
        self.dim = dim
        self.unsqueezes = unsqueezes

    def forward(self, x):
        x, _ = torch.max(x, dim=self.dim)
        return x.unsqueeze(self.dim) if self.unsqueezes else x


class EncoderBlock(nn.Module):
    def __init__(self, mlp1_configs, mlp2_configs):
        super().__init__()
        self.shared_mlp1 = make_mlp(mlp1_configs)
        self.sample_maxpool = DimMaxpool(dim=1)
        self.shared_mlp2 = make_mlp(mlp2_configs)
        self.var_maxpool = DimMaxpool(dim=2)

    def extract_variable_wise_features(self, x, num_samples_per_eq):
        # x: (num_equations, num_samples_per_eq, max_num_vars, input_embed_size)
        z1 = self.shared_mlp1(x)
        # z1: (num_equations, num_samples_per_eq, max_num_vars, mlp1_embed_size)
        #   point-wise features
        z2 = self.sample_maxpool(z1)
        # z2: (num_equations, 1, max_num_vars, mlp1_embed_size)
        #   variable-wise features
        z3 = z2.repeat(1, num_samples_per_eq, 1, 1)
        # z3: (num_equations, num_samples_per_eq, max_num_vars, mlp1_embed_size)
        #   broadcasted variable-wise features
        z4 = torch.cat([z1, z3], dim=-1)
        # z4: (num_equations, num_samples_per_eq, max_num_vars, 2 * mlp1_embed_size)
        #   concatenated representations
        return z4

    def extract_sample_wise_features(self, z4, max_num_vars):
        # z4: (num_equations, num_samples_per_eq, max_num_vars, 2 * mlp1_embed_size)
        z5 = self.shared_mlp2(z4)
        # z5: (num_equations, num_samples_per_eq, max_num_vars, mlp2_embed_size)
        #   feature represetations that capture the characteristics of each equation
        z6 = self.var_maxpool(z5)
        # z6: (num_equations, num_samples_per_eq, 1, mlp2_embed_size)
        #   sample-wise features
        z7 = z6.repeat(1, 1, max_num_vars, 1)
        # z7: (num_equations, num_samples_per_eq, max_num_vars, mlp2_embed_size)
        #   broadcasted sample-wise features
        z8 = torch.cat([z5, z7], dim=-1)
        # z8: (num_equations, num_samples_per_eq, max_num_vars, 2 * mlp2_embed_size)
        #   concatenated representations
        return z8
        
    def forward(self, x):
        # x: (num_equations, num_samples_per_eq, max_num_vars, input_embed_size)
        # input_embed_size = 1 if x is sample batch from dataset
        num_samples_per_eq, max_num_vars = x.shape[1:3]
        x = self.extract_variable_wise_features(x, num_samples_per_eq)
        # x: (num_equations, num_samples_per_eq, max_num_vars, 2 * mlp1_embed_size)
        x = self.extract_sample_wise_features(x, max_num_vars)
        # x: (num_equations, num_samples_per_eq, max_num_vars, 2 * mlp2_embed_size)
        return x


def build_encoder(encoder_config):
    block_list = list()
    for block_config in encoder_config['block_configs']:
        encoder_block = EncoderBlock(**block_config)
        block_list.append(encoder_block)
    return nn.Sequential(*block_list)
