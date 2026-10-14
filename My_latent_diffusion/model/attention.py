import torch
from torch.nn import functional as F
from torch import nn
import math

class SelfAttention(nn.Module):
    def __init__(self, N_heads: int, d_embed : int, in_proj_bias=True, out_proj_bias=True):
        super(SelfAttention, self).__init__()
        
        self.in_proj = nn.Linear(d_embed, 3*d_embed, bias=in_proj_bias)
        
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        
        self.n_head = N_heads
        self.d_embed = d_embed // N_heads
        
    def forward(self, x, causal_mask=False):
        input_shape = x.shape
        
        batch_size, sequence_length, d_embed = input_shape
        
        intermim_shape = (batch_size, sequence_length, self.n_head ,self.d_embed)

        # batch, seq_en, dim -> batch, seq_en, dim *3 ->(batch, seq_en, dim) * 3
        q, k, v, = self.in_proj(x).chunk(3, dim=-1)

        # batch, seq_en, dim -> batch, seq_en, H ,Dim/h -> batch, H, seq_len, Dim.H
        q = q.view(intermim_shape).transpose(1,2)
        k = k.view(intermim_shape).transpose(1,2)
        v = v.view(intermim_shape).transpose(1,2)

        # batch, H, seq_len, sqe_len
        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            mask=  torch.ones_like(weight, dtype=torch.bool).triu(diagonal=1)
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.n_head)

        weight = F.softmax(weight, dim=-1)

        output = weight @ v

        output = output.transpose(1, 2)

        output = output.reshape(input_shape)

        output = self.out_proj(output)

        return output

class CrossAttention(nn.Module):
    def __init__(self, N_heads: int, d_embed : int, d_cross:int, in_proj_bias=True, out_proj_bias=True):
        super(CrossAttention, self).__init__()

        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

        self.n_head = N_heads
        self.d_head = d_embed // N_heads

    def forward(self, x, y):
        # x : lantent (batch, seq_lin_Q, dim_q)
        # y : context (batch, seq_len_kv, dim_kv) = batch, 77, 768

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        interim_shape = (batch_size, -1, self.n_head ,self.d_head)

        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = q.view(interim_shape).transpose(1,2)
        k = k.view(interim_shape).transpose(1,2)
        v = v.view(interim_shape).transpose(1,2)

        weight = q @ k.transpose(-1, -2)

        weight /= math.sqrt(self.n_head)

        weight = F.softmax(weight, dim=-1)

        output = weight @ v

        output =output.transpose(1, 2)

        output = output.view(input_shape)

        output = self.out_proj(output)

        return output
