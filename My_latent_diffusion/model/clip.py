import torch
import torch.nn.functional as F
from torch import nn
from attention import SelfAttention


class CLPEmbedding(nn.Module):
    def __init__(self, n_vocab : int, n_embd:int, n_token:int):
        super(CLPEmbedding, self).__init__()
        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        self.position_embedding = nn.Parameter(torch.zeros(n_token, n_embd))

    def forward(self, x):
#         batch, seq_len -> batch, seq_len, Dim
        x = self.token_embedding(x)

        x += self.position_embedding

        return x

class CLIPLayer(nn.Module):
    def __init__(self, n_head:int, n_embd:int):
        super(CLIPLayer, self).__init__()

        self.layer_norm = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_embd, n_head)
        self.layer_norm2 = nn.LayerNorm(n_embd)
        self.linear_1 = nn.Linear(n_embd, 4*n_embd)
        self.linear_2 = nn.Linear(4*n_embd, n_embd)

    def forward(self, x):

        residual = x

#         self attention
        x = self.layer_norm(x)

        x = self.attention(x,causal_mask=True)

        x += residual

        residual = x

        x = self.layer_norm2(x)

        x = self.linear_1(x)

        x = x * torch.sigmoid(1.702*x)


        x= self.linear_2(x)

        x = x + residual

        return x


class CLIP(nn.Module):
    def __init__(self,):
        self.embedding = CLPEmbedding(49408,768,77)

        self.layers = nn.Module([
            CLIPLayer(12,768) for i in range(12)
        ])

        self.layernorm = nn. LayerNorm(768)

    def forward(self, token : torch.LongTensor):
        token = token,type(torch.long)
        state = self.embedding(token)

        for layer in self.layers:
            state = layer(state)

        return self.layernorm(state)