import math
import torch
import torch.nn as nn


class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super(InputEmbeddings, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, seq_max_len: int, dropout: float):
        super(PositionalEncoding, self).__init__()
        self.seq_max_len = seq_max_len
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

        pe = torch.zeros(seq_max_len, d_model)
        # create a vector of shape seq
        position = torch.arange(0, seq_max_len, dtype=torch.float).unsqueeze(1)
        # org is
        # div_term = torch.sin(position[::2] / 10000 **(torch.arange(0,d_model,2).float() / d_model))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # (1, seq_len, d_model)
        # register as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNorm(nn.Module):

    def __init__(self, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.beta


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)  # W2 and B2

    def forward(self, x):
        # x => (batch, seq, d_model)
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model, d_head, dropout):
        super(MultiHeadAttentionBlock, self).__init__()
        self.attention_scores = None
        self.d_model = d_model
        self.d_head = d_head
        assert d_model % d_head == 0

        self.d_k = d_model // d_head
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, dropout: nn.Dropout, mask=None, ):
        d_k = query.size(-1)

        attention_score = (query @ key.transpose(-1, -2)) / math.sqrt(d_k)  # h, seq -> seq,seq
        if mask:
            attention_score.masked_fill_(mask == 0, -float('inf'))
        attention_score = torch.softmax(attention_score, dim=-1)
        if dropout:
            attention_score = dropout(attention_score)

        return (attention_score @ value), attention_score

    def forward(self, q, k, v, mask=None):
        # Q K V is input
        query = self.w_q(q)  # (batch, seq, d_model) --> (batch, seq, d_model)
        key = self.w_k(k)
        value = self.w_v(v)

        # (batch, seq, d_model) -> (batch, h, seq, d_k)
        query = query.view(query.size(0), query.size(1), self.d_head, self.d_k).transpose(1,
                                                                                          2)  # d_model -> d_head, d_k
        key = key.view(key.size(0), key.size(1), self.d_head, self.d_k).transpose(1, 2)
        value = value.view(value.size(0), value.size(1), self.d_head, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, self.dropout, mask)
        # (batch, h, seq, d_k) --> (batch, seq, h, d_k) ---> (batch, seq, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_model)

        return self.w_o(x)


class ResidualConnection(nn.Module):

    def __init__(self, dropout):
        super(ResidualConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):

    def __init__(self, self_attention: MultiHeadAttentionBlock, feed_forward: FeedForward, dropout: float):
        super(EncoderBlock, self).__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout)
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, scr_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention(x, x, x, scr_mask))
        x = self.residual_connection[1](x, lambda x: self.feed_forward)
        return x


class Encoder(nn.Module):
    def __init__(self, layer: nn.ModuleList):
        super(Encoder, self).__init__()
        self.layer = layer
        self.norm = LayerNorm()

    def forward(self, x, mask):
        for layer in self.layer:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, self_attention: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock
                 , feed_forward: FeedForward, dropout: float):
        super(DecoderBlock, self).__init__()
        self.cross_attention_block = cross_attention_block
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention(x, x, x, tgt_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output,
                                                                                src_mask))
        x = self.residual_connection[2](x, lambda x: self.feed_forward)
        return x


class Decoder(nn.Module):

    def __init__(self, layer: nn.ModuleList, feed_forward: FeedForward, dropout: float):
        super(Decoder, self).__init__()
        self.layer = layer
        self.feed_forward = feed_forward
        self.norm = LayerNorm()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layer:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class Projection(nn.Module):
    def __init__(self, d_model, vocab_size, dropout):
        super(Projection, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch, seq, dim) --> (batch, seq, vocab)
        return torch.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):

    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 src_embed: InputEmbeddings,
                 tgt_embed: InputEmbeddings,
                 src_pos_embed: PositionalEncoding,
                 tgt_pos_embed: PositionalEncoding,
                 projection: Projection,

                 ):
        super(Transformer, self).__init__(),
        self.encoder = encoder
        self.decoder = decoder
        self.src_pos_embed = src_pos_embed
        self.tgt_pos_embed = tgt_pos_embed
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.projection = projection

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos_embed(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos_embed(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection(x)


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, scr_seq_len : int ,tgt_seq_len: int, d_model: int = 512,
                      N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048):
    # create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # create the positon encoding layer
    src_pos = PositionalEncoding(d_model, scr_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # create encoder block:
    encoder_block = []
    for _ in range(N):
        encoder_self_attention = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward = FeedForward(d_model, d_ff, dropout)
        encoder_block.append(EncoderBlock(encoder_self_attention, feed_forward, dropout))

    # create decoder_blocks
    decoder_block = []
    for _ in range(N):
        decoder_self_attention = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward = FeedForward(d_model, d_ff, dropout)
        decoder_block.append(DecoderBlock(decoder_self_attention, decoder_cross_attention,feed_forward, dropout))

    # createa encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_block))
    decoder = Decoder(nn.ModuleList(decoder_block))

    proj = Projection(d_model, tgt_vocab_size, dropout)

    # create transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos_embed=src_pos, tgt_pos_embed=tgt_pos, projection=proj)

    # initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer

