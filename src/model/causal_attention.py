import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import polars as pl

inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],
        [0.55, 0.87, 0.66],
        [0.57, 0.85, 0.64],
        [0.22, 0.58, 0.33],
        [0.77, 0.25, 0.10],
        [0.05, 0.80, 0.55],
    ]
)


# create a sample batch manually
batch = torch.stack((inputs, inputs), dim=0)
print(batch)
print(batch.shape)


# class CausalAttention(nn.Module):
#     def __init__(self, d_in, d_out, qkv_bias, context_length, dropout) -> None:
#         super().__init__()
#         self.d_in = d_in
#         self.d_out = d_out
#         self.qkv_bias = qkv_bias
#         self.context_length = context_length
#         self.dropout = nn.Dropout(dropout)
#         self.wq = nn.Linear(self.d_in,self.d_out,bias=self.qkv_bias)
#         self.wk = nn.Linear(self.d_in,self.d_out,bias=self.qkv_bias)
#         self.wv = nn.Linear(self.d_in,self.d_out,bias=self.qkv_bias)

#     def forward(self, inputs):
#         query   = self.wq(inputs)
#         keys = self.wk(inputs)
#         values= self.wv(inputs)

#         attention_scores = query @ keys.T
#         d_k = keys.shape[-1]
#         attention_weights = torch.softmax(attention_scores/d_k **0.5,dim = -1)
#         mask = torch.triu(torch.ones(self.context_length, self.context_length),diagonal=1)
#         masked_attention_weights = attention_weights * mask
#         dropout_masked_weights = self.dropout(masked_attention_weights)
#         context_vector = dropout_masked_weights @ values
#         return context_vector


print(inputs)
print(inputs.shape)
print(inputs.T.shape)
print(inputs.transpose(0, 1))
print(inputs.transpose(0, 1).shape)


class CausalAttention(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        context_length,
        dropout,
        qkv_bias=False,
    ):
        super().__init__()
        self.d_out = d_out
        self.wq = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.wk = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.wv = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        batch, num_tokens, d_in = x.shape
        query = self.wq(x)
        keys = self.wk(x)
        values = self.wv(x)
        attention_scores = query @ keys.transpose(
            1, 2
        )  # this simply swaps the dimensions.
        masked_attention_scores = attention_scores.masked_fill(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        attention_weights = torch.softmax(
            masked_attention_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        attention_weights = self.dropout(attention_weights)
        context_vector = attention_weights @ values
        return context_vector


torch.manual_seed(123)
context_length = batch.shape[1]
d_in = 3
d_out = 2
causal_attention = CausalAttention(d_in, d_out, context_length, dropout=0)


context_vectors = causal_attention(batch)
print(context_vectors.shape)
print(context_vectors)

# example how concatenate works
torch.cat((inputs[:, 1], inputs[:, 2]))


class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                CausalAttention(d_in, d_out, context_length, dropout)
                for _ in range(num_heads)
            ]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


print(batch.shape)  # shape is (2,6,3)

torch.manual_seed(123)
d_in = 3
d_out = 2
context_length = batch.shape[1]
dropout = 0.0
num_heads = 2
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, dropout, num_heads)
context_vectors = mha(batch)
print(context_vectors.shape)
print(context_vectors)
print(batch.shape)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_int, d_out, context_length, dropout, num_heads, qkv_bias):
        super().__init__()
        self.d_out = d_out
        self.context_length = context_length
        self.num_heads = num_heads
        self.dropout = nn.Dropout(0.0)
        self.wq = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.wk = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.wv = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.head_dim = self.d_out // self.num_heads
        self.out_proj = nn.Linear(d_out, d_out)
        assert (
            self.d_out % self.num_heads == 0
        ), "embedd dimensions d_out must be divisible by the number of heads"
        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(self.context_length, self.context_length), diagonal=1
            ),
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        queries = self.wq(x)
        keys = self.wk(x)
        values = self.wv(x)

        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        queries = queries.transpose(
            1, 2
        )  # (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(
            1, 2
        )  # (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        values = values.transpose(
            1, 2
        )  # (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)

        attention_scores = queries @ keys.transpose(2, 3)
        masked_attn = attention_scores.masked_fill(mask, -torch.inf)


torch.manual_seed(42)
t1 = torch.randn((3, 3))
t2 = torch.randn((4, 3))
print(t1)
print(t2)
