import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import polars as pl
import tiktoken

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


class GPTDatasetV1(nn.Module):
    def __init__(self, text, tokeniser, max_length, stride):
        super().__init__()
        self.inputs = []
        self.targets = []

        token_ids = tokeniser.encode(text, allowed_special={"<|endoftext|>"})
        # token ids is the full length of the document or sentence,
        # but during the each forward pass we pass a subset of it, thats is what max_length reprpesents here.
        for i in range(0, len(token_ids) - max_length, stride):
            # this make sure that we start with 0: 0+max_length and so on
            input_chunks = token_ids[i : i + max_length]
            # since this is the target, we need it to move by 1 token and thus it ends one token ahead.
            target_chunks = token_ids[i + 1 : (i + 1) + max_length]

            self.inputs.append(torch.tensor(input_chunks))
            self.targets.append(torch.tensor(target_chunks))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def create_dataloader(text, batch_size, max_length=256, stride=128, shuffle=False):
    tokeniser = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(text, tokeniser, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


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
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.context_length = context_length
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
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

        # before this point we have (b,num_tokens,d_out)
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

        attention_scores = queries @ keys.transpose(
            2, 3
        )  # (num_tokens x head_dim) (head_dim, num_tokens) -> (b, num_heads,num_tokens, num_tokens)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        masked_attn = attention_scores.masked_fill(mask_bool, -torch.inf)
        attention_weights = torch.softmax(masked_attn / keys.shape[-1] ** 0.5, dim=-1)
        attention_weights = self.dropout(attention_weights)
        context_vectors = (attention_weights @ values).transpose(1, 2)
        # (num_tokens x num_tokens) (num_tokens, head_dim) -> (b, num_heads,num_tokens, head_dim) -> (b, num_tokens, num_heads, head_dim)
        # operations like transpose, slicing etc can alter how the data is stored, as it may then consists of pointers
        # for better management. However for view we need the data to be in contigious form and thus we need to do as below.
        context_vectors = context_vectors.contiguous().view(b, num_tokens, self.d_out)
        # the following projection , takes the context vector to learn higher dimension parameters and bring it back to the original scale.
        context_vectors = self.out_proj(context_vectors)
        return context_vectors


# torch.manual_seed(123)
# b, max_length, output_dim = batch.shape
# print(batch.shape)
# context_length = max_length
# d_in = output_dim
# d_out = d_in
# print(d_out)

# mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
# cv = mha(batch)
# print(cv)
# print(cv.shape)

if __name__ == "__main__":

    vocab_size = 50257
    d_model = 256
    max_length = 1024

    with open("resources/verdict.txt") as f:
        text = f.read()
    tokeniser = tiktoken.get_encoding("gpt2")
    dataloader = create_dataloader(
        text, batch_size=1, max_length=max_length, shuffle=False
    )

    token_embeddings_layer = nn.Embedding(vocab_size, d_model)
    positional_embeddings_layer = nn.Embedding(max_length, d_model)

    for batch in dataloader:
        inputs, targets = batch
        token_embeddings = token_embeddings_layer(inputs)
        # this positional encoding is just simply an embedding
        positional_embeddings = positional_embeddings_layer(torch.arange(max_length))
        input_embeddings = token_embeddings + positional_embeddings

        mha = MultiHeadAttention(
            d_in=d_model,
            d_out=d_model,
            context_length=max_length,
            dropout=0.0,
            num_heads=2,
        )
        batch = input_embeddings
        context_vec = mha(batch)

    print(input_embeddings.shape)
    print(context_vec.shape)
