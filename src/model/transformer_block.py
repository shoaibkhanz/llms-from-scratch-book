import torch
import tiktoken
import torch.nn as nn
import matplotlib.pyplot as plt

CONFIG = {
    "vocab_size": 50257,  # this is the size of the full text, for english language, this would be all the words in english
    "context_length": 1024,  # this represents the length of text passed in each batch
    "emb_dim": 768,  # this is the embedding size of the model, the paper calls this d_model
    "n_heads": 12,  # number of splits of d_model across the text input
    "n_layers": 12,  # number of layers for encoder ,decoder blocks
    "drop_rate": 0.1,  # the dropout rate, to ensure the models is able to generalise
    "qkv_bias": False,  # query-key-value bias
}


class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, x):
        batch_size, seq_len = x.shape
        tok_embedds = self.tok_emb(x)
        pos_embedds = self.pos_emb(torch.arange(seq_len, device=x.device))
        inp_embedds = tok_embedds + pos_embedds
        inp_embedds = self.dropout(inp_embedds)
        inp_embedds = self.trf_blocks(inp_embedds)
        inp_embedds = self.final_norm(inp_embedds)
        logits = self.out_head(inp_embedds)
        return logits


class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        pass

    def forward(self, x):
        return x


class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        pass

    def forward(self, x):
        return x


tokenizer = tiktoken.get_encoding("gpt2")
batch = []
text1 = "Every effort moves you"
text2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(text1)))
batch.append(torch.tensor(tokenizer.encode(text2)))
batch = torch.stack(batch, dim=0)
print(batch)

torch.manual_seed(123)
model = DummyGPTModel(CONFIG)
logits = model(batch)

# exploring layer normalisation

torch.manual_seed(123)
batch_example = torch.randn(2, 5)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
print(out)

out.mean(dim=-1, keepdim=True)
out.var(dim=-1, keepdim=True)


# implementing layer normalisation
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        self.emb_dim = emb_dim
        super().__init__()
        self.eps = 1e-5  #  10^-5
        self.scale = nn.Parameter(torch.ones(self.emb_dim))
        self.shift = nn.Parameter(torch.zeros(self.emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(
            dim=-1, unbiased=False, keepdim=True
        )  # unbiased simple mean we divide by n and not by n-1 , this is also known as bessel correction
        norm = (x - mean) / torch.sqrt(
            var + self.eps
        )  # adding eps to avoid division by zero error
        return (
            self.scale * norm + self.shift
        )  # scale and shift are trainable parameters


layernorm = LayerNorm(emb_dim=5)
out_norm = layernorm(batch_example)
print(out_norm)

print(out_norm.mean(dim=-1, keepdim=True))
print(out_norm.var(dim=-1, unbiased=False, keepdim=True))


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


gelu, relu = GELU(), nn.ReLU()


x = torch.linspace(-15, 15, 100)  # A
y_gelu, y_relu = gelu(x), relu(x)
plt.figure(figsize=(8, 3))
for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
    plt.subplot(1, 2, i)
    plt.plot(x, y)
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Implementing FeedForwardModel
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


ffn = FeedForward(CONFIG)
rand_tensor = torch.rand((2, 3, 768))
out = ffn(rand_tensor)
print(out.shape)


class ExampleDNN(nn.Module):
    def __init__(self, layer_sizes, skip_connection: bool):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.skip_connection = skip_connection
        self.layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU()),
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
            if self.skip_connection and x.shape == out.shape:
                x = x + out
            else:
                x = out
        return x


layers = [3, 3, 3, 3, 3, 1]
dnn = ExampleDNN(layers, True)

sample_input = torch.tensor([[1.0, 0.0, -1.0]])
dnn(sample_input)


def print_gradients(model: ExampleDNN, data: torch.Tensor):
    output = model(data)
    target = torch.tensor([[0.0]])
    loss = nn.MSELoss()
    loss = loss(output, target)
    loss.backward()

    # HERE IS WHATS INSIDE
    # >>> n1 = dnn.named_parameters()
    # >>> n1
    # <generator object Module.named_parameters at 0x137504740>
    # >>> next(n1)
    # ('layers.0.0.weight',
    #  Parameter containing:
    #  tensor([[-0.1346, -0.1479,  0.0432],
    #          [ 0.5255,  0.2858, -0.0024],
    #          [ 0.4098, -0.2959,  0.2975]],
    #         requires_grad=True))

    for name, param in model.named_parameters():
        if "weight" in name:
            print(
                f"{name} of the parameter and the absolute gradient is "
                f"{param.grad.abs()}"
            )


torch.manual_seed(123)
model_noskip = ExampleDNN(layers, False)
model_skip = ExampleDNN(layers, True)
print_gradients(model_noskip, sample_input)
print_gradients(model_skip, sample_input)


CONFIG = {
    "vocab_size": 50257,  # this is the size of the full text, for english language, this would be all the words in english
    "context_length": 1024,  # this represents the length of text passed in each batch
    "emb_dim": 768,  # this is the embedding size of the model, the paper calls this d_model
    "n_heads": 12,  # number of splits of d_model across the text input
    "n_layers": 12,  # number of layers for encoder ,decoder blocks
    "drop_rate": 0.1,  # the dropout rate, to ensure the models is able to generalise
    "qkv_bias": False,  # query-key-value bias
}

from src.model.causal_attention import MultiHeadAttention


class GPTModel_v2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embedding_layer = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.positional_layer = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])
        self.transformer_block = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.result_layer = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, x):
        # here we get a bactch of data with a given sequence length
        batch_size, seq_len = x.shape
        token_embeddings = self.embedding_layer(x)
        pos_embeddings = self.positional_layer(torch.arange(seq_len, device=x.device))
        input_embeddings = pos_embeddings + token_embeddings
        dropout = self.dropout(input_embeddings)
        trans_block = self.transformer_block(dropout)
        final_norm = self.final_norm(trans_block)
        logits = self.result_layer(final_norm)
        return logits


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm_layers = nn.ModuleList(
            [LayerNorm(cfg["emb_dim"]), LayerNorm(cfg["emb_dim"])]
        )
        self.attention = MultiHeadAttention(
            cfg["emb_dim"],
            cfg["emb_dim"],
            cfg["context_length"],
            cfg["drop_rate"],
            cfg["n_heads"],
            cfg["qkv_bias"],
        )
        self.dropout = nn.Dropout(cfg["drop_rate"])
        self.ffn = FeedForward(cfg=cfg)

    def forward(self, x):
        skip_con = x
        x = self.norm_layers[0](x)
        x = self.attention(x)
        x = self.dropout(x)
        x = x + skip_con
        skip_con = x
        x = self.norm_layers[1](x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + skip_con
        return x


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.eps = 1e6
        self.scale = nn.Parameter(torch.ones(self.emb_dim))
        self.shift = nn.Parameter(torch.zeros(self.emb_dim))

    def forward(self, x: torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, unbiased=True, keepdim=True)
        norm = (x - mean) / (torch.sqrt(variance + self.eps))
        return (self.scale * norm) + self.shift


torch.manual_seed(123)
x = torch.rand(2, 4, 768)
trans_block = TransformerBlock(cfg=CONFIG)
transformer_out = trans_block(x)
print(x.shape)
print(transformer_out.shape)


torch.manual_seed(123)
model = GPTModel_v2(CONFIG)
out = model(batch)
out.shape


total_parameters = sum(p.numel() for p in model.parameters())
model.embedding_layer.weight.shape
model.result_layer.weight.shape
total_gpt2_parameters = total_parameters - sum(
    p.numel() for p in model.result_layer.parameters()
)
print(total_gpt2_parameters)
