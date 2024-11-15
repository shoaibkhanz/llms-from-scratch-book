import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, dataloader
from pathlib import Path
from src.model.transformer_block import GPTModel_v2


class GPTDataset(Dataset):
    def __init__(
        self, text, tokenizer: tiktoken.Encoding, max_seq_length: int, stride: int
    ):
        super().__init__()
        self.inputs = []
        self.targets = []
        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        for idx in range(0, len(token_ids) - max_seq_length, stride):
            self.inputs.append(torch.tensor(token_ids[idx : idx + max_seq_length]))
            self.targets.append(
                torch.tensor(token_ids[idx + 1 : idx + max_seq_length + 1])
            )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def create_dataloader(
    text: str,
    max_seq_length: int,
    stride: int,
    batch_size: int,
    shuffle: bool = False,
    drop_last: bool = True,
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(text, tokenizer, max_seq_length, stride)
    data_loader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )
    return data_loader


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


def calc_loss_loader(
    data_loader,
    model,
    device,
    num_batches=None,
):
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    total_loss = 0

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


if __name__ == "__main__":
    GPT2_small_config = {
        "vocab_size": 50257,  # Size of the vocabulary used by the model
        "context_length": 256,  # Maximum length of input sequences
        "emb_dim": 256,  # Dimensionality of the model's embeddings (d_model)
        "n_heads": 16,  # Number of attention heads in the multi-head attention mechanism
        "n_layers": 24,  # Number of transformer layers in the model
        "drop_rate": 0.1,  # Dropout rate for regularization
        "qkv_bias": False,  # Whether to include bias terms in the query, key, and value projections
    }

    filepath = Path("resources/verdict.txt")
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    total_characters = len(text)
    total_tokens = len(tiktoken.get_encoding("gpt2").encode(text))

    print(total_characters)
    print(total_tokens)
    print(total_tokens * 0.9)
    print(total_tokens * 0.1)

    train_ratio = 0.90
    split_idx = int(len(text) * train_ratio)
    train_data = text[:split_idx]
    val_data = text[split_idx:]
    batch_size = 2
    print(len(train_data))
    print(len(val_data))
    train_loader = create_dataloader(
        train_data,
        GPT2_small_config["context_length"],
        GPT2_small_config["context_length"],
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = create_dataloader(
        val_data,
        GPT2_small_config["context_length"],
        GPT2_small_config["context_length"],
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    for input, target in train_loader:
        print(input.shape, target.shape)

    for input, target in val_loader:
        print(input.shape, target.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = GPTModel_v2(GPT2_small_config)
    model.to(device)
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)
    print("train loss", train_loss)
    print("val loss", val_loss)
