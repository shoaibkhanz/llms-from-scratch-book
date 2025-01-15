import tiktoken
import torch
from torch.return_types import topk
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from src.model.transformer_block import CONFIG, GPTModel_v2
import matplotlib.pyplot as plt


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
    tokenizer: tiktoken.Encoding,
    shuffle: bool = False,
    drop_last: bool = True,
):
    # tokenizer = tiktoken.get_encoding("gpt2")
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


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )

        # Print a sample text after each epoch
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.positional_layer.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded, max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()


def plot_losses(train_losses, valid_losses, tokens_seen, epoch_seen):
    fig, ax1 = plt.subplots(figsize=(5, 1))
    ax1.plot(epoch_seen, train_losses, label="training data")
    ax1.plot(epoch_seen, valid_losses, label="validation data")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    plt.show()


def generate_text(
    model, idx, max_new_tokens, context_size, temperature, top_k=None, eos_id=None
):
    max_new_tokens = 3
    for _ in range(max_new_tokens):
        next_tokens = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(next_tokens)
        logits = logits[:, -1, :]
        topk_logits = torch.topk(logits, k=3)
        temp_scaled = topk_logits / temperature
        probas = torch.softmax(temp_scaled, dim=2)
        sample = torch.multinomial(probas, num_samples=1)


GPT2_small_config = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 256,  # Shortened context length (orig: 1024)
    "emb_dim": 768,  # Embedding dimension
    "n_heads": 12,  # Number of attention heads
    "n_layers": 12,  # Number of layers
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": False,  # Query-key-value bias
}

# GPT2_small_config = {
#     "vocab_size": 50257,  # Size of the vocabulary used by the model
#     "context_length": 256,  # Maximum length of input sequences
#     "emb_dim": 256,  # Dimensionality of the model's embeddings (d_model)
#     "n_heads": 16,  # Number of attention heads in the multi-head attention mechanism
#     "n_layers": 24,  # Number of transformer layers in the model
#     "drop_rate": 0.1,  # Dropout rate for regularization
#     "qkv_bias": False,  # Whether to include bias terms in the query, key, and value projections
# }

if __name__ == "__main__":
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

    # model trainning hasnt happened yet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(123)
    model = GPTModel_v2(GPT2_small_config)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

    tokenizer = tiktoken.get_encoding("gpt2")

    train_loader = create_dataloader(
        train_data,
        GPT2_small_config["context_length"],
        GPT2_small_config["context_length"],
        batch_size=batch_size,
        tokenizer=tokenizer,
        shuffle=True,
        drop_last=True,
    )
    val_loader = create_dataloader(
        val_data,
        GPT2_small_config["context_length"],
        GPT2_small_config["context_length"],
        batch_size=batch_size,
        tokenizer=tokenizer,
        shuffle=True,
        drop_last=False,
    )
    num_epochs = 10

    train_losses, val_losses, tokens_seen = train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=num_epochs,
        eval_freq=5,
        eval_iter=5,
        start_context="Every effort moves you",
        tokenizer=tokenizer,
    )

    epochs_seen = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(train_losses, val_losses, tokens_seen, epochs_seen)

    model.to("cpu")
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids("Every effort moves you", tokenizer),
        max_new_tokens=25,
        context_size=GPT2_small_config["context_length"],
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

    vocab = {
        "closer": 0,
        "every": 1,
        "effort": 2,
        "forward": 3,
        "inches": 4,
        "moves": 5,
        "pizza": 6,
        "toward": 7,
        "you": 8,
    }

    inverse_vocab = {v: k for k, v in vocab.items()}

    next_token_logits = torch.tensor(
        [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
    )
    probas = torch.softmax(next_token_logits, dim=0)
    print(probas)
    torch.argmax(probas).item()
    print(inverse_vocab[3])

    torch.manual_seed(123)
    sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")

    def softmax_with_temperature(logits, temperature):
        scaled_logits = logits / temperature
        return torch.softmax(scaled_logits, dim=0)

    temperatures = [1, 0.1, 5]

    scaled_probas = [
        softmax_with_temperature(next_token_logits, t) for t in temperatures
    ]

    len(scaled_probas)

    idx = text_to_token_ids("every efforts moves you", tokenizer=tokenizer)
    print(idx)
    context_size = 256
    next_tokens = idx[:, -context_size:]
    logits = model(next_tokens)
    torch.topk()
