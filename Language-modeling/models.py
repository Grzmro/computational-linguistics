import math

import torch
import torch.nn as nn
import torch.nn.functional as F

CONTEXT_LEN = 256


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.lstm = nn.LSTM(d_model, d_model, n_layers, batch_first=True)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x, state=None):
        h = self.embedding(x)
        h, state = self.lstm(h, state)
        return self.head(h), state


class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        B, T, C = x.shape
        return x.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

    def forward(self, x):
        B, T, C = x.shape
        q = self.split_heads(self.query(x))
        k = self.split_heads(self.key(x))
        v = self.split_heads(self.value(x))

        scores = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))
        future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(future, float("-inf"))
        weights = F.softmax(scores, dim=-1)

        y = (weights @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(y)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = SelfAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=320, n_layers=6, n_heads=5):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(CONTEXT_LEN, d_model)
        self.blocks = nn.Sequential(*[TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x, state=None):
        positions = torch.arange(x.size(1), device=x.device)
        h = self.token_embedding(x) + self.position_embedding(positions)
        h = self.blocks(h)
        return self.head(self.norm(h)), None


def build_model(name, vocab_size):
    if name == "lstm":
        return LSTMModel(vocab_size)
    if name == "transformer":
        return TransformerModel(vocab_size)
    raise ValueError(name)


@torch.no_grad()
def average_loss(model, data, batch_size=32):
    model.eval()
    device = next(model.parameters()).device
    n = (len(data) - 1) // CONTEXT_LEN
    x = data[: n * CONTEXT_LEN].view(n, CONTEXT_LEN)
    y = data[1 : n * CONTEXT_LEN + 1].view(n, CONTEXT_LEN)

    total = 0.0
    for i in range(0, n, batch_size):
        logits, _ = model(x[i : i + batch_size].to(device))
        targets = y[i : i + batch_size].to(device)
        total += F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="sum").item()
    model.train()
    return total / (n * CONTEXT_LEN)


@torch.no_grad()
def generate(model, ids, n_tokens, temperature=0.8, top_k=50):
    model.eval()
    device = next(model.parameters()).device
    x = torch.tensor([ids], device=device)
    new_input = x
    state = None

    for _ in range(n_tokens):
        if isinstance(model, LSTMModel):
            logits, state = model(new_input, state)
        else:
            logits, _ = model(x[:, -CONTEXT_LEN:])

        logits = logits[:, -1] / temperature
        threshold = torch.topk(logits, top_k).values[:, -1:]
        logits[logits < threshold] = float("-inf")
        new_input = torch.multinomial(F.softmax(logits, dim=-1), 1)
        x = torch.cat([x, new_input], dim=1)

    return x[0].tolist()
