import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from models import CONTEXT_LEN, average_loss, build_model

MODEL = sys.argv[1]
STEPS = 6000
BATCH_SIZE = 32
LR = 1e-3
EVAL_EVERY = 500

torch.manual_seed(0)
device = "cuda"

train_data = torch.load("data/train.pt")
val_data = torch.load("data/val.pt")
vocab_size = Tokenizer.from_file("data/tokenizer.json").get_vocab_size()

model = build_model(MODEL, vocab_size).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"{MODEL}: {n_params / 1e6:.2f}M parametrów, GPU: {torch.cuda.get_device_name()}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, STEPS, eta_min=LR / 10)


def get_batch():
    starts = torch.randint(len(train_data) - CONTEXT_LEN - 1, (BATCH_SIZE,))
    x = torch.stack([train_data[s : s + CONTEXT_LEN] for s in starts])
    y = torch.stack([train_data[s + 1 : s + CONTEXT_LEN + 1] for s in starts])
    return x.to(device), y.to(device)


history = []
train_time = 0.0
mark = time.time()

for step in range(1, STEPS + 1):
    x, y = get_batch()
    logits, _ = model(x)
    loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    if step % EVAL_EVERY == 0:
        torch.cuda.synchronize()
        train_time += time.time() - mark

        val_loss = average_loss(model, val_data)
        history.append({"step": step, "train_loss": loss.item(), "val_loss": val_loss, "time": train_time})
        print(f"krok {step}: train loss {loss.item():.3f}, val loss {val_loss:.3f}, "
              f"val ppl {math.exp(val_loss):.1f}, czas {train_time:.0f}s")
        mark = time.time()

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)
torch.save(model.state_dict(), f"checkpoints/{MODEL}.pt")

tokens_seen = STEPS * BATCH_SIZE * CONTEXT_LEN
summary = {
    "model": MODEL,
    "parameters": n_params,
    "steps": STEPS,
    "tokens_seen": tokens_seen,
    "train_time_s": train_time,
    "train_tokens_per_s": tokens_seen / train_time,
    "peak_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
    "history": history,
}
with open(f"results/{MODEL}_train.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"Trening: {train_time / 60:.1f} min, {tokens_seen / train_time:.0f} tokenów/s")
