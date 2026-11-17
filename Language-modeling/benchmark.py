import json
import time

import torch
import torch.nn.functional as F

from models import CONTEXT_LEN, build_model

WARMUP, STEPS, BATCH_SIZE = 20, 300, 32

torch.manual_seed(0)
device = "cuda"
train_data = torch.load("data/train.pt")


def get_batch():
    starts = torch.randint(len(train_data) - CONTEXT_LEN - 1, (BATCH_SIZE,))
    x = torch.stack([train_data[s : s + CONTEXT_LEN] for s in starts])
    y = torch.stack([train_data[s + 1 : s + CONTEXT_LEN + 1] for s in starts])
    return x.to(device), y.to(device)


results = {}
for name in ["lstm", "transformer"]:
    model = build_model(name, 8000).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for step in range(WARMUP + STEPS):
        if step == WARMUP:
            torch.cuda.synchronize()
            start = time.time()
        x, y = get_batch()
        logits, _ = model(x)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    torch.cuda.synchronize()

    seconds_per_step = (time.time() - start) / STEPS
    results[name] = {
        "seconds_per_step": seconds_per_step,
        "tokens_per_s": BATCH_SIZE * CONTEXT_LEN / seconds_per_step,
        "minutes_for_6000_steps": 6000 * seconds_per_step / 60,
    }
    print(f"{name}: {seconds_per_step:.3f} s/krok, {results[name]['tokens_per_s']:.0f} tokenów/s, "
          f"6000 kroków = {results[name]['minutes_for_6000_steps']:.1f} min")
    del model, optimizer

with open("results/benchmark.json", "w") as f:
    json.dump(results, f, indent=2)
