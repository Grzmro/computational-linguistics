import json
import math
import time

import torch
from tokenizers import Tokenizer

from models import average_loss, build_model, generate

PROMPTS = [
    "Polska jest krajem",
    "Stolicą Francji jest",
    "Mikołaj Kopernik urodził się",
    "Wczoraj wieczorem poszedłem do",
    "Najważniejszym wynalazkiem XX wieku",
    "W lesie można spotkać",
    "Sztuczna inteligencja to",
    "Rząd ogłosił, że",
    "Aby przygotować dobrą zupę pomidorową, należy",
    "Wisła jest najdłuższą rzeką",
]
NEW_TOKENS = 100

device = "cuda"
tokenizer = Tokenizer.from_file("data/tokenizer.json")
wiki_data = torch.load("data/wiki.pt")
wiki_words = len(tokenizer.decode(wiki_data.tolist()).split())
tokens_per_word = len(wiki_data) / wiki_words

for name in ["lstm", "transformer"]:
    model = build_model(name, tokenizer.get_vocab_size()).to(device)
    model.load_state_dict(torch.load(f"checkpoints/{name}.pt"))
    generate(model, [0], 5)

    torch.cuda.synchronize()
    start = time.time()
    loss = average_loss(model, wiki_data)
    torch.cuda.synchronize()
    eval_time = time.time() - start

    torch.manual_seed(0)
    start = time.time()
    completions = []
    for prompt in PROMPTS:
        ids = generate(model, tokenizer.encode(prompt).ids, NEW_TOKENS)
        completions.append(tokenizer.decode(ids))
    generation_time = time.time() - start

    results = {
        "model": name,
        "wiki_tokens": len(wiki_data),
        "wiki_words": wiki_words,
        "token_perplexity": math.exp(loss),
        "word_perplexity": math.exp(loss * tokens_per_word),
        "eval_time_s": eval_time,
        "eval_tokens_per_s": len(wiki_data) / eval_time,
        "generation_ms_per_token": 1000 * generation_time / (len(PROMPTS) * NEW_TOKENS),
        "completions": dict(zip(PROMPTS, completions)),
    }
    with open(f"results/{name}_eval.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n===== {name} =====")
    print(f"Perplexity (tokeny): {results['token_perplexity']:.1f}")
    print(f"Perplexity (słowa):  {results['word_perplexity']:.1f}")
    print(f"Ewaluacja Wikipedii: {eval_time:.1f}s ({results['eval_tokens_per_s']:.0f} tokenów/s)")
    print(f"Generowanie: {results['generation_ms_per_token']:.2f} ms/token")
    for completion in completions:
        print("-", completion.replace("\n", " "))
