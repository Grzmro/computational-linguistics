import os
from itertools import islice

import torch
from datasets import load_dataset
from speakleash import Speakleash
from tokenizers import ByteLevelBPETokenizer

VOCAB_SIZE = 8000
TRAIN_DATASET = "news_4_business_corpus"
WIKI_ARTICLES = 500

os.makedirs("data/speakleash", exist_ok=True)
sl = Speakleash("data/speakleash")
train_texts = list(sl.get(TRAIN_DATASET).data)

wiki = load_dataset("wikimedia/wikipedia", "20231101.pl", split="train", streaming=True)
wiki_texts = [article["text"] for article in islice(wiki, WIKI_ARTICLES)]

tokenizer = ByteLevelBPETokenizer()
tokenizer.train_from_iterator(train_texts, vocab_size=VOCAB_SIZE, special_tokens=["<|endoftext|>"])
tokenizer.save("data/tokenizer.json")
end_id = tokenizer.token_to_id("<|endoftext|>")


def encode(texts):
    return torch.cat([torch.tensor(enc.ids + [end_id]) for enc in tokenizer.encode_batch(texts)])


train_ids = encode(train_texts)
wiki_ids = encode(wiki_texts)

split = int(0.99 * len(train_ids))
torch.save(train_ids[:split], "data/train.pt")
torch.save(train_ids[split:], "data/val.pt")
torch.save(wiki_ids, "data/wiki.pt")

for name, texts, ids in [("train", train_texts, train_ids), ("wiki", wiki_texts, wiki_ids)]:
    chars = sum(len(t) for t in texts)
    words = sum(len(t.split()) for t in texts)
    print(f"{name}: {len(texts)} dokumentów, {chars} znaków, {words} słów, {len(ids)} tokenów")
