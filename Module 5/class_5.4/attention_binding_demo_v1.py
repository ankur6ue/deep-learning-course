"""
attention_binding_demo_v1.py

Baseline "pooled attention only" version of the binding demo.
- NO self-attention / contextualization block.
- Uses query-conditioned attention pooling over token embeddings (+ positional embeddings).

Purpose (teaching):
- Show that attention-as-pooling cannot reliably perform "binding" unless token representations
  already encode relational context (which self-attention provides).

Dataset (binding task):
- Sentence contains both aspects (food, service)
- Exactly one 'good' and one 'bad' appear (aspects always disagree)
- Query is either 'food' or 'service'
- Label = sentiment about the queried aspect (1 if good else 0)

Expected behavior:
- Mean pooling stays ~50%
- Pooled-attention-only may improve a bit but typically cannot reach perfect accuracy
  because it struggles to bind sentiment to the queried aspect without contextualization.

Run:
  python attention_binding_demo_v1.py --epochs 15 --device cpu
"""

from __future__ import annotations
import argparse
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


VOCAB = ["[PAD]", "[Q]", "food", "service", "good", "bad", "the", "was", "but", "and"]
PAD_ID = 0
TOK2ID = {t: i for i, t in enumerate(VOCAB)}
ID2TOK = {i: t for t, i in TOK2ID.items()}
ASPECTS = ["food", "service"]

@dataclass
class Example:
    query_aspect: str
    tokens: List[str]
    label: int  # 1=positive about query, 0=negative about query


def make_example(rng: random.Random, min_len=10, max_len=16) -> Example:
    """Always-disagree version (cleanest to show binding)."""
    good_aspect = rng.choice(ASPECTS)
    bad_aspect = "service" if good_aspect == "food" else "food"

    query_aspect = rng.choice(ASPECTS)
    label = 1 if query_aspect == good_aspect else 0

    clause_good = ["the", good_aspect, "was", "good"]
    clause_bad = ["but", "the", bad_aspect, "was", "bad"]

    if rng.random() < 0.5:
        sent = clause_good + clause_bad
    else:
        clause_bad2 = ["the", bad_aspect, "was", "bad"]
        clause_good2 = [rng.choice(["and", "but"]), "the", good_aspect, "was", "good"]
        sent = clause_bad2 + clause_good2

    fillers = ["the", "was", "and", "but"]
    target_len = rng.randint(min_len, max_len)
    while len(sent) < target_len:
        if rng.random() < 0.5:
            sent.insert(0, rng.choice(fillers))
        else:
            sent.append(rng.choice(fillers))

    return Example(query_aspect=query_aspect, tokens=sent, label=label)


class ToyAspectDataset(Dataset):
    def __init__(self, n: int, seed: int):
        rng = random.Random(seed)
        self.data = [make_example(rng) for _ in range(n)]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Example:
        return self.data[idx]


def encode(example: Example, max_len: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    toks = ["[Q]", example.query_aspect] + example.tokens
    ids = [TOK2ID[t] for t in toks][:max_len]
    attn = [1] * len(ids)
    while len(ids) < max_len:
        ids.append(PAD_ID)
        attn.append(0)
    return (
        torch.tensor(ids, dtype=torch.long),
        torch.tensor(attn, dtype=torch.long),
        torch.tensor(example.label, dtype=torch.long),
    )


def collate_fn(batch: List[Example], max_len: int):
    ids, attn, y = zip(*(encode(ex, max_len) for ex in batch))
    return torch.stack(ids), torch.stack(attn), torch.stack(y)


class MeanPoolClassifier(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.fc = nn.Linear(d_model, 2)

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = self.emb(input_ids)
        m = attn_mask.unsqueeze(-1).float()
        pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        return self.fc(pooled)


class PooledAttentionClassifier(nn.Module):
    """
    Query-conditioned attention pooling ONLY.
    - Token reps: embedding + position embedding (no contextualization across tokens)
    - Query: aspect token at position 1
    - Pooling: attend over sentence tokens only (exclude [Q] and aspect positions)
    """
    def __init__(self, vocab_size: int, d_model: int, d_k: int, max_len: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)

        self.Wq = nn.Linear(d_model, d_k, bias=False)
        self.Wk = nn.Linear(d_model, d_k, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)

        self.fc = nn.Linear(d_model, 2)

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = self.emb(input_ids) + self.pos(pos_ids)

        q = self.Wq(x[:, 1, :])
        k = self.Wk(x)
        v = self.Wv(x)

        scores = (k * q.unsqueeze(1)).sum(dim=-1) / (q.size(-1) ** 0.5)

        pool_mask = attn_mask.clone()
        pool_mask[:, 0] = 0
        pool_mask[:, 1] = 0
        scores = scores.masked_fill(pool_mask == 0, -1e9)

        alpha = torch.softmax(scores, dim=-1)
        h = (alpha.unsqueeze(-1) * v).sum(dim=1)
        return self.fc(h)

    @torch.no_grad()
    def attention_weights(self, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        assert B == 1
        pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = self.emb(input_ids) + self.pos(pos_ids)

        q = self.Wq(x[:, 1, :])
        k = self.Wk(x)
        scores = (k * q.unsqueeze(1)).sum(dim=-1) / (q.size(-1) ** 0.5)

        pool_mask = attn_mask.clone()
        pool_mask[:, 0] = 0
        pool_mask[:, 1] = 0
        scores = scores.masked_fill(pool_mask == 0, -1e9)
        return torch.softmax(scores, dim=-1)


def accuracy(model: nn.Module, dl: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for input_ids, attn_mask, y in dl:
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        y = y.to(device)
        pred = model(input_ids, attn_mask).argmax(dim=-1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return correct / max(1, total)


def train_epoch(model: nn.Module, dl: DataLoader, opt: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total = 0.0
    n = 0
    for input_ids, attn_mask, y in dl:
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        y = y.to(device)

        logits = model(input_ids, attn_mask)
        loss = nn.functional.cross_entropy(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        total += float(loss.item()) * y.size(0)
        n += y.size(0)
    return total / max(1, n)


def pretty(tokens: List[int], weights: List[float]) -> str:
    pairs = []
    for tid, w in zip(tokens, weights):
        tok = ID2TOK[int(tid)]
        if tok == "[PAD]":
            continue
        pairs.append(f"{tok}:{w:.2f}")
    return "  ".join(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_n", type=int, default=8000)
    ap.add_argument("--test_n", type=int, default=2000)
    ap.add_argument("--max_len", type=int, default=24)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--d_k", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)

    train_ds = ToyAspectDataset(args.train_n, seed=args.seed)
    test_ds = ToyAspectDataset(args.test_n, seed=args.seed + 1)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=lambda b: collate_fn(b, args.max_len))
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         collate_fn=lambda b: collate_fn(b, args.max_len))

    mean_model = MeanPoolClassifier(len(VOCAB), args.d_model).to(device)
    attn_model = PooledAttentionClassifier(len(VOCAB), args.d_model, args.d_k, args.max_len).to(device)

    mean_opt = torch.optim.AdamW(mean_model.parameters(), lr=args.lr)
    attn_opt = torch.optim.AdamW(attn_model.parameters(), lr=args.lr)

    print("Training...")
    for ep in range(1, args.epochs + 1):
        mean_loss = train_epoch(mean_model, train_dl, mean_opt, device)
        attn_loss = train_epoch(attn_model, train_dl, attn_opt, device)
        mean_acc = accuracy(mean_model, test_dl, device)
        attn_acc = accuracy(attn_model, test_dl, device)
        print(f"epoch {ep:02d} | mean: loss={mean_loss:.4f} acc={mean_acc:.3f} | pooled-attn: loss={attn_loss:.4f} acc={attn_acc:.3f}")

    for _ in range(3):
        ex = make_example(rng)
        input_ids, attn_mask, _ = encode(ex, args.max_len)
        input_ids_b = input_ids.unsqueeze(0).to(device)
        attn_mask_b = attn_mask.unsqueeze(0).to(device)

        alpha = attn_model.attention_weights(input_ids_b, attn_mask_b)[0].detach().cpu().numpy()
        toks = input_ids.detach().cpu().numpy().tolist()

        print("\nExample:")
        print(" query_aspect:", ex.query_aspect, "label:", ex.label)
        print(" tokens:", " ".join(ex.tokens))
        print(" pooling attention weights:")
        print(pretty(toks, alpha.tolist()))


if __name__ == "__main__":
    main()
