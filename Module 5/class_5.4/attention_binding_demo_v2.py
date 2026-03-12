"""
attention_binding_demo_v2.py

Interpretability-focused binding demo:
- Mean pooling baseline (bag-of-embeddings) ~ chance on a binding task
- Contextual attention model (1 self-attention block + pooling) ~ high accuracy

This version also SAVES learned embeddings for later visualization:
  outputs/embeddings_demo_v2.npz

Run:
  python attention_binding_demo_v2.py --epochs 10 --device cpu --save_dir outputs
Then:
  python visualize_embeddings.py --npz outputs/embeddings_demo_v2.npz
"""

from __future__ import annotations
import argparse
import random
from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path

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
    # 4 combos ~ equally likely
    food_good = (rng.random() < 0.5)
    service_good = (rng.random() < 0.5)

    query_aspect = rng.choice(ASPECTS)
    if query_aspect == "food":
        label = 1 if food_good else 0
    else:
        label = 1 if service_good else 0

    sent = [
        "the", "food", "was", "good" if food_good else "bad",
        "the", "service", "was", "good" if service_good else "bad",
    ]

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
    toks = [example.query_aspect] + example.tokens
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


class SelfAttentionBlock(nn.Module):
    """Single-head self-attention block that also returns attention matrix alpha for inspection."""
    def __init__(self, d_model: int, d_k: int):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_k, bias=False)
        self.Wk = nn.Linear(d_model, d_k, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)

        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.Wq(x)  # [B,L,d_k]
        k = self.Wk(x)  # [B,L,d_k]
        v = self.Wv(x)  # [B,L,d_model]

        scores = torch.matmul(q, k.transpose(1, 2)) / (q.size(-1) ** 0.5)  # [B,L,L]
        key_mask = attn_mask.unsqueeze(1)  # [B,1,L]
        scores = scores.masked_fill(key_mask == 0, -1e9)

        alpha = torch.softmax(scores, dim=-1)          # [B,L,L]
        ctx = torch.matmul(alpha, v)                   # [B,L,d_model]

        x = self.ln1(x + ctx)
        x2 = self.ff(x)
        x = self.ln2(x + x2)
        return x, alpha


class ContextualQueryAttentionClassifier(nn.Module):
    """
    Contextualize tokens, then do query-conditioned pooling.
    Pooling attends ONLY over sentence tokens (exclude aspect positions) for interpretability.
    """
    def __init__(self, vocab_size: int, d_model: int, d_k: int, max_len: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)

        self.encoder = SelfAttentionBlock(d_model=d_model, d_k=d_k)

        # separate projections for pooling step
        self.Pq = nn.Linear(d_model, d_k, bias=False)
        self.Pk = nn.Linear(d_model, d_k, bias=False)
        self.Pv = nn.Linear(d_model, d_model, bias=False)

        self.fc = nn.Linear(d_model, 2)

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = self.emb(input_ids) + self.pos(pos_ids)
        x, _ = self.encoder(x, attn_mask)

        q = self.Pq(x[:, 1, :])                         # aspect token position
        k = self.Pk(x)
        v = self.Pv(x)

        scores = (k * q.unsqueeze(1)).sum(dim=-1) / (q.size(-1) ** 0.5)  # [B,L]

        pool_mask = attn_mask.clone()
        pool_mask[:, 0] = 0  # exclude aspect token
        scores = scores.masked_fill(pool_mask == 0, -1e9)

        alpha_pool = torch.softmax(scores, dim=-1)
        h = (alpha_pool.unsqueeze(-1) * v).sum(dim=1)
        return self.fc(h)

    @torch.no_grad()
    def inspect(self, input_ids: torch.Tensor, attn_mask: torch.Tensor):
        """Returns encoder self-attn row for aspect token and pooling attention weights."""
        B, L = input_ids.shape
        assert B == 1, "inspect expects batch size 1"
        pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = self.emb(input_ids) + self.pos(pos_ids)
        x, alpha_ctx = self.encoder(x, attn_mask)  # alpha_ctx: [1,L,L]

        q = self.Pq(x[:, 1, :])
        k = self.Pk(x)
        scores = (k * q.unsqueeze(1)).sum(dim=-1) / (q.size(-1) ** 0.5)

        pool_mask = attn_mask.clone()
        pool_mask[:, 0] = 0 # exclude aspect
        scores = scores.masked_fill(pool_mask == 0, -1e9)
        alpha_pool = torch.softmax(scores, dim=-1)  # [1,L]
        tok_id = TOK2ID['good']
        tok_id_list = input_ids.tolist()
        alpha_ctx_row_good = []
        alpha_ctx_row_bad = []
        if tok_id in tok_id_list[0]:
            pos_good = tok_id_list[0].index(tok_id)
            alpha_ctx_row_good = alpha_ctx[0, pos_good, :].detach().cpu().numpy()
        tok_id = TOK2ID['bad']
        if tok_id in tok_id_list[0]:
            pos_bad = tok_id_list[0].index(tok_id)
            alpha_ctx_row_bad = alpha_ctx[0, pos_bad, :].detach().cpu().numpy()
        # aspect token attends to...
        return alpha_ctx_row_good, alpha_ctx_row_bad, alpha_pool[0].detach().cpu()


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

def cosine_sim_matrix(P):  # P: [L,D] torch CPU float
    Pn = P / (P.norm(dim=1, keepdim=True) + 1e-12)
    return Pn @ Pn.T  # [L,L]

# Used to plot a heatmap of the position embeddings before and after training..
# The position embeddings don't move much.. the model needs to move the embeddings just enough to
# solve this easy task
def plot_pos_similarity(P, title="Positional embedding cosine similarity"):
    import matplotlib.pyplot as plt
    S = cosine_sim_matrix(P).numpy()
    plt.figure()
    plt.imshow(S, aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("position j")
    plt.ylabel("position i")
    plt.tight_layout()
    plt.show()


def plot_position_movement(pos_snaps, title="How much each position embedding moved"):
    import matplotlib.pyplot as plt
    P0 = pos_snaps[0]
    Plast = pos_snaps[-1]
    delta = (Plast - P0).norm(dim=1).numpy()  # [L]
    plt.figure()
    plt.plot(delta, marker="o", linewidth=1)
    plt.title(title)
    plt.xlabel("position")
    plt.ylabel("||P_final - P_init||")
    plt.tight_layout()
    plt.show()


def save_embeddings(save_dir: str, model: ContextualQueryAttentionClassifier, max_len: int) -> str:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(save_dir) / "embeddings_demo_v2.npz"

    token_emb = model.emb.weight.detach().cpu().numpy()          # [V, d_model]
    pos_emb = model.pos.weight.detach().cpu().numpy()            # [max_len, d_model]
    np.savez(
        out_path,
        vocab=np.array(VOCAB, dtype=object),
        token_emb=token_emb,
        pos_emb=pos_emb,
        max_len=np.array([max_len], dtype=np.int32),
    )
    return str(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_n", type=int, default=8000)
    ap.add_argument("--test_n", type=int, default=2000)
    ap.add_argument("--max_len", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--d_k", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--save_dir", type=str, default="outputs", help="Where to save learned embeddings")
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
    attn_model = ContextualQueryAttentionClassifier(len(VOCAB), args.d_model, args.d_k, args.max_len).to(device)

    mean_opt = torch.optim.AdamW(mean_model.parameters(), lr=args.lr)
    attn_opt = torch.optim.AdamW(attn_model.parameters(), lr=args.lr)
    plot_pos_similarity(attn_model.pos.weight.detach(), title="Positional embedding cosine similarity")
    pos_snaps = []
    # detach() stops autograd, but it doesn’t copy the tensor. You’re appending a tensor that still points to the
    # same underlying storage. That's why we need the clone
    pos_snaps.append(attn_model.pos.weight.detach().clone().cpu())
    print("Training...")
    for ep in range(1, args.epochs + 1):
        mean_loss = train_epoch(mean_model, train_dl, mean_opt, device)
        attn_loss = train_epoch(attn_model, train_dl, attn_opt, device)
        mean_acc = accuracy(mean_model, test_dl, device)
        attn_acc = accuracy(attn_model, test_dl, device)
        print(f"epoch {ep:02d} | mean: loss={mean_loss:.4f} acc={mean_acc:.3f} | attn+ctx: loss={attn_loss:.4f} acc={attn_acc:.3f}")

    # Save embeddings for separate visualization
    npz_path = save_embeddings(args.save_dir, attn_model, args.max_len)
    pos_snaps.append(attn_model.pos.weight.detach().clone().cpu())
    plot_position_movement(pos_snaps)
    print(f"\nSaved embeddings -> {npz_path}")
    plot_pos_similarity(attn_model.pos.weight.detach(), title="Positional embedding cosine similarity")
    # Inspect a few examples
    for _ in range(6):
        ex = make_example(rng)
        input_ids, attn_mask, _ = encode(ex, args.max_len)
        input_ids_b = input_ids.unsqueeze(0).to(device)
        attn_mask_b = attn_mask.unsqueeze(0).to(device)

        alpha_ctx_row_good, alpha_ctx_row_bad, alpha_pool = attn_model.inspect(input_ids_b, attn_mask_b)
        toks = input_ids.detach().cpu().numpy().tolist()

        print("\nExample:")
        print(" query_aspect:", ex.query_aspect, "label:", ex.label)
        print(" tokens:", " ".join(ex.tokens))
        if len(alpha_ctx_row_good) > 0:
            print(" encoder self-attn ('good' token attends to...):")
            print(pretty(toks, alpha_ctx_row_good))
        if len(alpha_ctx_row_bad) > 0:
            print(" encoder self-attn ('bad' token attends to...):")
            print(pretty(toks, alpha_ctx_row_bad))
        print(" pooling attn (where we read out from):")
        print(pretty(toks, alpha_pool.numpy().tolist()))


if __name__ == "__main__":
    main()
