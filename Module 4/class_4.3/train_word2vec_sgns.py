#!/usr/bin/env python
"""
Minimal word2vec Skip-gram with Negative Sampling (SGNS) in PyTorch.

- Reads a plain text file (e.g., text8) from disk
- Tokenizes, builds vocab
- Creates (center, context) pairs with a sliding window
- Trains word embeddings using SGNS

Usage:
  python train_word2vec_sgns.py

You may want to start with a small corpus or subset for speed.
"""

from __future__ import annotations
import math
import random
from collections import Counter
from typing import List, Dict, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Config
# -----------------------------

DATA_PATH = "../data/text8"      # path to your corpus .txt file
MIN_FREQ = 5                  # minimum word frequency to keep in vocab
MAX_VOCAB_SIZE = 50000        # cap vocab size for speed
EMBED_DIM = 128
WINDOW_SIZE = 5               # max context window size
NUM_NEGATIVES = 5             # negatives per positive
BATCH_SIZE = 512
EPOCHS = 6
LR = 0.002
SUBSAMPLE_T = 1e-5            # subsampling threshold; set None to disable
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42


# -----------------------------
# 1. Tokenization / corpus loading
# -----------------------------

def simple_tokenize(text: str) -> List[str]:
    # Very simple tokenizer: lowercase + split on whitespace.
    # You can make this more sophisticated if you want.
    text = text.lower()
    return text.split()


def load_corpus(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    tokens = simple_tokenize(text)
    return tokens


# -----------------------------
# 2. Vocab and subsampling
# -----------------------------

class Vocab:
    def __init__(self, tokens: List[str], min_freq: int, max_size: int):
        counter = Counter(tokens)
        # Drop low-frequency tokens
        freq_pairs = [(w, c) for w, c in counter.items() if c >= min_freq]
        # Sort by frequency
        freq_pairs.sort(key=lambda x: x[1], reverse=True)
        # Cap vocab size
        freq_pairs = freq_pairs[:max_size]

        self.itos = ["<PAD>", "<UNK>"]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

        for w, c in freq_pairs:
            self.stoi[w] = len(self.itos)
            self.itos.append(w)

        self.freqs = torch.tensor([0.0] * len(self.itos))
        for w, idx in self.stoi.items():
            if w in counter:
                self.freqs[idx] = counter[w]

        self.total_tokens = sum(counter.values())

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens: List[str]) -> List[int]:
        unk_idx = self.stoi["<UNK>"]
        return [self.stoi.get(t, unk_idx) for t in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        return [self.itos[i] for i in ids]


def subsample_tokens(token_ids: List[int], freqs: torch.Tensor, total_tokens: int, t: float) -> List[int]:
    """
    Subsample high-frequency tokens as in Mikolov et al.

        P(drop w) = 1 - sqrt(t / f_w)

    where f_w is the empirical frequency ratio of w.

    token_ids: list of word indices
    freqs: tensor [vocab_size] of raw counts
    total_tokens: total number of tokens in original corpus
    t: threshold (e.g. 1e-5)
    """
    if t is None:
        return token_ids

    freqs_ratio = freqs / total_tokens  # [vocab_size]

    kept = []
    for idx in token_ids:
        f = freqs_ratio[idx].item()
        p_drop = 1.0 - math.sqrt(t / f) if f > 0 else 0.0
        if random.random() > p_drop:
            kept.append(idx)
    return kept


# -----------------------------
# 3. Dataset: generate (center, context) skip-gram pairs
# -----------------------------

class SkipGramDataset(Dataset):
    def __init__(
        self,
        token_ids: List[int],
        window_size: int,
    ):
        self.pairs: List[Tuple[int, int]] = []
        for i, center in enumerate(token_ids):
            # Random window size in [1, window_size]
            ws = random.randint(1, window_size)
            start = max(0, i - ws)
            end = min(len(token_ids), i + ws + 1)
            for j in range(start, end):
                if j == i:
                    continue
                context = token_ids[j]
                self.pairs.append((center, context))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[int, int]:
        return self.pairs[idx]


# -----------------------------
# 4. Word2Vec SGNS model
# -----------------------------

class Word2VecSGNS(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_dim)
        self.out_embed = nn.Embedding(vocab_size, embed_dim)

        # Initialize embeddings
        initrange = 0.5 / embed_dim
        self.in_embed.weight.data.uniform_(-initrange, initrange)
        self.out_embed.weight.data.uniform_(-0, 0)  # as in original word2vec

    def forward(
        self,
        center_words: torch.Tensor,        # [B]
        pos_context_words: torch.Tensor,   # [B]
        neg_context_words: torch.Tensor,   # [B, K]
    ) -> torch.Tensor:
        """
        center_words: indices of center words
        pos_context_words: indices of positive context words
        neg_context_words: indices of negative samples

        Returns: loss (scalar)
        """
        batch_size = center_words.size(0)
        embed_center = self.in_embed(center_words)          # [B, D]
        embed_pos = self.out_embed(pos_context_words)       # [B, D]
        embed_neg = self.out_embed(neg_context_words)       # [B, K, D]

        # Positive score: u_o^T v_c
        pos_score = torch.sum(embed_center * embed_pos, dim=1)  # [B]
        pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-10)

        # Negative score: u_n^T v_c
        # embed_center: [B, D] -> [B, 1, D]
        embed_center_expanded = embed_center.unsqueeze(1)        # [B, 1, D]
        neg_score = torch.bmm(embed_neg.neg(), embed_center_expanded.transpose(1, 2)).squeeze(2)  # [B, K]
        # Equivalent: -u_n^T v_c so we can use sigmoid
        neg_loss = -torch.sum(torch.log(torch.sigmoid(neg_score) + 1e-10), dim=1)  # [B]

        loss = (pos_loss + neg_loss).mean()
        return loss

    def get_input_embeddings(self) -> torch.Tensor:
        return self.in_embed.weight.data


# -----------------------------
# 5. Negative sampler
# -----------------------------

class NegativeSampler:
    def __init__(self, freqs: torch.Tensor, num_negatives: int):
        """
        freqs: tensor [vocab_size] of raw counts
        """
        self.num_negatives = num_negatives
        # Compute unigram^0.75 distribution
        probs = freqs.float().pow(0.75)
        probs = probs / probs.sum()
        self.probs = probs

    def sample(self, batch_size: int) -> torch.Tensor:
        """
        Return [B, K] tensor of negative word indices.
        """
        negs = torch.multinomial(
            self.probs,
            num_samples=batch_size * self.num_negatives,
            replacement=True,
        )
        return negs.view(batch_size, self.num_negatives)


# -----------------------------
# 6. Training loop
# -----------------------------

def main():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    print("Loading corpus...")
    tokens = load_corpus(DATA_PATH)
    print(f"Total tokens in raw corpus: {len(tokens):,}")

    print("Building vocab...")
    vocab = Vocab(tokens, MIN_FREQ, MAX_VOCAB_SIZE)
    print(f"Vocab size: {len(vocab):,}")

    print("Encoding tokens...")
    token_ids = vocab.encode(tokens)
    print(f"Encoded token sequence length: {len(token_ids):,}")

    if SUBSAMPLE_T is not None:
        print("Subsampling frequent words...")
        token_ids = subsample_tokens(token_ids, vocab.freqs, vocab.total_tokens, SUBSAMPLE_T)
        print(f"After subsampling, token sequence length: {len(token_ids):,}")

    print("Building Skip-gram dataset...")
    sg_dataset = SkipGramDataset(token_ids, WINDOW_SIZE)
    print(f"Number of (center, context) pairs: {len(sg_dataset):,}")

    dataloader = DataLoader(
        sg_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    model = Word2VecSGNS(len(vocab), EMBED_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    neg_sampler = NegativeSampler(vocab.freqs, NUM_NEGATIVES)

    print("Starting training...")
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for step, (center, context) in enumerate(dataloader, start=1):
            center = center.to(DEVICE)
            context = context.to(DEVICE)
            negs = neg_sampler.sample(center.size(0)).to(DEVICE)

            optimizer.zero_grad()
            loss = model(center, context, negs)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if step % 500 == 0:
                avg_loss = total_loss / 500
                print(f"Epoch {epoch} Step {step} - Avg loss: {avg_loss:.4f}")
                total_loss = 0.0

        print(f"Epoch {epoch} done.")

    # Save embeddings
    embeddings = model.get_input_embeddings().cpu()
    torch.save(
        {
            "embeddings": embeddings,
            "itos": vocab.itos,
            "stoi": vocab.stoi,
        },
        "word2vec_sgns_embeddings.pt",
    )
    print("Saved embeddings to word2vec_sgns_embeddings.pt")

    # Show nearest neighbors for a few example words
    def show_neighbors(word: str, top_k: int = 5):
        if word not in vocab.stoi:
            print(f"Word {word!r} not in vocab")
            return
        idx = vocab.stoi[word]
        w_emb = embeddings[idx]  # [D]
        # cosine similarity to all others
        sims = torch.mv(embeddings, w_emb) / (
            embeddings.norm(dim=1) * (w_emb.norm() + 1e-9)
        )
        sims[idx] = -1e9  # exclude itself
        topk = torch.topk(sims, k=top_k)
        print(f"Neighbors for {word!r}:")
        for score, j in zip(topk.values, topk.indices):
            print(f"  {vocab.itos[j]}  (cos={float(score):.3f})")

    for w in ["king", "queen", "man", "woman", "hike", "san"]:
        show_neighbors(w)


if __name__ == "__main__":
    main()
