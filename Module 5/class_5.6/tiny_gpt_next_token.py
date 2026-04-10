import math
import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Tiny tokenizer / vocabulary
# ============================================================

SPECIAL_TOKENS = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]


def build_vocab(texts: List[str]) -> Tuple[dict, dict]:
    words = set()
    for text in texts:
        for w in text.lower().split():
            words.add(w)

    idx_to_token = SPECIAL_TOKENS + sorted(words)
    token_to_idx = {tok: i for i, tok in enumerate(idx_to_token)}
    idx_to_token_map = {i: tok for tok, i in token_to_idx.items()}
    return token_to_idx, idx_to_token_map


def tokenize(text: str, token_to_idx: dict) -> List[int]:
    tokens = ["[BOS]"] + text.lower().split() + ["[EOS]"]
    return [token_to_idx.get(tok, token_to_idx["[UNK]"]) for tok in tokens]


def detokenize(token_ids: List[int], idx_to_token: dict) -> str:
    toks = []
    for i in token_ids:
        tok = idx_to_token[i]
        if tok in {"[PAD]", "[BOS]"}:
            continue
        if tok == "[EOS]":
            break
        toks.append(tok)
    return " ".join(toks)


def pad_batch(batch_ids: List[List[int]], pad_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        input_ids: [B, T]
        padding_mask: [B, T], 1 for real tokens, 0 for padding
    """
    max_len = max(len(x) for x in batch_ids)
    padded = []
    padding_mask = []
    for ids in batch_ids:
        padded_ids = ids + [pad_idx] * (max_len - len(ids))
        padded_mask = [1] * len(ids) + [0] * (max_len - len(ids))
        padded.append(padded_ids)
        padding_mask.append(padded_mask)
    return torch.tensor(padded, dtype=torch.long), torch.tensor(padding_mask, dtype=torch.long)


# ============================================================
# 2. Toy language modeling data
# ============================================================

TEXTS = [
    "the cat slept",
    "the cat sat on the mat",
    "the cat quietly sat on the warm mat by the window",
    "the dog ran",
    "the dog chased the cat through the garden",
    "the dog happily chased the small cat through the wet garden after lunch",
    "the movie was great",
    "the movie was surprisingly thoughtful and emotionally engaging",
    "the movie was unexpectedly long but visually stunning and emotionally powerful",
    "the food was awful",
    "the food was absolutely delicious and beautifully presented",
    "the food at the restaurant was delicious but the service was painfully slow",
    "the book was boring",
    "the book was interesting and easy to follow",
    "the book on medieval history was detailed, thoughtful, and full of memorable examples",
    "the weather changed",
    "the weather was sunny in the morning but rainy by late afternoon",
    "the weather was cold windy and unpleasant for most of the day",
    "the actor performed well",
    "the actor gave a deeply moving performance in the final scene",
    "the actor gave a poor performance that weakened an otherwise excellent film",
    "dinner tasted bad",
    "the dinner tasted wonderful after a long day of travel",
    "the dinner tasted wonderful and the dessert was even better than expected",
]

TOKEN_TO_IDX, IDX_TO_TOKEN = build_vocab(TEXTS)

PAD_IDX = TOKEN_TO_IDX["[PAD]"]
BOS_IDX = TOKEN_TO_IDX["[BOS]"]
EOS_IDX = TOKEN_TO_IDX["[EOS]"]
UNK_IDX = TOKEN_TO_IDX["[UNK]"]


# ============================================================
# 3. Causal mask
# ============================================================

def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Returns [T, T] with:
      0 on allowed positions
      -inf on disallowed positions

    Position i may only attend to positions <= i.
    Suitable for additive masking of attention logits.
    """
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask


# ============================================================
# 4. Model
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, T, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 128,
        max_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)

        block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        # Decoder-only behavior is enforced via the causal mask.
        self.blocks = nn.TransformerEncoder(block, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [B, T]
            padding_mask: [B, T], 1 for real tokens, 0 for padding

        Returns:
            logits: [B, T, V]
        """
        x = self.token_emb(input_ids)
        x = self.pos_enc(x)

        seq_len = input_ids.size(1)
        causal_mask = make_causal_mask(seq_len, device=input_ids.device)

        # PyTorch expects True for positions to ignore.
        key_padding_mask = padding_mask == 0

        h = self.blocks(
            x,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
        )
        h = self.norm(h)
        logits = self.lm_head(h)
        return logits


# ============================================================
# 5. Training data preparation
# ============================================================

def make_language_model_batch(texts: List[str], token_to_idx: dict):
    """
    For decoder-only next-token prediction:

      full sequence: [BOS] w1 w2 ... wN [EOS]

      input  = [BOS] w1 w2 ... wN
      target = [w1]  w2 ... wN [EOS]

    Returns:
        input_ids: [B, T]
        padding_mask: [B, T]
        target_ids: [B, T]
    """
    tokenized = [tokenize(t, token_to_idx) for t in texts]

    inputs = [ids[:-1] for ids in tokenized]
    targets = [ids[1:] for ids in tokenized]

    input_ids, padding_mask = pad_batch(inputs, PAD_IDX)
    target_ids, _ = pad_batch(targets, PAD_IDX)
    return input_ids, padding_mask, target_ids


def iterate_minibatches(items: List, batch_size: int):
    items = items[:]
    random.shuffle(items)
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


# ============================================================
# 6. Generation
# ============================================================

@torch.no_grad()
def generate_greedy(
    model: nn.Module,
    prompt: str,
    token_to_idx: dict,
    idx_to_token: dict,
    max_new_tokens: int = 10,
    device: str = "cpu",
) -> str:
    model.eval()

    generated = [BOS_IDX] + [
        token_to_idx.get(tok, token_to_idx["[UNK]"])
        for tok in prompt.lower().split()
    ]

    for _ in range(max_new_tokens):
        input_ids = torch.tensor([generated], dtype=torch.long, device=device)
        padding_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

        logits = model(input_ids, padding_mask)   # [1, T, V]
        next_token_logits = logits[0, -1]
        next_token_id = int(next_token_logits.argmax().item())

        generated.append(next_token_id)

        if next_token_id == EOS_IDX:
            break

    return detokenize(generated, idx_to_token)


@torch.no_grad()
def generate_with_sampling(
    model: nn.Module,
    prompt: str,
    token_to_idx: dict,
    idx_to_token: dict,
    max_new_tokens: int = 10,
    temperature: float = 1.0,
    device: str = "cpu",
) -> str:
    model.eval()

    generated = [BOS_IDX] + [
        token_to_idx.get(tok, token_to_idx["[UNK]"])
        for tok in prompt.lower().split()
    ]

    for _ in range(max_new_tokens):
        input_ids = torch.tensor([generated], dtype=torch.long, device=device)
        padding_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

        logits = model(input_ids, padding_mask)
        next_token_logits = logits[0, -1] / temperature
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token_id = int(torch.multinomial(probs, num_samples=1).item())

        generated.append(next_token_id)

        if next_token_id == EOS_IDX:
            break

    return detokenize(generated, idx_to_token)


# ============================================================
# 7. Evaluation helper
# ============================================================

@torch.no_grad()
def next_token_probe(
    model: nn.Module,
    prefix_tokens: List[str],
    token_to_idx: dict,
    idx_to_token: dict,
    device: str,
    topk: int = 5,
):
    """
    Show the model's next-token distribution after a given prefix.
    """
    model.eval()

    ids = [BOS_IDX] + [token_to_idx.get(t, UNK_IDX) for t in prefix_tokens]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    padding_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

    logits = model(input_ids, padding_mask)
    next_logits = logits[0, -1]
    probs = torch.softmax(next_logits, dim=-1)

    values, indices = torch.topk(probs, k=topk)

    print(f"\nPrefix: {' '.join(prefix_tokens)}")
    print("Top next-token predictions:")
    for p, idx in zip(values.tolist(), indices.tolist()):
        print(f"  {idx_to_token[idx]:15s}  prob={p:.4f}")


# ============================================================
# 8. Main
# ============================================================

def main():
    random.seed(42)
    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = TinyGPT(
        vocab_size=len(TOKEN_TO_IDX),
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_ff=128,
        max_len=32,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 80
    batch_size = 4

    print("\n=== Training decoder-only model with next-token prediction ===")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0

        for batch in iterate_minibatches(TEXTS, batch_size):
            input_ids, padding_mask, target_ids = make_language_model_batch(batch, TOKEN_TO_IDX)
            input_ids = input_ids.to(device)
            padding_mask = padding_mask.to(device)
            target_ids = target_ids.to(device)

            logits = model(input_ids, padding_mask)  # [B, T, V]

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                ignore_index=PAD_IDX,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d} | LM loss = {total_loss / max(steps, 1):.4f}")

    print("\n=== Next-token probes ===")
    next_token_probe(model, ["the", "movie", "was"], TOKEN_TO_IDX, IDX_TO_TOKEN, device)
    next_token_probe(model, ["the", "food", "was"], TOKEN_TO_IDX, IDX_TO_TOKEN, device)
    next_token_probe(model, ["the", "dog"], TOKEN_TO_IDX, IDX_TO_TOKEN, device)

    print("\n=== Greedy generation ===")
    prompts = [
        "the movie was",
        "the food was",
        "the cat",
        "the actor gave",
    ]
    for prompt in prompts:
        out = generate_greedy(
            model,
            prompt=prompt,
            token_to_idx=TOKEN_TO_IDX,
            idx_to_token=IDX_TO_TOKEN,
            max_new_tokens=8,
            device=device,
        )
        print(f"Prompt: {prompt:20s} -> {out}")

    print("\n=== Sampled generation ===")
    for prompt in prompts:
        out = generate_with_sampling(
            model,
            prompt=prompt,
            token_to_idx=TOKEN_TO_IDX,
            idx_to_token=IDX_TO_TOKEN,
            max_new_tokens=8,
            temperature=0.8,
            device=device,
        )
        print(f"Prompt: {prompt:20s} -> {out}")


if __name__ == "__main__":
    main()