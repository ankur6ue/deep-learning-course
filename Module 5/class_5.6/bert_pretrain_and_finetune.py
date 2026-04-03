import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Tiny tokenizer / vocabulary
# ============================================================

SPECIAL_TOKENS = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]


def build_vocab(sentences: List[str]) -> Tuple[dict, dict]:
    vocab_words = set()
    for s in sentences:
        for w in s.lower().split():
            vocab_words.add(w)

    idx_to_token = SPECIAL_TOKENS + sorted(vocab_words)
    token_to_idx = {tok: i for i, tok in enumerate(idx_to_token)}
    return token_to_idx, {i: tok for tok, i in token_to_idx.items()}


def tokenize(sentence: str, token_to_idx: dict) -> List[int]:
    tokens = ["[CLS]"] + sentence.lower().split() + ["[SEP]"]
    return [token_to_idx.get(tok, token_to_idx["[UNK]"]) for tok in tokens]


def pad_batch(batch_ids: List[List[int]], pad_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        input_ids: [B, T]
        attention_mask: [B, T] with 1 for real tokens, 0 for padding
    """
    max_len = max(len(x) for x in batch_ids)
    padded = []
    attn_mask = []
    for ids in batch_ids:
        padded_ids = ids + [pad_idx] * (max_len - len(ids))
        mask = [1] * len(ids) + [0] * (max_len - len(ids))
        padded.append(padded_ids)
        attn_mask.append(mask)
    return torch.tensor(padded, dtype=torch.long), torch.tensor(attn_mask, dtype=torch.long)


# ============================================================
# 2. Data
# ============================================================

# Unlabeled corpus for MLM pretraining
UNLABELED_TEXT = [
    "the movie was great",
    "the movie was terrible",
    "the food was delicious",
    "the food was awful",
    "the book was interesting",
    "the book was boring",
    "the service was excellent",
    "the service was slow",
    "the weather was sunny",
    "the weather was rainy",
    "the cat sat on the mat",
    "the dog sat on the rug",
    "the actor gave a great performance",
    "the actor gave a poor performance",
    "the dinner tasted wonderful",
    "the dinner tasted bad",
]

# Labeled data for sentence classification
LABELED_DATA = [
    ("the movie was great", 1),
    ("the movie was terrible", 0),
    ("the food was delicious", 1),
    ("the food was awful", 0),
    ("the book was interesting", 1),
    ("the book was boring", 0),
    ("the service was excellent", 1),
    ("the service was slow", 0),
    ("the actor gave a great performance", 1),
    ("the actor gave a poor performance", 0),
    ("the dinner tasted wonderful", 1),
    ("the dinner tasted bad", 0),
]

ALL_TEXT = UNLABELED_TEXT + [x for x, _ in LABELED_DATA]
TOKEN_TO_IDX, IDX_TO_TOKEN = build_vocab(ALL_TEXT)

PAD_IDX = TOKEN_TO_IDX["[PAD]"]
CLS_IDX = TOKEN_TO_IDX["[CLS]"]
SEP_IDX = TOKEN_TO_IDX["[SEP]"]
MASK_IDX = TOKEN_TO_IDX["[MASK]"]
UNK_IDX = TOKEN_TO_IDX["[UNK]"]


# ============================================================
# 3. Tiny BERT-like model
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


class TinyBertEncoder(nn.Module):
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

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        input_ids: [B, T]
        attention_mask: [B, T], 1 for real tokens, 0 for pad
        returns hidden states [B, T, D]
        """
        x = self.token_emb(input_ids)
        x = self.pos_enc(x)

        # src_key_padding_mask expects True for positions to ignore
        key_padding_mask = attention_mask == 0
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.norm(h)


class BertForMLM(nn.Module):
    def __init__(self, encoder: TinyBertEncoder, vocab_size: int):
        super().__init__()
        self.encoder = encoder
        d_model = encoder.token_emb.embedding_dim
        self.mlm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        h = self.encoder(input_ids, attention_mask)   # [B, T, D]
        logits = self.mlm_head(h)                     # [B, T, V]
        return logits


class BertForSequenceClassification(nn.Module):
    def __init__(self, encoder: TinyBertEncoder, num_classes: int = 2):
        super().__init__()
        self.encoder = encoder
        d_model = encoder.token_emb.embedding_dim
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        h = self.encoder(input_ids, attention_mask)   # [B, T, D]
        cls_hidden = h[:, 0, :]                       # [CLS] token representation
        logits = self.classifier(cls_hidden)          # [B, C]
        return logits


# ============================================================
# 4. Masked LM data creation
# ============================================================

def create_mlm_batch(
    sentences: List[str],
    token_to_idx: dict,
    mask_prob: float = 0.15
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        masked_input_ids: [B, T]
        attention_mask:   [B, T]
        mlm_labels:       [B, T], -100 where no MLM loss should be applied
    """
    tokenized = [tokenize(s, token_to_idx) for s in sentences]
    input_ids, attention_mask = pad_batch(tokenized, pad_idx=PAD_IDX)

    labels = input_ids.clone()
    masked_input_ids = input_ids.clone()

    special_ids = {PAD_IDX, CLS_IDX, SEP_IDX}

    for b in range(masked_input_ids.size(0)):
        for t in range(masked_input_ids.size(1)):
            token_id = masked_input_ids[b, t].item()

            # don't mask padding / special tokens
            if token_id in special_ids:
                labels[b, t] = -100
                continue

            if random.random() < mask_prob:
                # 80% replace with [MASK]
                # 10% replace with random token
                # 10% keep unchanged
                r = random.random()
                if r < 0.8:
                    masked_input_ids[b, t] = MASK_IDX
                elif r < 0.9:
                    masked_input_ids[b, t] = random.randint(len(SPECIAL_TOKENS), len(token_to_idx) - 1)
                else:
                    pass
            else:
                labels[b, t] = -100

    return masked_input_ids, attention_mask, labels


# ============================================================
# 5. Training helpers
# ============================================================

@dataclass
class TrainConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    pretrain_epochs: int = 40
    finetune_epochs: int = 40
    batch_size: int = 4
    lr_pretrain: float = 2e-3
    lr_finetune: float = 1e-3
    seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def iterate_minibatches(items: List, batch_size: int):
    shuffled = items[:]
    random.shuffle(shuffled)
    for i in range(0, len(shuffled), batch_size):
        yield shuffled[i:i + batch_size]


def evaluate_classifier(model: nn.Module, data: List[Tuple[str, int]], device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in iterate_minibatches(data, batch_size=4):
            texts = [x for x, _ in batch]
            labels = torch.tensor([y for _, y in batch], dtype=torch.long, device=device)

            tokenized = [tokenize(s, TOKEN_TO_IDX) for s in texts]
            input_ids, attention_mask = pad_batch(tokenized, pad_idx=PAD_IDX)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            logits = model(input_ids, attention_mask)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()

    return correct / total


# ============================================================
# 6. Main experiment
# ============================================================

def main():
    cfg = TrainConfig()
    set_seed(cfg.seed)
    device = cfg.device
    print(f"Using device: {device}")

    vocab_size = len(TOKEN_TO_IDX)

    # Shared encoder backbone
    encoder = TinyBertEncoder(
        vocab_size=vocab_size,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_ff=128,
        max_len=32,
    ).to(device)

    # --------------------------------------------------------
    # Stage A: Unsupervised pretraining with MLM
    # --------------------------------------------------------
    print("\n=== Stage A: MLM pretraining ===")
    mlm_model = BertForMLM(encoder, vocab_size=vocab_size).to(device)
    optimizer = torch.optim.Adam(mlm_model.parameters(), lr=cfg.lr_pretrain)

    for epoch in range(1, cfg.pretrain_epochs + 1):
        mlm_model.train()
        epoch_loss = 0.0
        steps = 0

        for batch in iterate_minibatches(UNLABELED_TEXT, cfg.batch_size):
            input_ids, attention_mask, mlm_labels = create_mlm_batch(batch, TOKEN_TO_IDX, mask_prob=0.15)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            mlm_labels = mlm_labels.to(device)

            logits = mlm_model(input_ids, attention_mask)  # [B, T, V]
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                mlm_labels.view(-1),
                ignore_index=-100,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            steps += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d} | MLM loss = {epoch_loss / max(steps, 1):.4f}")

    # --------------------------------------------------------
    # Probe MLM predictions on a masked sentence
    # --------------------------------------------------------
    print("\n=== MLM probe ===")
    mlm_model.eval()
    probe_sentence = "the movie was great"
    probe_ids = tokenize(probe_sentence, TOKEN_TO_IDX)
    # mask the word "great" (position before [SEP])
    probe_ids[-2] = MASK_IDX

    input_ids, attention_mask = pad_batch([probe_ids], pad_idx=PAD_IDX)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        logits = mlm_model(input_ids, attention_mask)  # [1, T, V]
        pred_id = logits[0, -2].argmax().item()

    print("Input tokens :", [IDX_TO_TOKEN[i] for i in input_ids[0].tolist()])
    print("Predicted masked token:", IDX_TO_TOKEN[pred_id])

    # --------------------------------------------------------
    # Stage B: Task-specific fine-tuning with [CLS] head
    # --------------------------------------------------------
    print("\n=== Stage B: Fine-tuning for classification using [CLS] ===")
    clf_model = BertForSequenceClassification(encoder, num_classes=2).to(device)
    optimizer = torch.optim.Adam(clf_model.parameters(), lr=cfg.lr_finetune)

    for epoch in range(1, cfg.finetune_epochs + 1):
        clf_model.train()
        epoch_loss = 0.0
        steps = 0

        for batch in iterate_minibatches(LABELED_DATA, cfg.batch_size):
            texts = [x for x, _ in batch]
            labels = torch.tensor([y for _, y in batch], dtype=torch.long, device=device)

            tokenized = [tokenize(s, TOKEN_TO_IDX) for s in texts]
            input_ids, attention_mask = pad_batch(tokenized, pad_idx=PAD_IDX)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            logits = clf_model(input_ids, attention_mask)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            steps += 1

        if epoch % 5 == 0 or epoch == 1:
            acc = evaluate_classifier(clf_model, LABELED_DATA, device=device)
            print(f"Epoch {epoch:02d} | CLS classification loss = {epoch_loss / max(steps, 1):.4f} | acc = {acc:.3f}")

    # --------------------------------------------------------
    # Final probes
    # --------------------------------------------------------
    print("\n=== Classification probes ===")
    clf_model.eval()
    test_sentences = [
        "the movie was great",
        "the movie was terrible",
        "the food was delicious",
        "the book was boring",
    ]

    with torch.no_grad():
        tokenized = [tokenize(s, TOKEN_TO_IDX) for s in test_sentences]
        input_ids, attention_mask = pad_batch(tokenized, pad_idx=PAD_IDX)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        logits = clf_model(input_ids, attention_mask)
        probs = logits.softmax(dim=-1)

    for sent, p in zip(test_sentences, probs):
        pred = p.argmax().item()
        label_name = "positive" if pred == 1 else "negative"
        print(f"{sent:30s} -> {label_name:8s} | probs = {p.cpu().numpy()}")


if __name__ == "__main__":
    main()