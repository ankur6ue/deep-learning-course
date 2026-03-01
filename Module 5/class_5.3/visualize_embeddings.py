"""
visualize_embeddings.py

Loads embeddings saved by attention_binding_demo_v2.py and shows:
- cosine similarities among selected tokens
- nearest neighbors by cosine similarity
- 2D PCA scatter plot of token embeddings (saved as PNG)

Run:
  python visualize_embeddings.py --npz outputs/embeddings_demo_v2.npz
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))


def topk_neighbors(emb: np.ndarray, vocab: List[str], token: str, k: int = 5) -> List[Tuple[str, float]]:
    idx = vocab.index(token)
    X = normalize_rows(emb)
    sims = X @ X[idx]
    order = np.argsort(-sims)
    out = []
    for j in order[1:k+1]:
        out.append((vocab[int(j)], float(sims[int(j)])))
    return out


def pca_2d(X: np.ndarray) -> np.ndarray:
    # center
    Xc = X - X.mean(axis=0, keepdims=True)
    # SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=False, default='outputs/embeddings_demo_v2.npz')
    ap.add_argument("--out_png", type=str, default="token_embeddings_pca.png")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    vocab = list(data["vocab"])
    token_emb = data["token_emb"]  # [V, d]

    print("Loaded:", args.npz)
    print("Vocab:", vocab)

    # 1) Cosine sanity checks
    pairs = [
        ("good", "bad"),
        ("food", "service"),
        ("food", "good"),
        ("service", "good"),
        ("food", "bad"),
        ("service", "bad"),
    ]
    print("\nCosine similarities (token embeddings):")
    for a, b in pairs:
        ca = token_emb[vocab.index(a)]
        cb = token_emb[vocab.index(b)]
        print(f"  cos({a:7s}, {b:7s}) = {cosine(ca, cb): .4f}")

    # 2) Nearest neighbors
    print("\nNearest neighbors (cosine) among tokens:")
    for t in ["good", "bad", "food", "service"]:
        nbrs = topk_neighbors(token_emb, vocab, t, k=args.k)
        print(f"  {t}: " + ", ".join([f"{w}({s:.3f})" for w, s in nbrs]))

    # 3) PCA plot
    Z = pca_2d(token_emb)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(Z[:, 0], Z[:, 1])

    for i, w in enumerate(vocab):
        ax.text(Z[i, 0], Z[i, 1], str(w), fontsize=10)

    ax.set_title("Token embedding PCA (2D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.tight_layout()

    out_png = Path(args.out_png)
    fig.savefig(out_png)
    print(f"\nSaved PCA plot -> {out_png.resolve()}")

if __name__ == "__main__":
    main()
