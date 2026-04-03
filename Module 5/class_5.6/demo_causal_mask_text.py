import math
import torch
import matplotlib.pyplot as plt


def make_causal_mask(seq_len: int) -> torch.Tensor:
    """
    Returns [T, T] matrix with:
      0 on allowed positions
      -inf on disallowed positions
    suitable for adding to attention logits
    """
    mask = torch.full((seq_len, seq_len), float("-inf"))
    mask = torch.triu(mask, diagonal=1)  # upper triangular above diagonal is blocked
    return mask


def softmax_attention(Q: torch.Tensor, K: torch.Tensor, causal: bool = False) -> torch.Tensor:
    """
    Q, K: [T, D]
    returns attention weights [T, T]
    """
    d = Q.size(-1)
    logits = (Q @ K.T) / math.sqrt(d)

    if causal:
        logits = logits + make_causal_mask(Q.size(0))

    weights = torch.softmax(logits, dim=-1)
    return weights


def print_token_attention(tokens, weights, title):
    print(f"\n{title}")
    print("-" * len(title))
    for i, tok in enumerate(tokens):
        row = weights[i].tolist()
        pretty = ", ".join(f"{tokens[j]}:{row[j]:.2f}" for j in range(len(tokens)))
        print(f"Query token '{tok}' attends to -> {pretty}")


def plot_heatmap(weights: torch.Tensor, tokens, title: str):
    plt.figure(figsize=(6, 5))
    plt.imshow(weights.numpy(), aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(tokens)), tokens, rotation=45, ha="right")
    plt.yticks(range(len(tokens)), tokens)
    plt.title(title)
    plt.xlabel("Key positions")
    plt.ylabel("Query positions")
    plt.tight_layout()


def main():
    torch.manual_seed(0)

    tokens = ["[CLS]", "the", "cat", "sat", "[SEP]"]
    T = len(tokens)
    D = 8

    # Random token representations for demonstration
    X = torch.randn(T, D)

    # Simple learned projections
    Wq = torch.randn(D, D)
    Wk = torch.randn(D, D)

    Q = X @ Wq
    K = X @ Wk

    full_weights = softmax_attention(Q, K, causal=False)
    causal_weights = softmax_attention(Q, K, causal=True)

    print_token_attention(tokens, full_weights, "Full self-attention (encoder-style)")
    print_token_attention(tokens, causal_weights, "Causal self-attention (decoder-style)")

    print("\nCausal mask matrix (0 means allowed, -inf means blocked):")
    print(make_causal_mask(T))

    plot_heatmap(full_weights, tokens, "Full self-attention")
    plot_heatmap(causal_weights, tokens, "Causal self-attention")
    plt.show()


if __name__ == "__main__":
    main()