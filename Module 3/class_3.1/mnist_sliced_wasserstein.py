#!/usr/bin/env python3
# © 2025 Ankur Mohan

import argparse
import math
import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from train_mnist_cnn import MnistCNN, MODEL_PATH
import matplotlib.pyplot as plt
import random
from pathlib import Path
from PIL import Image
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = Path("../../Module 2/data")
OOD_DIR = DATA_DIR / "ood_mnist_digits"

def load_ood_images():
    # Return list of (PIL_image, label_str)
    imgs = []
    for ddir in OOD_DIR.iterdir():
        if not ddir.is_dir():
            continue
        label = ddir.name  # "10", "11", etc.
        for p in ddir.glob("*.png"):
            imgs.append((p, label))
    random.shuffle(imgs)
    return imgs


# -----------------------------
# 3) 1D Wasserstein (by hand) for unequal sample sizes
# -----------------------------
def wasserstein_1d_sorted(a_sorted: torch.Tensor, b_sorted: torch.Tensor) -> torch.Tensor:
    """
    Compute 1D Wasserstein-1 distance between two empirical distributions.
    Works for unequal sample sizes via quantile interpolation.

    a_sorted: (Na,) sorted
    b_sorted: (Nb,) sorted
    returns scalar tensor
    """
    Na = a_sorted.numel()
    Nb = b_sorted.numel()

    # Evaluate both inverse CDFs at a shared grid of quantiles
    # Grid size = Na + Nb is a good compromise (accurate, still cheap)
    K = Na + Nb
    q = torch.linspace(0.0, 1.0, steps=K, device=a_sorted.device)

    def quantile_interp(x_sorted, q):
        n = x_sorted.numel()
        # positions in [0, n-1]
        pos = q * (n - 1)
        lo = torch.floor(pos).long()
        hi = torch.clamp(lo + 1, max=n - 1)
        w = (pos - lo.float())
        return (1 - w) * x_sorted[lo] + w * x_sorted[hi]

    qa = quantile_interp(a_sorted, q)
    qb = quantile_interp(b_sorted, q)

    # Wasserstein-1 in 1D = integral |F^{-1}_a(q) - F^{-1}_b(q)| dq
    # Approximate integral via mean over quantiles (uniform grid)
    return torch.mean(torch.abs(qa - qb))


# -----------------------------
# 4) Sliced Wasserstein Distance (SWD)
# -----------------------------
@torch.no_grad()
def sliced_wasserstein_distance(X: torch.Tensor,
                                Y: torch.Tensor,
                                num_projections: int = 128,
                                seed: int = 0) -> float:
    """
    X: (N, D) reference embeddings
    Y: (M, D) production embeddings
    SWD = mean over random directions of 1D Wasserstein distance of projections.
    """
    assert X.dim() == 2 and Y.dim() == 2 and X.size(1) == Y.size(1)
    device = X.device
    D = X.size(1)

    g = torch.Generator(device=device)
    g.manual_seed(seed)

    # random directions on unit sphere
    dirs = torch.randn(num_projections, D, generator=g, device=device)
    dirs = dirs / (dirs.norm(dim=1, keepdim=True) + 1e-12)

    # project
    Xp = X @ dirs.t()  # (N, P)
    Yp = Y @ dirs.t()  # (M, P)

    # compute 1D Wasserstein for each projection
    dists = []
    for p in range(num_projections):
        a = torch.sort(Xp[:, p]).values
        b = torch.sort(Yp[:, p]).values
        dists.append(wasserstein_1d_sorted(a, b))

    return torch.mean(torch.stack(dists)).item()


# -----------------------------
# 5) Embedding extraction
# -----------------------------
@torch.no_grad()
def collect_embeddings(model: nn.Module,
                       loader,
                       device: str,
                       max_batches: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns embeddings (N,D) and labels (N,)
    """
    model.eval()
    embs = []
    labels = []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        logits, emb = model(x)
        embs.append(emb.detach().float().cpu())
        labels.append(y.detach().cpu())
    return torch.cat(embs, dim=0), torch.cat(labels, dim=0)

def load_model():
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model = MnistCNN(embedding_dim=ckpt["embedding_dim"])
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()
    return model

def img_to_tensor(img: Image.Image):
    # Convert to tensor with same normalization as MNIST
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    return transform(img)

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--train-epochs", type=int, default=1)
    ap.add_argument("--ref-batches", type=int, default=50, help="How many train batches for reference embeddings")
    ap.add_argument("--prod-batches", type=int, default=20, help="How many test batches per production slice")
    ap.add_argument("--num-projections", type=int, default=128)
    ap.add_argument("--ood-strength", type=float, default=1.0, help="Increase to make OOD more severe")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", action="store_false")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = DEVICE

    tf = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST("data", train=True, download=True, transform=tf)
    test_ds  = datasets.MNIST("data", train=False, download=True, transform=tf)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=2,
                                              pin_memory=True)
    test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = load_model().to(device)

    # Reference embeddings from training distribution
    ref_embs_cpu, _ = collect_embeddings(model, train_loader, device, max_batches=args.ref_batches)
    ref_embs = ref_embs_cpu.to(device)

    # Production in-dist: MNIST test
    prod_embs_cpu, prod_y_cpu = collect_embeddings(model, test_loader, device, max_batches=args.prod_batches)
    prod_embs = prod_embs_cpu.to(device)

    # Production OOD: generate from MNIST test batches
    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed + 123)

    ood_embs_list = []
    ood_count = 0
    model.eval()
    ood_imgs = load_ood_images()
    for j, (path, label_str) in enumerate(ood_imgs):
        img = Image.open(path).convert("L")
        x = img_to_tensor(img)
        x_batch = x.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits, embedding = model(x_batch)
            ood_embs_list.append(embedding.detach().float().cpu())
            ood_count += x.size(0)


    ood_embs = torch.cat(ood_embs_list, dim=0).to(device)

    # Compute SWD
    t0 = time.time()
    swd_in = sliced_wasserstein_distance(ref_embs, prod_embs, num_projections=args.num_projections, seed=args.seed)
    t1 = time.time()
    swd_ood = sliced_wasserstein_distance(ref_embs, ood_embs, num_projections=args.num_projections, seed=args.seed)
    t2 = time.time()

    print("\n=== Sliced Wasserstein Drift on Embeddings ===")
    print(f"Reference embeddings: {tuple(ref_embs.shape)}")
    print(f"Prod (MNIST test) embeddings: {tuple(prod_embs.shape)}")
    print(f"Prod (OOD) embeddings: {tuple(ood_embs.shape)}")
    print(f"SWD(ref, prod_mnist) = {swd_in:.6f}  (time {t1 - t0:.2f}s)")
    print(f"SWD(ref, prod_ood)   = {swd_ood:.6f}  (time {t2 - t1:.2f}s)")
    print(f"OOD strength = {args.ood_strength}")

    if args.plot:
        plt.figure()
        plt.bar(["prod_mnist", "prod_ood"], [swd_in, swd_ood])
        plt.ylabel("Sliced Wasserstein Distance (embeddings)")
        plt.title("Embedding Drift: Reference vs Production Slices")
        plt.show()


if __name__ == "__main__":
    main()
