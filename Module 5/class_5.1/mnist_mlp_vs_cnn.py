"""
mnist_mlp_vs_cnn.py
© 2025 Ankur Mohan

Train an MLP and a small CNN on MNIST.
Show:
- CNN achieves similar (or better) accuracy with fewer parameters
- Intermediate feature maps become progressively more structured

Dependencies:
  pip install torch torchvision matplotlib
"""

import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# -------------------------
# Utilities
# -------------------------

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        loss_sum += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()

    return loss_sum / total, correct / total


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    loss_sum = 0.0
    total = 0
    correct = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * y.numel()
        total += y.numel()
        correct += (logits.argmax(dim=1) == y).sum().item()

    return loss_sum / total, correct / total


def plot_curves(history, outpath):
    epochs = list(range(1, len(history["train_acc"]) + 1))
    plt.figure()
    plt.plot(epochs, history["train_acc"], label="train acc")
    plt.plot(epochs, history["test_acc"], label="test acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def save_feature_maps_grid(feature_maps, outpath, max_channels=32, cols=8, title=None):
    """
    feature_maps: torch.Tensor shape (C, H, W), already on CPU
    Saves a grid of the first max_channels channels.
    """
    fm = feature_maps[:max_channels]  # (C, H, W)
    C, H, W = fm.shape
    rows = (C + cols - 1) // cols

    plt.figure(figsize=(cols * 1.6, rows * 1.6))
    if title:
        plt.suptitle(title)

    for idx in range(C):
        ax = plt.subplot(rows, cols, idx + 1)
        ax.imshow(fm[idx].numpy(), cmap="gray")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


# -------------------------
# Models
# -------------------------

class MLP(nn.Module):
    """
    Flatten -> big fully connected layers.
    This is intentionally larger to hit strong accuracy.
    """
    def __init__(self, hidden=1024, depth=2, dropout=0.1):
        super().__init__()
        layers = []
        in_dim = 28 * 28

        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden

        layers.append(nn.Linear(in_dim, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        return self.net(x)


class TinyCNN_GAP(nn.Module):
    """
    A small CNN with global average pooling (GAP).
    This avoids a large fully-connected layer, keeping params tiny.

    Conv -> ReLU -> Conv -> ReLU -> Conv -> ReLU -> GAP -> Linear(64->10)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # 28x28 -> 28x28
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 28x28 -> 28x28
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))      # (N,16,28,28)
        x = F.max_pool2d(x, 2)         # (N,16,14,14)

        x = F.relu(self.conv2(x))      # (N,32,14,14)
        x = F.max_pool2d(x, 2)         # (N,32,7,7)

        x = F.relu(self.conv3(x))      # (N,64,7,7)

        # Global average pooling over spatial dims
        x = x.mean(dim=(2, 3))         # (N,64)
        x = self.classifier(x)         # (N,10)
        return x


# -------------------------
# Feature map extraction
# -------------------------

@torch.no_grad()
def extract_feature_maps(cnn_model, x_single, device):
    """
    Returns a dict of feature maps from conv layers for a single image.
    x_single: (1,1,28,28) tensor
    """
    cnn_model.eval()
    x_single = x_single.to(device)

    fm = {}

    a1 = F.relu(cnn_model.conv1(x_single))      # (1,16,28,28)
    fm["conv1"] = a1.squeeze(0).cpu()           # (16,28,28)
    p1 = F.max_pool2d(a1, 2)                    # (1,16,14,14)

    a2 = F.relu(cnn_model.conv2(p1))            # (1,32,14,14)
    fm["conv2"] = a2.squeeze(0).cpu()           # (32,14,14)
    p2 = F.max_pool2d(a2, 2)                    # (1,32,7,7)

    a3 = F.relu(cnn_model.conv3(p2))            # (1,64,7,7)
    fm["conv3"] = a3.squeeze(0).cpu()           # (64,7,7)

    return fm


# -------------------------
# Main
# -------------------------

@dataclass
class Config:
    batch_size: int = 128
    epochs: int = 5
    lr: float = 1e-3
    seed: int = 0
    outdir: str = "mnist_outputs"


def main():
    cfg = Config()
    os.makedirs(cfg.outdir, exist_ok=True)

    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Standard MNIST normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Models
    mlp = MLP(hidden=1024, depth=2, dropout=0.1).to(device)
    cnn = TinyCNN_GAP().to(device)

    print("\nParameter counts:")
    print(f"  MLP params: {count_params(mlp):,}")
    print(f"  CNN params: {count_params(cnn):,}")

    # Optimizers
    opt_mlp = torch.optim.Adam(mlp.parameters(), lr=cfg.lr)
    opt_cnn = torch.optim.Adam(cnn.parameters(), lr=cfg.lr)

    # Train both and track history
    hist_mlp = {"train_acc": [], "test_acc": []}
    hist_cnn = {"train_acc": [], "test_acc": []}

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        mlp_train_loss, mlp_train_acc = train_one_epoch(mlp, train_loader, opt_mlp, device)
        mlp_test_loss,  mlp_test_acc  = evaluate(mlp, test_loader, device)

        cnn_train_loss, cnn_train_acc = train_one_epoch(cnn, train_loader, opt_cnn, device)
        cnn_test_loss,  cnn_test_acc  = evaluate(cnn, test_loader, device)

        hist_mlp["train_acc"].append(mlp_train_acc)
        hist_mlp["test_acc"].append(mlp_test_acc)

        hist_cnn["train_acc"].append(cnn_train_acc)
        hist_cnn["test_acc"].append(cnn_test_acc)

        dt = time.time() - t0
        print(f"\nEpoch {epoch}/{cfg.epochs}  ({dt:.1f}s)")
        print(f"  MLP: train acc={mlp_train_acc:.4f}  test acc={mlp_test_acc:.4f}")
        print(f"  CNN: train acc={cnn_train_acc:.4f}  test acc={cnn_test_acc:.4f}")

    # Save accuracy curves
    plot_curves(hist_mlp, os.path.join(cfg.outdir, "mlp_accuracy.png"))
    plot_curves(hist_cnn, os.path.join(cfg.outdir, "cnn_accuracy.png"))
    print("\nSaved accuracy plots to:", cfg.outdir)

    # -------------------------
    # Feature map visualization
    # -------------------------
    # Pick one sample from test set
    x0, y0 = test_ds[0]  # x0: (1,28,28)
    x0_batch = x0.unsqueeze(0)  # (1,1,28,28)

    # Also save original image (unnormalized view)
    # Undo normalization for display
    x0_display = (x0 * 0.3081 + 0.1307).clamp(0, 1).squeeze(0).numpy()
    plt.figure()
    plt.imshow(x0_display, cmap="gray")
    plt.title(f"Input image (label={y0})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.outdir, "input_image.png"), dpi=160)
    plt.close()

    fmaps = extract_feature_maps(cnn, x0_batch, device=device)

    save_feature_maps_grid(
        fmaps["conv1"],
        os.path.join(cfg.outdir, "featuremaps_conv1.png"),
        max_channels=16,
        cols=8,
        title="Conv1 feature maps (often edge / stroke detectors)"
    )
    save_feature_maps_grid(
        fmaps["conv2"],
        os.path.join(cfg.outdir, "featuremaps_conv2.png"),
        max_channels=32,
        cols=8,
        title="Conv2 feature maps (corners / curves / parts)"
    )
    save_feature_maps_grid(
        fmaps["conv3"],
        os.path.join(cfg.outdir, "featuremaps_conv3.png"),
        max_channels=32,
        cols=8,
        title="Conv3 feature maps (higher-level digit parts / templates)"
    )

    print("Saved feature map visualizations to:", cfg.outdir)
    print("Files:")
    print("  input_image.png")
    print("  featuremaps_conv1.png")
    print("  featuremaps_conv2.png")
    print("  featuremaps_conv3.png")


if __name__ == "__main__":
    main()
