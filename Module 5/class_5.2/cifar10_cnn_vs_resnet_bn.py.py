# cifar10_cnn_vs_resnet_bn.py
# Compare: Plain deep CNN (no BN, no skips) vs ResNet-style (BN + skips) on CIFAR-10
# Metrics: loss/acc curves, gradient norms, update/weight ratio, throughput, param count.

import os
import time
import math
import random
from dataclasses import dataclass, asdict
import torch.multiprocessing as mp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as T


# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------
# Utilities
# ----------------------------
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


def grad_global_norm(model: nn.Module) -> float:
    # L2 norm over all grads
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().double().pow(2).sum().item()
    return math.sqrt(total)


@torch.no_grad()
def weight_global_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.requires_grad:
            total += p.detach().float().pow(2).sum().item()
    return math.sqrt(total)


@torch.no_grad()
def update_to_weight_ratio(model: nn.Module, lr: float) -> float:
    """
    Rough proxy for "how violent" updates are:
    || -lr * grad || / || weight ||
    Note: for AdamW this is only an approximation (since it uses moments),
    but still a useful stability indicator.
    """
    g2 = 0.0
    w2 = 0.0
    for p in model.parameters():
        if p.requires_grad:
            w2 += p.detach().float().pow(2).sum().item()
            if p.grad is not None:
                g2 += p.grad.detach().float().pow(2).sum().item()
    if w2 <= 0:
        return float("nan")
    return (lr * math.sqrt(g2)) / math.sqrt(w2)


def has_nan_or_inf(t: torch.Tensor) -> bool:
    return (torch.isnan(t).any() or torch.isinf(t).any()).item()


# ----------------------------
# Models
# ----------------------------
class PlainCNN(nn.Module):
    """
    Deeper CNN WITHOUT BatchNorm and WITHOUT skip connections.
    Uses Conv -> ReLU stacks and occasional stride-2 convs to downsample.
    This is intentionally "harder to optimize" at depth than a ResNet+BN.
    """

    def __init__(self, num_classes=10, width=64, depth=10):
        super().__init__()
        assert depth >= 6 and depth % 2 == 0, "Use an even depth >= 6 for this simple stack."

        layers = []
        in_ch = 3
        ch = width

        # Stem
        layers += [
            nn.Conv2d(in_ch, ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
        ]
        in_ch = ch

        # Stack blocks, downsample a couple of times by stride-2 conv
        # depth here counts conv layers after stem approximately; keep simple.
        num_convs = depth
        downsample_at = {num_convs // 3, 2 * num_convs // 3}  # two downsample points

        for i in range(1, num_convs + 1):
            stride = 2 if i in downsample_at else 1
            layers += [
                nn.Conv2d(in_ch, ch, kernel_size=3, stride=stride, padding=1, bias=True),
                nn.ReLU(inplace=True),
            ]
            in_ch = ch
            # optionally increase width after downsample
            if i in downsample_at:
                ch *= 2

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class BasicBlock(nn.Module):
    """
    ResNet basic block: (Conv-BN-ReLU) -> (Conv-BN) + skip -> ReLU
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = None
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        skip = x if self.shortcut is None else self.shortcut(x)
        out = F.relu(out + skip, inplace=True)
        return out


class SmallResNet(nn.Module):
    """
    CIFAR-style small ResNet with BN + skip connections.
    """
    def __init__(self, num_classes=10, width=64, blocks=(2, 2, 2)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.in_ch = width

        self.layer1 = self._make_layer(width,  blocks[0], stride=1)
        self.layer2 = self._make_layer(width*2, blocks[1], stride=2)
        self.layer3 = self._make_layer(width*4, blocks[2], stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(width*4, num_classes)

    def _make_layer(self, out_ch, num_blocks, stride):
        layers = [BasicBlock(self.in_ch, out_ch, stride=stride)]
        self.in_ch = out_ch
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# ----------------------------
# Training
# ----------------------------
@dataclass
class TrainConfig:
    seed: int = 0
    epochs: int = 20
    grad_clip: float = 5.0
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 5e-4
    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True  # mixed precision on GPU
    log_every: int = 100  # steps


def get_cifar10_loaders(batch_size: int, num_workers: int):
    # Standard CIFAR-10 augments
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)),
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)),
    ])

    train_ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tf)
    test_ds = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, persistent_workers=True, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, persistent_workers=True, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy_from_logits(logits, y) * bs
        n += bs
    return total_loss / n, total_acc / n


def train_one_model(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, cfg: TrainConfig, name: str):
    device = cfg.device
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.startswith("cuda")))

    history = {
        "train_loss": [], "train_acc": [],
        "test_loss": [], "test_acc": [],
        "grad_norm": [], "upd_w_ratio": [],
        "imgs_per_sec": [],
    }

    print(f"\n==== Training: {name} ====")
    print("config:", asdict(cfg))
    print("params:", count_params(model))
    print("device:", device)

    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        running_acc = 0.0
        running_grad = 0.0
        running_ratio = 0.0
        n = 0
        num_steps_in_epoch = 0
        for step, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(cfg.amp and device.startswith("cuda"))):
                logits = model(x)
                loss = F.cross_entropy(logits, y)

            scaler.scale(loss).backward()

            # unscale grads so norm is meaningful
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            # Detect if update skipped because some of the scaled gradients overflowed
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()

            step_was_skipped = (scale_after < scale_before)
            if step_was_skipped:
                print("step skipped")

            # detect non-finite grads
            found_bad = False
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    found_bad = True
                    break

            if found_bad:
                # This step will be skipped by scaler.step anyway.
                # Avoid polluting your metrics:
                continue

            scaler.step(optimizer)
            scaler.update()

            gnorm = grad_global_norm(model)
            ratio = update_to_weight_ratio(model, cfg.lr)

            bs = x.size(0)
            running_loss += loss.item()
            running_acc += accuracy_from_logits(logits, y)
            running_grad += gnorm
            running_ratio += ratio
            n += bs
            num_steps_in_epoch += 1
            global_step += 1
            if cfg.log_every and (global_step % cfg.log_every == 0):
                print(f"[{name}] epoch {epoch:02d} step {step:04d} "
                      f"loss {loss.item():.4f} acc {accuracy_from_logits(logits, y):.3f} "
                      f"grad_norm {gnorm:.2f} upd/w {ratio:.2e}")

        elapsed = time.time() - t0
        imgs_sec = n / max(elapsed, 1e-9)

        train_loss = running_loss / num_steps_in_epoch
        train_acc = running_acc / num_steps_in_epoch
        avg_grad = running_grad / num_steps_in_epoch
        avg_ratio = running_ratio / num_steps_in_epoch

        test_loss, test_acc = evaluate(model, test_loader, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["grad_norm"].append(avg_grad)
        history["upd_w_ratio"].append(avg_ratio)
        history["imgs_per_sec"].append(imgs_sec)

        print(f"[{name}] epoch {epoch:02d}/{cfg.epochs} "
              f"train_loss {train_loss:.4f} train_acc {train_acc:.3f} | "
              f"test_loss {test_loss:.4f} test_acc {test_acc:.3f} | "
              f"grad_norm {avg_grad:.2f} upd/w {avg_ratio:.2e} imgs/s {imgs_sec:.0f}")

    return model, history


# ----------------------------
# Plotting (matplotlib)
# ----------------------------
def plot_histories(hist_a, name_a, hist_b, name_b, out_path="cifar10_bn_residual_comparison.png"):
    import matplotlib.pyplot as plt

    epochs = np.arange(1, len(hist_a["train_loss"]) + 1)

    def plot_two(ykey, ylabel, title, filename_suffix):
        plt.figure()
        plt.plot(epochs, hist_a[ykey], label=name_a)
        plt.plot(epochs, hist_b[ykey], label=name_b)
        plt.xlabel("epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()

    plot_two("train_loss", "loss", "Train loss", "train_loss")
    plot_two("test_loss", "loss", "Test loss", "test_loss")
    plot_two("train_acc", "accuracy", "Train accuracy", "train_acc")
    plot_two("test_acc", "accuracy", "Test accuracy", "test_acc")
    plot_two("grad_norm", "L2 norm", "Average grad norm per epoch", "grad_norm")
    plot_two("upd_w_ratio", "ratio", "Update-to-weight ratio (proxy stability)", "upd_w")

    # Save a combined figure with subplots? (User asked no constraints; keep simple: save last one)
    # Instead, save each plot as separate files + one summary print.
    # We'll write them all.
    plt.figure(); plt.axis("off")
    plt.text(0.01, 0.9,
             "Saved separate plots:\n"
             "  train_loss, test_loss, train_acc, test_acc, grad_norm, upd_w_ratio\n"
             "Check current directory.\n", fontsize=12)
    plt.tight_layout()

    # Actually save each figure by iterating existing figures
    figs = [plt.figure(i) for i in plt.get_fignums()]
    base = os.path.splitext(out_path)[0]
    for idx, fig in enumerate(figs[:-1], start=1):
        fig.savefig(f"{base}_{idx}.png", dpi=160)

    print(f"\nSaved plots to: {base}_*.png")


# ----------------------------
# Main
# ----------------------------
def main():
    cfg = TrainConfig(
        seed=0,
        epochs=20,
        grad_clip=5,
        batch_size=128,
        lr=1e-3,
        weight_decay=5e-4,
        num_workers=2,
        amp=True,
        log_every=200,
    )
    set_seed(cfg.seed)

    train_loader, test_loader = get_cifar10_loaders(cfg.batch_size, cfg.num_workers)

    # Keep widths similar so the comparison is fair-ish (not perfect param match, but close enough for demo)
    plain = PlainCNN(num_classes=10, width=64, depth=12)          # deeper, no BN, no skip
    resnet = SmallResNet(num_classes=10, width=64, blocks=(2, 2, 2))  # ResNet-ish, BN + skips

    # Train both
    _, hist_plain = train_one_model(plain, train_loader, test_loader, cfg, name="PlainCNN (no BN, no skips)")
    _, hist_resnet = train_one_model(resnet, train_loader, test_loader, cfg, name="ResNet (BN + skips)")

    # Print a concise comparison summary
    def best_acc(hist):
        return max(hist["test_acc"])
    def last_acc(hist):
        return hist["test_acc"][-1]

    print("\n==== Summary ====")
    print(f"PlainCNN: params={count_params(plain)} best_test_acc={best_acc(hist_plain):.3f} last_test_acc={last_acc(hist_plain):.3f}")
    print(f"ResNet  : params={count_params(resnet)} best_test_acc={best_acc(hist_resnet):.3f} last_test_acc={last_acc(hist_resnet):.3f}")

    # Plot
    plot_histories(hist_plain, "PlainCNN", hist_resnet, "ResNetBN+Skip",
                   out_path="cifar10_plain_vs_resnet.png")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()


