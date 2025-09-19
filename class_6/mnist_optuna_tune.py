import math, os, random, time
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, random_split, Subset

from torchvision import datasets, transforms
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# -----------------------------
# Repro & device
# -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
)
print("Device:", DEVICE)

# -----------------------------
# Data
# -----------------------------
def get_loaders(batch_size: int, train_subset_size: int = 10000) -> Tuple[DataLoader, DataLoader]:
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    root = os.environ.get("MNIST_DATA_ROOT", "./data")
    full_train = datasets.MNIST(root=root, train=True, download=True, transform=tfm)
    test = datasets.MNIST(root=root, train=False, download=True, transform=tfm)

    # Small, fixed subset for quick trials (approx. 10k train / 2k val by default)
    if train_subset_size is not None and train_subset_size < len(full_train):
        full_train = Subset(full_train, list(range(train_subset_size)))

    # 80/20 split for validation
    val_len = max(2000, int(0.2 * len(full_train)))
    train_len = len(full_train) - val_len
    train_ds, val_ds = random_split(full_train, [train_len, val_len], generator=torch.Generator().manual_seed(0))

    pin = DEVICE == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin)
    return train_loader, val_loader

# -----------------------------
# Model (MLP)
# -----------------------------
class MLP(nn.Module):
    def __init__(self, hidden1: int, hidden2: int, p: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(hidden2, 10),
        )
    def forward(self, x): return self.net(x)

# -----------------------------
# Training / Eval
# -----------------------------
def train_one_epoch(model, loader, optimizer, scheduler, scaler, criterion):
    model.train()
    running_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(DEVICE=="cuda")):
            logits = model(xb)
            loss = criterion(logits, yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        running_loss += loss.item() * xb.size(0)
    return running_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss_sum += loss.item() * xb.size(0)
        preds = logits.argmax(1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
    return loss_sum / total, correct / total

# -----------------------------
# LR schedule: linear warmup then cosine decay
# -----------------------------
def build_warmup_cosine(optimizer, total_steps: int, warmup_ratio: float = 0.1):
    warmup_steps = max(1, int(warmup_ratio * total_steps))
    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda=lr_lambda)

# -----------------------------
# Optuna objective
# -----------------------------
def objective(trial: optuna.trial.Trial):
    # --- Search space ---
    lr = trial.suggest_float("lr", 1e-4, 5e-2, log=True)
    hidden1 = trial.suggest_int("hidden1", 64, 1024, log=True)
    hidden2 = trial.suggest_int("hidden2", 32, max(128, hidden1//2), log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    epochs = 4  # keep trials quick; increase later if needed
    train_subset = 10000  # use a small subset per trial to speed up BO

    # --- Data ---
    train_loader, val_loader = get_loaders(batch_size=batch_size, train_subset_size=train_subset)

    # --- Model/opt ---
    model = MLP(hidden1, hidden2, dropout).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE=="cuda"))

    # Steps for per-iteration scheduler (one step per optimizer step)
    total_steps = epochs * len(train_loader)
    scheduler = build_warmup_cosine(optimizer, total_steps, warmup_ratio=0.1)

    # --- Train + prune-aware eval ---
    best_val_acc = 0.0
    global_step = 0
    for epoch in range(1, epochs + 1):
        train_one_epoch(model, train_loader, optimizer, scheduler, scaler, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        # Report intermediate value for pruning
        trial.report(1.0 - val_acc, epoch)  # minimize error (1-acc)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        best_val_acc = max(best_val_acc, val_acc)

    # Return validation error (so study.direction="minimize" makes sense)
    return 1.0 - best_val_acc

# -----------------------------
# Run study
# -----------------------------
def main():
    sampler = TPESampler(seed=42, n_startup_trials=10, multivariate=True, group=True)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=25, timeout=None, gc_after_trial=True)

    print("\nBest trial:")
    best = study.best_trial
    print("  value (val error):", best.value)
    print("  params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    # Optional: retrain best on a larger subset or full train set, then test
    print("\nRetraining best configuration on larger subset for a fairer score...")
    p = best.params
    # Larger subset & more epochs for the final fit (edit to full dataset if you want)
    final_epochs = 6
    batch_size = p["batch_size"]
    train_loader, val_loader = get_loaders(batch_size=batch_size, train_subset_size=30000)

    model = MLP(int(p["hidden1"]), int(p["hidden2"]), float(p["dropout"])).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=float(p["lr"]), weight_decay=float(p["weight_decay"]))
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE=="cuda"))
    scheduler = build_warmup_cosine(optimizer, final_epochs*len(train_loader), warmup_ratio=0.1)

    for epoch in range(1, final_epochs+1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} | val_acc={val_acc*100:.2f}%")

    # Save model if you like
    torch.save(model.state_dict(), "mnist_mlp_best_optuna.pt")
    print("\nSaved model -> mnist_mlp_best_optuna.pt")

if __name__ == "__main__":
    main()
