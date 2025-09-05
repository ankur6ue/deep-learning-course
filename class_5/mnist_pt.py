# mnist_mlp_warmup_cosine.py
import math, random, os
import torch
import torch.nn as nn
import sys
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# -----------------------
# Repro & device
# -----------------------
def set_seed(s=42):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

set_seed(42)
device = (
    "cuda" if torch.cuda.is_available()
     else "cpu"
)
print(f"Using device: {device}")

# -----------------------
# Data
# -----------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),  # standard MNIST stats
])

MNIST_DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
full_train = datasets.MNIST(root=MNIST_DATA_ROOT, train=True, download=False, transform=transform)
test_ds     = datasets.MNIST(root=MNIST_DATA_ROOT, train=False, download=False, transform=transform)

# 55k train / 5k val split
train_len = 55000
val_len   = len(full_train) - train_len
train_ds, val_ds = random_split(full_train, [train_len, val_len], generator=torch.Generator().manual_seed(42))

batch_size = 128
pin = device == "cuda"
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=pin)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin)

# -----------------------
# Model (MLP)
# -----------------------
class MLP(nn.Module):
    def __init__(self, in_features=28*28, hidden1=512, hidden2=256, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, hidden1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x):
        return self.net(x)

model = MLP().to(device)

# -----------------------
# Optimizer, Scheduler (warmup + cosine)
# -----------------------
base_lr = 1e-3
optimizer = AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

epochs = 5
num_training_steps = epochs * len(train_loader)
num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup

def lr_lambda(current_step: int):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

use_amp = device == "cuda"
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

def accuracy(logits, y):
    return (logits.argmax(dim=1) == y).float().mean().item()

# -----------------------
# Train / Validate
# -----------------------
best_val_acc = 0.0
global_step = 0
for epoch in range(1, epochs + 1):
    model.train()
    running_loss, running_acc = 0.0, 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(xb)
            loss = criterion(logits, yb)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        global_step += 1

        running_loss += loss.item() * xb.size(0)
        running_acc  += (logits.argmax(dim=1) == yb).float().sum().item()

    train_loss = running_loss / len(train_loader.dataset)
    train_acc  = running_acc  / len(train_loader.dataset)

    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            logits = model(xb)
            val_correct += (logits.argmax(dim=1) == yb).sum().item()
            val_total   += yb.size(0)
    val_acc = val_correct / val_total

    current_lr = optimizer.param_groups[0]["lr"]
    print(f"Epoch {epoch:02d}/{epochs} | lr={current_lr:.6f} | train_loss={train_loss:.4f} | "
          f"train_acc={train_acc*100:.2f}% | val_acc={val_acc*100:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "mnist_mlp_best.pt")
        print(f"  Saved new best model (val_acc={best_val_acc*100:.2f}%).")

# -----------------------
# Test evaluation (with best model)
# -----------------------
state = torch.load("mnist_mlp_best.pt", map_location=device)
model.load_state_dict(state)
model.eval()

test_correct = 0
test_total = 0
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        logits = model(xb)
        test_correct += (logits.argmax(dim=1) == yb).sum().item()
        test_total   += yb.size(0)

print(f"Test accuracy: {100.0 * test_correct / test_total:.2f}%")
