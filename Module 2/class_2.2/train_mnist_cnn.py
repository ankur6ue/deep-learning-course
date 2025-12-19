# train_mnist_cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
EPOCHS = 5
MODEL_DIR = Path("../models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "mnist_cnn.pt"

class MnistCNN(nn.Module):
    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (B,32,14,14)
        x = self.pool(F.relu(self.conv2(x)))  # (B,64,7,7)
        x = x.view(x.size(0), -1)
        embedding = F.relu(self.fc1(x))       # (B, embedding_dim)
        logits = self.fc2(embedding)
        return logits, embedding

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # standard MNIST norm
    ])

    train_ds = datasets.MNIST(root="../data", train=True, download=False, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = MnistCNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits, _ = model(x)
            loss = criterion(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)

        print(f"Epoch {epoch+1}/{EPOCHS} - loss={total_loss / len(train_loader.dataset):.4f}")

    torch.save({
        "model_state": model.state_dict(),
        "embedding_dim": 64,
    }, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    main()
