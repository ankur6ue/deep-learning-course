import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 1. Generate training data
np.random.seed(0)
x_np = np.linspace(-2 * np.pi, 2 * np.pi, 200)
y_np = np.sin(x_np) + 0.1 * np.random.randn(*x_np.shape)  # noisy sin(x)

x_train = torch.tensor(x_np, dtype=torch.float32).unsqueeze(1)
y_train = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)


# 2. Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self, hidden_dim=10):
        super(SimpleNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


model = SimpleNN(hidden_dim=100)

# 3. Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. Training loop
epochs = 2000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# 5. Plot predictions
model.eval()
x_test = torch.linspace(-2 * np.pi, 2 * np.pi, 300).unsqueeze(1)
y_pred = model(x_test).detach().numpy()

plt.figure(figsize=(10, 5))
plt.scatter(x_np, y_np, label="Training data (noisy)", s=10, alpha=0.6)
plt.plot(x_test.numpy(), y_pred, color='red', label="NN Approximation")
plt.plot(x_np, np.sin(x_np), color='green', linestyle='--', label="True sin(x)")
plt.title("Neural Network Approximating sin(x)")
plt.legend()
plt.grid(True)
plt.show()