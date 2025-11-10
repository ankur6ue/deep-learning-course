# Copyright 2025 Ankur Mohan
# Pure Python / Numpy demo: two linear (matrix) layers WITH tanh non-linearity
# Denominator layout: batch dimension is the second dimension (x ∈ R^{n×B})
#
# Forward equations:
#   h = W1 x + b1          → (H,B)
#   a = tanh(h)            → (H,B)
#   ŷ = W2 a + b2          → (1,B)
#
# Backward equations (denominator notation):
#   dL/dŷ = (2/B) * (ŷ - y)
#   dL/dW2 = (dL/dŷ) aᵀ
#   dL/db2 = Σ_columns(dL/dŷ)
#   dL/da  = W2ᵀ (dL/dŷ)
#   dL/dh  = dL/da ⊙ (1 - tanh²(h))
#   dL/dW1 = (dL/dh) xᵀ
#   dL/db1 = Σ_columns(dL/dh)

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(123)

# ---- Dataset ----
B = 256  # batch size
x = rng.uniform(-2*np.pi, 2*np.pi, size=(1, B))  # (1,B)
y = np.sin(x)                                   # (1,B)

# ---- Model parameters ----
H = 8  # hidden width
W1 = rng.normal(scale=0.3, size=(H, 1))
b1 = np.zeros((H, 1))
W2 = rng.normal(scale=0.3, size=(1, H))
b2 = np.zeros((1, 1))

def tanh(u):
    return np.tanh(u)

def tanh_grad(u):
    t = np.tanh(u)
    return 1.0 - t**2

def forward(x):
    h = W1 @ x + b1         # (H,B)
    a = tanh(h)             # (H,B)
    y_hat = W2 @ a + b2     # (1,B)
    return y_hat, h, a

def mse(y_hat, y):
    return np.mean((y_hat - y)**2)

# ---- Training ----
lr = 5e-3
epochs = 5000

for t in range(epochs):
    y_hat, h, a = forward(x)
    loss = mse(y_hat, y)

    # Gradients
    dL_dyhat = (2.0 / B) * (y_hat - y)        # (1,B)
    dL_dW2 = dL_dyhat @ a.T                   # (1,H)
    dL_db2 = np.sum(dL_dyhat, axis=1, keepdims=True)  # (1,1)

    dL_da = W2.T @ dL_dyhat                   # (H,B)
    dL_dh = dL_da * tanh_grad(h)              # (H,B)

    dL_dW1 = dL_dh @ x.T                      # (H,1)
    dL_db1 = np.sum(dL_dh, axis=1, keepdims=True)     # (H,1)

    # Parameter updates
    W2 -= lr * dL_dW2
    b2 -= lr * dL_db2
    W1 -= lr * dL_dW1
    b1 -= lr * dL_db1

    if (t+1) % 500 == 0:
        print(f"epoch {t+1:4d}  loss={loss:.6f}")

# ---- Visualization ----
xs = np.linspace(-2*np.pi, 2*np.pi, 400).reshape(1, -1)
ys_true = np.sin(xs)
ys_pred, _, _ = forward(xs)

plt.figure(figsize=(7,4))
plt.plot(xs.flatten(), ys_true.flatten(), label="sin(x)")
plt.scatter(x.flatten(), y.flatten(), s=10, alpha=0.4, label="training points")
plt.plot(xs.flatten(), ys_pred.flatten(), "--", label="two layers + tanh")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Two Linear Layers + tanh (Denominator notation)")
plt.legend()
plt.tight_layout()
plt.show()
