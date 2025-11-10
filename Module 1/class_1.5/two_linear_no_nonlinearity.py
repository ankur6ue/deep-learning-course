# Copyright 2025 Ankur Mohan
# Pure Python / Numpy demo: two linear (matrix) layers WITHOUT a non-linearity
# Denominator layout: batch is the second dimension.
# Shows that two linear layers still represent an affine map y = A x + b.

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ---- Dataset: (x, sin(x)) ----
B = 256 # Number of points in our training dataset
x = rng.uniform(-2*np.pi, 2*np.pi, size=(1, B))  # (1,B)
y = np.sin(x)                                    # (1,B)

# ---- Model parameters ----
H = 8
# First layer takes a 1*B (B = batch size) vector, and produces a H*B output. H is the hidden layer dimension
W1 = rng.normal(scale=0.2, size=(H, 1))
b1 = np.zeros((H, 1))
# Second layer takes a H*B (B = batch size) vector, and produces a !*B output.
W2 = rng.normal(scale=0.2, size=(1, H))
b2 = np.zeros((1, 1))

def forward(x):
    h = W1 @ x + b1           # (H,B)
    y_hat = W2 @ h + b2       # (1,B)
    return y_hat, h

def mse(y_hat, y):
    return np.mean((y_hat - y)**2)

lr = 1e-2
epochs = 3000

for t in range(epochs):
    y_hat, h = forward(x)
    loss = mse(y_hat, y)

    # ---- Gradients (denominator notation) ----
    dL_dyhat = (2.0 / B) * (y_hat - y)       # (1,B)
    dL_dW2 = dL_dyhat @ h.T                  # (1,H)
    dL_db2 = np.sum(dL_dyhat, axis=1, keepdims=True)  # (1,1)
    dL_dh = W2.T @ dL_dyhat                  # (H,B)
    dL_dW1 = dL_dh @ x.T                     # (H,1)
    dL_db1 = np.sum(dL_dh, axis=1, keepdims=True)      # (H,1)

    # ---- Parameter updates ----
    W2 -= lr * dL_dW2
    b2 -= lr * dL_db2
    W1 -= lr * dL_dW1
    b1 -= lr * dL_db1

    if (t+1) % 500 == 0:
        print(f"epoch {t+1:4d}  loss={loss:.6f}")

# ---- Equivalent single affine map ----
A = W2 @ W1      # (1,1)
C = W2 @ b1 + b2 # (1,1)
print("\nEquivalent affine map: y = A x + C")
print(f"A = {A.item():.6f}, C = {C.item():.6f}")

# ---- Visualization ----
xs = np.linspace(-2*np.pi, 2*np.pi, 400).reshape(1, -1)
ys_true = np.sin(xs)
ys_pred, _ = forward(xs)
ys_line = A @ xs + C

plt.figure(figsize=(7,4))
plt.plot(xs.flatten(), ys_true.flatten(), label="sin(x)")
plt.scatter(x.flatten(), y.flatten(), s=10, alpha=0.4, label="training points")
plt.plot(xs.flatten(), ys_pred.flatten(), "--", label="two linear layers")
plt.plot(xs.flatten(), ys_line.flatten(), ":", label="affine fit")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Two Linear Layers (Denominator notation)")
plt.legend()
plt.tight_layout()
plt.show()
