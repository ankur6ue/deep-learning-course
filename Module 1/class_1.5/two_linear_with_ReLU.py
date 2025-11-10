# Copyright 2025 Ankur Mohan
# Pure Python / Numpy demo: two linear (matrix) layers WITH ReLU
# Denominator layout: batch is the second dimension.
# Demonstrates piecewise-linear approximation of sin(x).

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
rng = np.random.default_rng(123)

# ---- Dataset ----
B = 256
x = rng.uniform(-2*np.pi, 2*np.pi, size=(1, B))
y = np.sin(x)

# ---- Model parameters ----
H = 8
W1 = rng.normal(scale=0.3, size=(H, 1))
b1 = np.zeros((H, 1))
W2 = rng.normal(scale=0.3, size=(1, H))
b2 = np.zeros((1, 1))

def relu(u): return np.maximum(u, 0)
def relu_grad(u): return (u > 0).astype(float)

def forward(x):
    h = W1 @ x + b1           # (H,B)
    a = relu(h)               # (H,B)
    y_hat = W2 @ a + b2       # (1,B)
    return y_hat, h, a

def mse(y_hat, y):
    return np.mean((y_hat - y)**2)

lr = 5e-3
epochs = 5000

for t in range(epochs):
    y_hat, h, a = forward(x)
    loss = mse(y_hat, y)

    # ---- Gradients ----
    dL_dyhat = (2.0 / B) * (y_hat - y)         # (1,B)
    dL_dW2 = dL_dyhat @ a.T                    # (1,H)
    dL_db2 = np.sum(dL_dyhat, axis=1, keepdims=True)
    dL_da = W2.T @ dL_dyhat                    # (H,B)
    dL_dh = dL_da * relu_grad(h) # element wise product
    dL_dW1 = dL_dh @ x.T                       # (H,1)
    dL_db1 = np.sum(dL_dh, axis=1, keepdims=True)

    # ---- Updates ----
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
plt.plot(xs.flatten(), ys_pred.flatten(), "--", label="two layers + ReLU")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Two Linear Layers + ReLU (Denominator notation)")
plt.legend()
plt.tight_layout()
plt.show()
