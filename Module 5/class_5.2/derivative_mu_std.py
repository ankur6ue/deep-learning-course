import numpy as np

# -----------------------------
# Setup
# -----------------------------
B = 10
rng = np.random.default_rng(0)
x = rng.random(B)  # shape (B,)

def mu_fn(x):
    return np.mean(x)

def std_fn(x):
    # population std (ddof=0) to match BatchNorm-style batch variance convention
    return np.std(x, ddof=0)


mu = mu_fn(x)
s = std_fn(x)          # s = std

print("x     =", x)
print("mu    =", mu)
print("std s =", s)

# -----------------------------
# Analytic derivatives
# -----------------------------
dmu_dx = np.full(B, 1.0 / B)

if s > 0:
    # ds/dx_i = (x_i - mu) / (B * s)
    ds_dx = (x - mu) / (B * s)

else:
    ds_dx = np.full(B, np.nan)

print("\nAnalytic dmu/dx       =", dmu_dx)
print("Analytic dstd/dx (ds) =", ds_dx)

# -----------------------------
# Finite differences
# -----------------------------
eps = 1e-6
dmu_fd = np.zeros(B)
ds_fd = np.zeros(B)
dsigma_fd = np.zeros(B)

for i in range(B):
    x_pos = x.copy()
    x_neg = x.copy()
    x_pos[i] += eps
    x_neg[i] -= eps

    dmu_fd[i] = (mu_fn(x_pos) - mu_fn(x_neg)) / (2 * eps)
    ds_fd[i] = (std_fn(x_pos) - std_fn(x_neg)) / (2 * eps)

print("\nFD dmu/dx             =", dmu_fd)
print("FD dstd/dx (ds)       =", ds_fd)
print("FD dsigma/dx          =", dsigma_fd)

# -----------------------------
# Compare
# -----------------------------
print("\nMax abs error dmu     =", np.max(np.abs(dmu_fd - dmu_dx)))
print("Max abs error dstd    =", np.max(np.abs(ds_fd - ds_dx)))
