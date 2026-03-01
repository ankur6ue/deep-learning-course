import numpy as np
import torch
import torch.nn.functional as F

# ============================================================
# BatchNorm for MLP: X shape (B, D), normalize per feature (dim=0)
# Verify:
#  (1) Finite differences vs analytic for ONE chosen feature d0
#  (2) Analytic gradients vs PyTorch autograd for ALL features
# ============================================================

def bn_forward_mlp(X, gamma, beta, eps=1e-5):
    """
    X: (B, D)
    gamma, beta: (D,) (per-feature)
    BN stats computed over batch dimension for each feature d.

    Returns:
      Y: (B, D)
      cache: needed for backward
    """
    B, D = X.shape
    mu = X.mean(axis=0)                          # (D,)
    Xc = X - mu                                  # (B, D)
    var = (Xc * Xc).mean(axis=0)                 # (D,) population variance
    std = np.sqrt(var + eps)                     # (D,)
    inv = 1.0 / std                              # (D,)
    Xhat = Xc * inv                              # (B, D) broadcast inv over rows
    Y = Xhat * gamma + beta                      # (B, D)
    cache = (Xhat, inv, gamma, eps)              # enough for simplified backward
    return Y, cache

def bn_backward_mlp(dY, cache):
    """
    dY: (B, D) upstream gradient
    Returns:
      dX: (B, D)
      dgamma: (D,)
      dbeta: (D,)
    """
    Xhat, inv, gamma, eps = cache
    B, D = dY.shape

    dbeta = dY.sum(axis=0)                       # (D,)
    dgamma = (dY * Xhat).sum(axis=0)             # (D,)

    dXhat = dY * gamma                           # (B, D), broadcast gamma (D, ) over Batch

    S1 = dXhat.sum(axis=0)                       # (D,)
    S2 = (dXhat * Xhat).sum(axis=0)              # (D,)

    # Simplified BN backward (per feature independently), vectorized over D
    dX = (1.0 / B) * inv * (B * dXhat - S1 - Xhat * S2)  # broadcasts (D,) over rows
    return dX, dgamma, dbeta

# -----------------------------
# Finite differences (1 feature)
# -----------------------------
def fd_grad_feature_column(loss_fn, X, d0, eps=1e-6):
    """
    Central-difference gradient of loss wrt X[:, d0] only.
    """
    B, D = X.shape
    g = np.zeros(B, dtype=np.float64)
    for i in range(B):
        Xp = X.copy(); Xm = X.copy()
        Xp[i, d0] += eps
        Xm[i, d0] -= eps
        g[i] = (loss_fn(Xp) - loss_fn(Xm)) / (2 * eps)
    return g

def fd_grad_scalar(loss_fn_scalar, a, eps=1e-6):
    return (loss_fn_scalar(a + eps) - loss_fn_scalar(a - eps)) / (2 * eps)

# -----------------------------
# Main test
# -----------------------------
if __name__ == "__main__":
    # Problem size
    B = 16
    D = 8
    eps = 1e-5
    rng = np.random.default_rng(0)

    # Data
    X = rng.normal(size=(B, D)).astype(np.float64)
    gamma = rng.normal(size=(D,)).astype(np.float64) # Note: I'm using the symbol lambda in the lecture notes
    beta  = rng.normal(size=(D,)).astype(np.float64)

    # Make a non-trivial scalar loss: L = sum_{i,d} W[i,d] * Y[i,d]
    W = rng.normal(size=(B, D)).astype(np.float64)

    def loss_numpy(X_in, gamma_in, beta_in):
        Y, _ = bn_forward_mlp(X_in, gamma_in, beta_in, eps=eps)
        return float(np.sum(Y * W))

    # ---- Analytic grads (NumPy)
    Y, cache = bn_forward_mlp(X, gamma, beta, eps=eps)
    dY = W.copy()  # since L = sum Y*W, dL/dY = W
    dX_analytic, dgamma_analytic, dbeta_analytic = bn_backward_mlp(dY, cache)

    # ============================================================
    # (1) Finite-difference check for ONE chosen feature d0
    # ============================================================
    d0 = 3  # pick any feature index

    dx_fd_col = fd_grad_feature_column(
        loss_fn=lambda Xtest: loss_numpy(Xtest, gamma, beta),
        X=X,
        d0=d0,
        eps=1e-6
    )

    # Also check gamma_d0 and beta_d0 via FD (optional but useful)
    dgamma_fd_d0 = fd_grad_scalar(
        loss_fn_scalar=lambda g: loss_numpy(X, np.where(np.arange(D)==d0, g, gamma), beta),
        a=gamma[d0],
        eps=1e-6
    ) # np.where(np.arange(D)==d0, g, gamma) replaces the value at the d0 coordinate in the gamma vector with the
    # provided value
    dbeta_fd_d0 = fd_grad_scalar(
        loss_fn_scalar=lambda b: loss_numpy(X, gamma, np.where(np.arange(D)==d0, b, beta)),
        a=beta[d0],
        eps=1e-6
    )

    print("Finite diff vs analytic (one feature column d0 = {})".format(d0))
    print("max|dX[:,d0] - FD| =", np.max(np.abs(dX_analytic[:, d0] - dx_fd_col)))
    print("|dgamma[d0] - FD|  =", abs(dgamma_analytic[d0] - dgamma_fd_d0))
    print("|dbeta[d0]  - FD|  =", abs(dbeta_analytic[d0]  - dbeta_fd_d0))

    # ============================================================
    # (2) Compare analytic grads vs PyTorch autograd for FULL matrix
    # ============================================================
    torch.set_default_dtype(torch.float64)

    Xt = torch.tensor(X, requires_grad=True)           # (B, D)
    gt = torch.tensor(gamma, requires_grad=True)       # (D,)
    bt = torch.tensor(beta, requires_grad=True)        # (D,)
    Wt = torch.tensor(W)                               # (B, D)

    # F.batch_norm expects (N, C) or (N, C, ...) with C=features.
    running_mean = torch.zeros(D, dtype=torch.float64)
    running_var  = torch.ones(D, dtype=torch.float64)

    Yt = F.batch_norm(
        Xt, running_mean, running_var,
        weight=gt, bias=bt,
        training=True, momentum=0.0, eps=eps
    )  # (B, D)

    Lt = (Yt * Wt).sum()
    Lt.backward()

    dX_torch = Xt.grad.detach().cpu().numpy()
    dgamma_torch = gt.grad.detach().cpu().numpy()
    dbeta_torch = bt.grad.detach().cpu().numpy()

    print("\nAnalytic vs PyTorch autograd (full matrix)")
    print("max|dX - dX_torch|         =", np.max(np.abs(dX_analytic - dX_torch)))
    print("max|dgamma - dgamma_torch| =", np.max(np.abs(dgamma_analytic - dgamma_torch)))
    print("max|dbeta - dbeta_torch|   =", np.max(np.abs(dbeta_analytic - dbeta_torch)))

