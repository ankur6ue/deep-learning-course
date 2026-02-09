"""
conv2d_single_channel.py
© 2025 Ankur Mohan

Single-channel (1 input channel, 1 output channel) 2D convolution
(cross-correlation) forward/backward in NumPy, with gradient checks.
"""

import numpy as np


def conv2d_forward_single(x, k, b):
    """
    x: (H, W)
    k: (kH, kW)
    b: scalar

    returns:
      y: (H_out, W_out)
      cache: values for backward
    """
    H, W = x.shape
    kH, kW = k.shape

    H_out = H - kH + 1
    W_out = W - kW + 1
    y = np.zeros((H_out, W_out), dtype=x.dtype)

    for i in range(H_out):
        for j in range(W_out):
            patch = x[i:i+kH, j:j+kW]          # (kH, kW)
            y[i, j] = np.sum(patch * k) + b    # cross-correlation

    cache = (x, k, b)
    return y, cache


def conv2d_backward_single(dy, cache):
    """
    dy: upstream gradient, shape (H_out, W_out)
    cache: (x, k, b)

    returns:
      dx: (H, W)
      dk: (kH, kW)
      db: scalar
    """
    x, k, b = cache
    H, W = x.shape
    kH, kW = k.shape
    H_out, W_out = dy.shape

    dx = np.zeros_like(x)
    dk = np.zeros_like(k)
    db = np.sum(dy)

    # dk: correlate dy with input patches
    for u in range(kH):
        for v in range(kW):
            s = 0.0
            for i in range(H_out):
                for j in range(W_out):
                    s += dy[i, j] * x[i + u, j + v]
            dk[u, v] = s

    # This follows the math in the lecture slides
    # This asks the question which Y_ij are influenced by X_ij?
    dxp = np.zeros_like(x)
    kHp = kH - 1
    kWp = kW - 1
    for i in range(H):
        for j in range(W):
            s = 0.0
            for p in range(max(0, i - kHp), min(i + 1, H_out)):
                for q in range(max(0, j - kWp), min(j + 1, W_out)):
                    s += dy[p, q] * k[i - p, j - q]
            dxp[i, j] = s

    # And is equivalent to this more efficient implementation
    # This asks the question which X_ij contributed to a Y_ij? It loops over Y_ij and accumulates the weighted influence
    # of corresponding G_ij over the X that contributed to that Y_ij. It results in more concise implementation as
    # we don't need to deal with boundary conditions
    # dx: scatter-add dy weighted by kernel into input regions
    for i in range(H_out):
        for j in range(W_out):
            g = dy[i, j]
            dx[i:i+kH, j:j+kW] += g * k

    return dx, dk, db


def loss_fn(y):
    """
    Scalar loss:
      L = 0.5 * sum(y^2)
    So dL/dy = y
    """
    return 0.5 * np.sum(y ** 2)


def finite_diff_param(param, compute_loss, eps=1e-5, num_checks=25, rng=None):
    """
    Centered finite differences on random coordinates of 'param'.
    Returns a sparse array of same shape with estimates at sampled coords.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    grad_est = np.zeros_like(param)

    for _ in range(num_checks):
        idx = tuple(rng.integers(0, s) for s in param.shape)
        old = param[idx]

        param[idx] = old + eps
        lp = compute_loss()

        param[idx] = old - eps
        lm = compute_loss()

        param[idx] = old
        grad_est[idx] = (lp - lm) / (2 * eps)

    return grad_est


def finite_diff_scalar(b_container, compute_loss, eps=1e-5):
    """
    Finite difference for scalar b stored in a dict so we can mutate it.
    """
    old = b_container["b"]

    b_container["b"] = old + eps
    lp = compute_loss()

    b_container["b"] = old - eps
    lm = compute_loss()

    b_container["b"] = old
    return (lp - lm) / (2 * eps)


def rel_error(a, b, tol=1e-12):
    return np.abs(a - b) / np.maximum(tol, np.abs(a) + np.abs(b))


def main():
    rng = np.random.default_rng(42)

    # Small, readable sizes
    H, W = 6, 6
    kH, kW = 3, 3

    x = rng.normal(size=(H, W))
    k = rng.normal(size=(kH, kW))
    b_box = {"b": float(rng.normal())}  # mutable scalar container

    # Forward
    y, cache = conv2d_forward_single(x, k, b_box["b"])
    L = loss_fn(y)

    # Backward (since L = 0.5 sum(y^2), dy = y)
    dy = y.copy()
    dx, dk, db = conv2d_backward_single(dy, cache)

    print("Loss:", L)
    print("||dx||:", np.linalg.norm(dx))
    print("||dk||:", np.linalg.norm(dk))
    print("db:", db)

    # --- Finite difference checks ---
    eps = 1e-5
    num_checks = 25

    def compute_loss_current():
        yy, _ = conv2d_forward_single(x, k, b_box["b"])
        return loss_fn(yy)

    # Check kernel gradient
    dk_fd = finite_diff_param(k, compute_loss_current, eps=eps, num_checks=num_checks, rng=rng)

    # Check input gradient
    dx_fd = finite_diff_param(x, compute_loss_current, eps=eps, num_checks=num_checks, rng=rng)

    # Check bias gradient
    db_fd = finite_diff_scalar(b_box, compute_loss_current, eps=eps)

    # Compare only sampled entries (nonzero in fd arrays)
    k_mask = (dk_fd != 0)
    x_mask = (dx_fd != 0)

    print("\nFinite difference check (max relative error on sampled entries):")
    print("dk:", np.max(rel_error(dk[k_mask], dk_fd[k_mask])))
    print("dx:", np.max(rel_error(dx[x_mask], dx_fd[x_mask])))
    print("db:", rel_error(db, db_fd))


if __name__ == "__main__":
    main()
