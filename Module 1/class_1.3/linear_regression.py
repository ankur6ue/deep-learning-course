
# Copyright 2025 Ankur Mohan
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import numpy as np
import matplotlib.pyplot as plt

# Generate points along a line or quadratic curve, add Gaussian noise,
# then fit the best *line* using closed-form linear regression.
# Plots: noisy points, true curve, and fitted line.


def main(curve_type: str) -> None:
    def gen_y_linear(x, m, b):
        return m * x + b

    def gen_y_quadratic(x, a, b, c):
        return a * x * x + b * x + c

    n = 100
    x = np.linspace(0.0, 10.0, n)

    # Ground-truth function (we still fit a line either way)
    if curve_type == "quadratic":
        y_true = gen_y_quadratic(x, 1.0, 2.0, 3.0)
        title_prefix = "Quadratic ground truth"
    else:
        m, b = 2.0, 5.0
        y_true = gen_y_linear(x, m, b)
        title_prefix = "Linear ground truth"

    # Add noise
    noise = np.random.normal(0.0, 1.5, size=n)
    y_noisy = y_true + noise

    # Closed-form fit via sufficient statistics: y ≈ m_est * x + b_est
    sx = np.sum(x)
    sy = np.sum(y_noisy)
    sxx = np.dot(x, x)
    sxy = np.dot(x, y_noisy)
    denom = sxx - sx * sx / n
    if np.isclose(denom, 0.0):
        m_est, b_est = 0.0, float(np.mean(y_noisy))  # degenerate case
    else:
        m_est = (sxy - sx * sy / n) / denom
        b_est = (sy - m_est * sx) / n

    # This is discussed in lecture 1.4
    # Same result via normal equations (explicit inverse on purpose)
    X = np.c_[x, np.ones(n)]               # shape (n, 2)
    Y = y_noisy.reshape(-1, 1)             # shape (n, 1)
    A_est = np.linalg.inv(X.T @ X) @ (X.T @ Y)  # [m; b] using (X^T X)^{-1} X^T y
    # Uncomment to verify numerically:
    # print("[matrix] m =", A_est[0, 0], " b =", A_est[1, 0])

    y_fit = m_est * x + b_est

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(x, y_true, label="True f(x)")
    plt.plot(x, y_fit, label=f"Fitted line: y={m_est:.3f}x+{b_est:.3f}")
    plt.scatter(x, y_noisy, s=12, label="Noisy points")
    plt.title(f"{title_prefix} (line fit shown)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate points along a line or a quadratic curve, add Gaussian noise, "
            "and fit a *line* using closed-form linear regression."
        )
    )
    parser.add_argument("--curve_type", choices=["linear", "quadratic"], default="linear")
    args = parser.parse_args()
    main(args.curve_type)
