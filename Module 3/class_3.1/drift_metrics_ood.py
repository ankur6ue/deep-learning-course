#!/usr/bin/env python3
"""
OOD severity demo: PSI, JS, Wasserstein (hist approx) vs Wasserstein (samples)

Key idea:
- Histogram-based metrics (PSI/JS and even a histogram-approx Wasserstein) can
  saturate when OOD mass accumulates in the edge bins.
- Sample-based 1D Wasserstein continues to reflect *how far* OOD is.

This version uses **finite-wide bins** (no +/-inf edges) so histogram Wasserstein
is well-defined.

Dependencies:
  pip install numpy scipy matplotlib

Run:
  python ood_metrics_demo.py
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance


# -----------------------------
# Metrics
# -----------------------------
def _safe_probs(counts: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = counts.astype(np.float64)
    p = p / max(p.sum(), 1.0)
    return np.clip(p, eps, 1.0)


def psi_from_hist(p_counts: np.ndarray, q_counts: np.ndarray, eps: float = 1e-12) -> float:
    """
    PSI(P->Q) = sum_i (q_i - p_i) * ln(q_i / p_i)
    Where P is reference (expected) and Q is production (actual).
    """
    p = _safe_probs(p_counts, eps)
    q = _safe_probs(q_counts, eps)
    return float(np.sum((q - p) * np.log(q / p)))


def js_from_hist(p_counts: np.ndarray, q_counts: np.ndarray, eps: float = 1e-12) -> float:
    """
    Jensen-Shannon divergence using histogram probabilities.
    JS(P,Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), M = 0.5*(P+Q)
    """
    p = _safe_probs(p_counts, eps)
    q = _safe_probs(q_counts, eps)
    m = 0.5 * (p + q)

    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * kl_pm + 0.5 * kl_qm)


def wasserstein_hist_from_counts(
    p_counts: np.ndarray, q_counts: np.ndarray, bin_edges: np.ndarray, eps: float = 1e-12
) -> float:
    """
    1D Wasserstein (EMD) approximation using histogram bins.

    Uses bin midpoints as support points, and approximates:
      W1 ≈ ∫ |CDF_P(x) - CDF_Q(x)| dx
    discretized as:
      W1 ≈ sum_i |CDFdiff_i| * bin_width_i

    bin_edges must be finite (no +/-inf).
    """
    if not np.all(np.isfinite(bin_edges)):
        raise ValueError("bin_edges must be finite for histogram Wasserstein.")

    p = _safe_probs(p_counts, eps)
    q = _safe_probs(q_counts, eps)

    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    cdf_diff = np.abs(cdf_p - cdf_q)

    widths = np.diff(bin_edges)  # len = num_bins
    if len(widths) != len(p):
        raise ValueError("bin_edges length must be num_bins+1.")

    return float(np.sum(cdf_diff * widths))


# -----------------------------
# Binning (finite-wide)
# -----------------------------
def make_finite_wide_bin_edges_from_reference(
    ref: np.ndarray,
    num_quantile_bins: int = 10,
    widen_std: float = 6.0,
) -> np.ndarray:
    """
    Build bins using reference quantiles, but force finite "wide" outer edges.
    This avoids +/-inf while still catching OOD samples in the first/last bin.

    Strategy:
      - interior edges = reference quantiles
      - outer edges = mean(ref) +/- widen_std * std(ref)
    """
    ref = np.asarray(ref, dtype=np.float64)
    mu = float(ref.mean())
    sigma = float(ref.std(ddof=1))
    lo = mu - widen_std * sigma
    hi = mu + widen_std * sigma

    edges = np.quantile(ref, np.linspace(0, 1, num_quantile_bins + 1))
    edges = edges.astype(np.float64)

    # Replace outer edges with wide finite edges
    edges[0] = lo
    edges[-1] = hi

    # Ensure strictly increasing (rarely quantiles can tie)
    # If ties occur, add tiny jitter to enforce monotonicity
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-9

    return edges


def hist_counts(x: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(x, bins=bin_edges)
    return counts


# -----------------------------
# Experiment
# -----------------------------
def run_experiment(
    n_ref: int = 100_000,
    n_prod: int = 100_000,
    deltas: np.ndarray | None = None,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)

    if deltas is None:
        deltas = np.linspace(0, 8, 33)  # mean-shift severity

    # Reference distribution: N(0,1)
    ref = rng.normal(loc=0.0, scale=1.0, size=n_ref)

    # Bins derived from reference
    bin_edges = make_finite_wide_bin_edges_from_reference(ref, num_quantile_bins=10, widen_std=6.0)
    ref_counts = hist_counts(ref, bin_edges)

    rows = []
    for d in deltas:
        prod = rng.normal(loc=float(d), scale=1.0, size=n_prod)
        prod_counts = hist_counts(prod, bin_edges)

        # Time histogram metrics
        t0 = time.perf_counter()
        psi = psi_from_hist(ref_counts, prod_counts)
        js = js_from_hist(ref_counts, prod_counts)
        w1_hist = wasserstein_hist_from_counts(ref_counts, prod_counts, bin_edges)
        t_hist = time.perf_counter() - t0

        # Time sample Wasserstein
        t1 = time.perf_counter()
        w1_sample = float(wasserstein_distance(ref, prod))
        t_sample = time.perf_counter() - t1

        rows.append(
            dict(
                delta=float(d),
                PSI=psi,
                JS=js,
                W1_hist=w1_hist,
                W1_sample=w1_sample,
                time_hist_sec=t_hist,
                time_w1_sample_sec=t_sample,
            )
        )

    return rows, bin_edges


def main():

    deltas = np.linspace(0, 10, 41)  # stronger shifts to see saturation
    rows, _ = run_experiment(
        n_ref=100_000,
        n_prod=100_000,
        deltas=deltas,
        seed=42,
    )

    # Convert to arrays for plotting
    delta = np.array([r["delta"] for r in rows])
    psi = np.array([r["PSI"] for r in rows])
    js = np.array([r["JS"] for r in rows])
    w1_hist = np.array([r["W1_hist"] for r in rows])
    w1_sample = np.array([r["W1_sample"] for r in rows])

    t_hist_ms = 1e3 * np.array([r["time_hist_sec"] for r in rows])
    t_w1_ms = 1e3 * np.array([r["time_w1_sample_sec"] for r in rows])

    # Plot 1: metrics vs severity
    plt.figure()
    plt.plot(delta, psi, label="PSI (hist)")
    plt.plot(delta, js, label="JS divergence (hist)")
    plt.plot(delta, w1_hist, label="Wasserstein (hist approx)")
    plt.plot(delta, w1_sample, label="Wasserstein (samples)")
    plt.xlabel("OOD severity (mean shift δ)")
    plt.ylabel("Metric value")
    plt.title("OOD severity vs drift metrics")
    plt.legend()
    plt.tight_layout()

    # Plot 2: runtime vs severity
    plt.figure()
    plt.plot(delta, t_hist_ms, label="Histogram metrics (PSI+JS+W1_hist)")
    plt.plot(delta, t_w1_ms, label="W1 sample (scipy)")
    plt.xlabel("OOD severity (mean shift δ)")
    plt.ylabel("Runtime (ms) per δ")
    plt.title("Compute cost: histogram vs sample Wasserstein")
    plt.legend()
    plt.tight_layout()

    # Print a small timing summary
    print("\nTiming summary over all deltas:")
    print(f"  Histogram metrics: mean={t_hist_ms.mean():.3f} ms, p95={np.percentile(t_hist_ms, 95):.3f} ms")
    print(f"  Sample W1        : mean={t_w1_ms.mean():.3f} ms, p95={np.percentile(t_w1_ms, 95):.3f} ms")
    print("\nTip: Increase n_ref/n_prod to amplify the runtime gap (sample W1 grows with N due to sorting).")

    plt.show()


if __name__ == "__main__":
    main()
