# Copyright 2025 Ankur Mohan
# Simulate API response times from a long-tailed (log-normal) distribution,
# fit Normal, Log-normal, and Rayleigh distributions (MLE),
# compare goodness-of-fit using AIC and KL divergence,
# and visualize the histogram bins + model bin probabilities.

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# -----------------------------
# 1. Simulate "API latency" data
# -----------------------------
rng = np.random.default_rng(42)

# True underlying log-normal parameters (for simulation)
mu_true = 3.5        # mean of log(latency)
sigma_true = 0.5     # std of log(latency)
n_samples = 5000

# Latencies in milliseconds, long-tailed
latencies = rng.lognormal(mean=mu_true, sigma=sigma_true, size=n_samples)

# -----------------------------
# 2. Fit Normal distribution (MLE)
# -----------------------------
mu_norm, sigma_norm = stats.norm.fit(latencies)

# -----------------------------
# 3. Fit Log-normal distribution (MLE)
# -----------------------------
# SciPy lognorm: X ~ lognorm(s=shape, loc, scale) => log(X - loc) ~ N(log(scale), shape)
# For latency, loc=0 is natural (support x>0).
shape_logn, loc_logn, scale_logn = stats.lognorm.fit(latencies, floc=0)
mu_logn = np.log(scale_logn)       # mean of underlying normal on log(X)
sigma_logn = shape_logn            # std of underlying normal on log(X)

# -----------------------------
# 4. Fit Rayleigh distribution (MLE)
# -----------------------------
# SciPy rayleigh: rayleigh(loc, scale), support x >= loc.
# Again, loc=0 is natural for latency.
loc_ray, scale_ray = stats.rayleigh.fit(latencies, floc=0)

# -----------------------------
# 5. AIC for Normal, Log-normal, Rayleigh
# -----------------------------
def aic(log_likelihood: float, k: int) -> float:
    """Akaike Information Criterion: AIC = 2k - 2 log L."""
    return 2 * k - 2 * log_likelihood

# Log-likelihood under each model
ll_norm = np.sum(stats.norm.logpdf(latencies, loc=mu_norm, scale=sigma_norm))
ll_logn = np.sum(stats.lognorm.logpdf(latencies, s=shape_logn, loc=loc_logn, scale=scale_logn))
ll_ray  = np.sum(stats.rayleigh.logpdf(latencies, loc=loc_ray, scale=scale_ray))

# Parameter counts (k)
# Normal: mu, sigma -> k=2
# Log-normal: shape, scale (loc fixed) -> k=2
# Rayleigh: scale (loc fixed) -> k=1
aic_norm = aic(ll_norm, k=2)
aic_logn = aic(ll_logn, k=2)
aic_ray  = aic(ll_ray,  k=1)

print("=== AIC scores (smaller is better) ===")
print(f"Normal:     AIC = {aic_norm:.2f}")
print(f"Log-normal: AIC = {aic_logn:.2f}")
print(f"Rayleigh:   AIC = {aic_ray:.2f}")
print()

# -----------------------------
# 6. Histogram-based empirical distribution
# -----------------------------
# We build a discrete empirical distribution over K bins:
#   P_emp(bin k) = count_in_bin_k / n_samples
# Model bin probabilities:
#   Q_model(bin k) = CDF(b_{k+1}) - CDF(b_k)

K = 80  # number of bins; trade-off between resolution and robustness
counts, bin_edges = np.histogram(latencies, bins=K, density=False)
p_emp = counts / counts.sum()  # empirical probabilities over bins

# Helper to compute model bin probs from CDF
def model_bin_probs(cdf_fn, edges):
    """
    Given a CDF function cdf_fn(x) and histogram bin edges, compute
    model probabilities q_k = CDF(edge_{k+1}) - CDF(edge_k).
    """
    cdf_vals = cdf_fn(edges)
    # q_k = F(b_{k+1}) - F(b_k)
    q = np.diff(cdf_vals)
    return q

# Model CDFs
F_norm = lambda x: stats.norm.cdf(x, loc=mu_norm, scale=sigma_norm)
F_logn = lambda x: stats.lognorm.cdf(x, s=shape_logn, loc=loc_logn, scale=scale_logn)
F_ray  = lambda x: stats.rayleigh.cdf(x, loc=loc_ray, scale=scale_ray)

q_norm = model_bin_probs(F_norm, bin_edges)
q_logn = model_bin_probs(F_logn, bin_edges)
q_ray  = model_bin_probs(F_ray,  bin_edges)

# -----------------------------
# 7. KL divergence between P_emp and Q_model
# -----------------------------
# Discrete KL: D_KL(P || Q) = sum_k p_k log(p_k / q_k),
# with care for p_k=0 or q_k=0.
def kl_divergence(p, q, eps=1e-12):
    """
    Compute D_KL(p || q) for discrete distributions p, q over same support.
    p, q must be 1D arrays summing to 1 (we'll renormalize slightly).
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    # small epsilon for numerical stability
    p_safe = p + eps
    q_safe = q + eps
    p_safe /= p_safe.sum()
    q_safe /= q_safe.sum()
    return np.sum(p_safe * np.log(p_safe / q_safe))

kl_norm = kl_divergence(p_emp, q_norm)
kl_logn = kl_divergence(p_emp, q_logn)
kl_ray  = kl_divergence(p_emp, q_ray)

print("=== KL divergence D_KL(P_emp || Q_model) (smaller is better) ===")
print(f"Normal:     KL = {kl_norm:.6f}")
print(f"Log-normal: KL = {kl_logn:.6f}")
print(f"Rayleigh:   KL = {kl_ray:.6f}")
print()

# -----------------------------
# 8. Plot 1: Histogram + fitted PDFs
# -----------------------------
x = np.linspace(np.percentile(latencies, 0.1),
                np.percentile(latencies, 99.9),
                400)

pdf_norm = stats.norm.pdf(x, loc=mu_norm, scale=sigma_norm)
pdf_logn = stats.lognorm.pdf(x, s=shape_logn, loc=loc_logn, scale=scale_logn)
pdf_ray  = stats.rayleigh.pdf(x, loc=loc_ray, scale=scale_ray)

plt.figure(figsize=(8, 5))
plt.hist(latencies, bins=K, density=True, alpha=0.35, label="Empirical histogram")
plt.plot(x, pdf_norm, label="Normal MLE", linewidth=2)
plt.plot(x, pdf_logn, label="Log-normal MLE", linewidth=2)
plt.plot(x, pdf_ray, label="Rayleigh MLE", linewidth=2)
plt.title("Latency data with fitted Normal / Log-normal / Rayleigh PDFs")
plt.xlabel("Latency (ms)")
plt.ylabel("Density")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# -----------------------------
# 9. Plot 2: Histogram with bin edges visualized
# -----------------------------
plt.figure(figsize=(8, 5))
# Use same histogram, but draw vertical lines at bin edges
plt.hist(latencies, bins=bin_edges, density=True, alpha=0.5, label="Empirical histogram")

for edge in bin_edges:
    plt.axvline(edge, linestyle="--", linewidth=0.5, alpha=0.4)

plt.title("Histogram with bin edges (support for empirical P_k)")
plt.xlabel("Latency (ms)")
plt.ylabel("Density")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# -----------------------------
# 10. Plot 3: Bin-level P_emp vs model Q_k (visual KL intuition)
# -----------------------------
# We'll plot bar charts for p_emp and q_model for a subset of bins
# to keep the plot readable (e.g., first 30 bins).
max_bins_to_show = min(30, K)
indices = np.arange(max_bins_to_show)

width = 0.25  # bar width

plt.figure(figsize=(10, 4))
plt.bar(indices - width, p_emp[:max_bins_to_show],
        width=width, alpha=0.7, label="Empirical p_k")
plt.bar(indices, q_logn[:max_bins_to_show],
        width=width, alpha=0.7, label="Log-normal q_k")
plt.bar(indices + width, q_norm[:max_bins_to_show],
        width=width, alpha=0.7, label="Normal q_k")

plt.title("Bin probabilities: empirical vs Normal vs Log-normal (first ~30 bins)")
plt.xlabel("Bin index (k)")
plt.ylabel("Probability in bin k")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.show()
