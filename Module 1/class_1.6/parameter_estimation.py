# Copyright 2025 Ankur Mohan
# Simulate API response times from a long-tailed (log-normal) distribution,
# estimate Normal and Log-normal parameters by MLE (SciPy + manual),
# visualize goodness of fit, and compare AIC scores.

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
# SciPy MLE
mu_norm_scipy, sigma_norm_scipy = stats.norm.fit(latencies)

# Manual MLE for Normal:
#   mu_hat = mean(x)
#   sigma_hat^2 = (1/n) * sum (x - mu_hat)^2  (MLE uses 1/n, NOT 1/(n-1))
mu_norm_manual = np.mean(latencies)
sigma_norm_manual = np.sqrt(np.mean((latencies - mu_norm_manual) ** 2))

# -----------------------------
# 3. Fit Log-normal distribution (MLE)
# -----------------------------
# SciPy's lognorm parameterization:
#   lognorm(shape=s, loc, scale)
#   X ~ lognorm(s, loc, scale) => log(X - loc) ~ Normal(log(scale), s)
# For latency, enforcing loc=0 makes sense (support x > 0)
shape_scipy, loc_scipy, scale_scipy = stats.lognorm.fit(latencies, floc=0)
# Convert to "mu, sigma" of underlying normal:
mu_logn_scipy = np.log(scale_scipy)
sigma_logn_scipy = shape_scipy

# Manual MLE for Log-normal:
#   Let y_i = log(x_i)
#   mu_hat = mean(y_i)
#   sigma_hat^2 = (1/n) * sum (y_i - mu_hat)^2
log_lat = np.log(latencies)
mu_logn_manual = np.mean(log_lat)
sigma_logn_manual = np.sqrt(np.mean((log_lat - mu_logn_manual) ** 2))

# -----------------------------
# 4. Compare parameter estimates
# -----------------------------
print("=== Normal MLE estimates ===")
print(f"SciPy:  mu = {mu_norm_scipy:.4f}, sigma = {sigma_norm_scipy:.4f}")
print(f"Manual: mu = {mu_norm_manual:.4f}, sigma = {sigma_norm_manual:.4f}")
print()

print("=== Log-normal MLE estimates (underlying Normal on log(X)) ===")
print(f"True:   mu = {mu_true:.4f}, sigma = {sigma_true:.4f}")
print(f"SciPy:  mu = {mu_logn_scipy:.4f}, sigma = {sigma_logn_scipy:.4f}")
print(f"Manual: mu = {mu_logn_manual:.4f}, sigma = {sigma_logn_manual:.4f}")
print()

# -----------------------------
# 5. Compute AIC for both models
# -----------------------------
def aic(log_likelihood: float, k: int) -> float:
    """Akaike Information Criterion: AIC = 2k - 2 log L."""
    return 2 * k - 2 * log_likelihood

# Log-likelihood under Normal model (using SciPy fit)
ll_norm = np.sum(stats.norm.logpdf(latencies, loc=mu_norm_scipy, scale=sigma_norm_scipy))

# Log-likelihood under Log-normal model (using SciPy fit)
ll_logn = np.sum(stats.lognorm.logpdf(latencies, s=shape_scipy, loc=loc_scipy, scale=scale_scipy))

# Both models effectively have k=2 free parameters (mu, sigma), with loc fixed = 0 for log-normal
k_norm = 2
k_logn = 2

aic_norm = aic(ll_norm, k_norm)
aic_logn = aic(ll_logn, k_logn)

print("=== AIC scores (smaller is better) ===")
print(f"Normal:    AIC = {aic_norm:.2f}")
print(f"Log-normal: AIC = {aic_logn:.2f}")
if aic_logn < aic_norm:
    print("Log-normal provides a better fit by AIC.")
else:
    print("Normal provides a better fit by AIC (unlikely for this simulated data).")
print()

# -----------------------------
# 6. Plot histograms + fitted PDFs
# -----------------------------
# Create grid for PDF evaluation
x = np.linspace(np.percentile(latencies, 0.1),
                np.percentile(latencies, 99.9),
                400)

# Normal PDF with SciPy MLE params
pdf_norm = stats.norm.pdf(x, loc=mu_norm_scipy, scale=sigma_norm_scipy)

# Log-normal PDF with SciPy MLE params
pdf_logn = stats.lognorm.pdf(x, s=shape_scipy, loc=loc_scipy, scale=scale_scipy)

plt.figure(figsize=(8, 5))
# Histogram of data
plt.hist(latencies, bins=80, density=True, alpha=0.4, label="Simulated data")

# Overlay PDFs
plt.plot(x, pdf_norm, label="Normal MLE fit", linewidth=2)
plt.plot(x, pdf_logn, label="Log-normal MLE fit", linewidth=2)

plt.title("API Latency: Data vs Normal/Log-normal MLE Fits")
plt.xlabel("Latency (ms)")
plt.ylabel("Density")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
