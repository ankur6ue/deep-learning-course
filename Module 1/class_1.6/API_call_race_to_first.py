# Copyright 2025 Ankur Mohan
# Simulation of 3 parallel API calls with identical Normal(mu, sigma)
# Z = max(A,B,C); verify simulated p99 matches analytic formula.

import numpy as np
from scipy.stats import norm

# Parameters (same for all three APIs)
mu = 60.0    # mean latency in ms
sigma = 40.0  # std deviation in ms
p = 0.99      # target percentile

# --- Analytic p99 ---
z_theoretical = mu + sigma * norm.ppf(1 - (1 - p) ** (1/3))
print(f"Theoretical p99 (Z=max of 3 Normals): {z_theoretical:.3f} ms")

# --- Simulation ---
rng = np.random.default_rng(42)
N = 1_000_000
A = rng.normal(mu, sigma, N)
B = rng.normal(mu, sigma, N)
C = rng.normal(mu, sigma, N)
Z = np.minimum.reduce([A, B, C])

z_empirical = np.quantile(Z, p)
print(f"Empirical p99 (1M trials): {z_empirical:.3f} ms")
print(f"Relative error: {100 * abs(z_empirical - z_theoretical) / z_theoretical:.3f}%")
