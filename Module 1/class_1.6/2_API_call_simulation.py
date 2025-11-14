# Copyright 2025 Ankur Mohan
# Simulate calling two APIs in parallel and waiting for both (Z = max(A, B))
# A ~ N(40, 30^2), B ~ N(70, 30^2)
# Plot histograms for A, B, and Z.

import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
mu_A, mu_B = 50.0, 170.0     # mean latencies (ms)
sigma_A, sigma_B = 80.0, 30.0
N = 1_000_000               # number of simulated requests

# --- Simulation ---
rng = np.random.default_rng(42)
A = rng.normal(mu_A, sigma_A, N)
B = rng.normal(mu_B, sigma_B, N)
Z = np.maximum(A, B)        # system waits for both

# --- Summary statistics ---
print(f"Mean(A) = {np.mean(A):.2f} ms, p99(A) = {np.quantile(A, 0.99):.2f} ms")
print(f"Mean(B) = {np.mean(B):.2f} ms, p99(B) = {np.quantile(B, 0.99):.2f} ms")
print(f"Mean(Z) = {np.mean(Z):.2f} ms, p99(Z) = {np.quantile(Z, 0.99):.2f} ms")

# --- Plot histograms ---
plt.figure(figsize=(8,5))
bins = np.linspace(-40, 350, 150)  # common x-axis bins for comparability

plt.hist(A, bins=bins, color='skyblue', alpha=0.5, density=True, label='API A (μ=40)')
plt.hist(B, bins=bins, color='orange', alpha=0.5, density=True, label='API B (μ=70)')
plt.hist(Z, bins=bins, color='green', alpha=0.6, density=True, label='System Z = max(A,B)')

plt.title("Response Time Distributions for A, B, and System Z")
plt.xlabel("Response time (ms)")
plt.ylabel("Probability density")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
