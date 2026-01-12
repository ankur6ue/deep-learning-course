import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Compute decile edges for standard normal
quantiles = np.linspace(0, 1, 11)
bin_edges = norm.ppf(quantiles)

# Replace infinities for plotting clarity
bin_edges_plot = bin_edges.copy()
bin_edges_plot[0] = -4
bin_edges_plot[-1] = 4

# Plot bin edges as vertical lines
plt.figure(figsize=(8, 4))

x = np.linspace(-4, 4, 1000)
plt.plot(x, norm.pdf(x), label="Normal PDF", color="black")

for edge in bin_edges_plot:
    plt.axvline(edge, color="red", linestyle="--", alpha=0.7)

plt.title("Decile Bin Edges for a Normal Distribution")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()