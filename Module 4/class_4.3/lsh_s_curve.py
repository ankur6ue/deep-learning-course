"""
Plot the LSH S-curve:

P_candidate(s) = 1 - (1 - s^r)^b

where:
  s = Jaccard similarity
  r = rows per band
  b = number of bands
"""

import numpy as np
import matplotlib.pyplot as plt


def lsh_candidate_prob(s, b, r):
    """
    Probability that two MinHash signatures with Jaccard similarity s
    become candidates under LSH banding with b bands and r rows per band.
    """
    return 1 - (1 - s**r)**b


def main():
    # Similarity values from 0 to 1
    s_vals = np.linspace(0, 1, 500)

    # Example (b, r) settings: same total length n = b * r = 100
    configs = [
        (20, 5),   # b=20 bands, r=5 rows/band (more selective bands)
        (50, 2),   # b=50, r=2  (less selective bands)
        (10, 10),  # b=10, r=10 (very selective bands)
    ]

    plt.figure()
    for b, r in configs:
        p_vals = lsh_candidate_prob(s_vals, b, r)
        label = f"b={b}, r={r}"
        plt.plot(s_vals, p_vals, label=label)

    plt.xlabel("Jaccard similarity s")
    plt.ylabel("P(candidate | similarity s)")
    plt.title("LSH Banding S-curves: P = 1 - (1 - s^r)^b")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
