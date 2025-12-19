#!/usr/bin/env python3
"""
Compute KL, PSI, JS and Wasserstein drift metrics for MNIST monitoring example.

Assumptions:
- Reference and production data are stored as parquet files
- Columns include things like:
    - pred_conf (scalar prediction confidence)
    - embedding_0, embedding_1, ... (embedding dimensions)
- You can compute metrics either:
    - For a single scalar feature (e.g. pred_conf)
    - For all embedding_* columns (metrics per-dim + macro-average)

Example usage:

  # On scalar feature
  python mnist_drift_metrics.py \
      --ref ../data/mnist_reference.parquet \
      --prod ../data/mnist_production.parquet \
      --feature pred_conf

  # On all embedding dims
  python mnist_drift_metrics.py \
      --ref ../data/mnist_reference.parquet \
      --prod ../data/mnist_production.parquet \
      --feature-type embedding
"""

import argparse
from dataclasses import dataclass
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import entropy, wasserstein_distance


# ------------------------
#  Utility: binning & hist
# ------------------------

def compute_hist_probs(
    ref: np.ndarray,
    prod: np.ndarray,
    num_bins: int = 10,
    use_deciles: bool = True,
    epsilon: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build histograms for reference & production on a single feature.

    Parameters
    ----------
    ref : np.ndarray
        Reference sample values (1D)
    prod : np.ndarray
        Production sample values (1D)
    num_bins : int
        Number of bins
    use_deciles : bool
        If True, compute bin edges from ref deciles (PSI-style).
        If False, use uniform bins between min(ref, prod) and max(ref, prod).
    epsilon : float
        Small value added to each bin probability to avoid zeros in KL/PSI.

    Returns
    -------
    p : np.ndarray
        Reference bin probabilities, shape (num_bins,)
    q : np.ndarray
        Production bin probabilities, shape (num_bins,)
    bin_edges : np.ndarray
        Bin edges used, shape (num_bins + 1,)
    """
    ref = np.asarray(ref).ravel()
    prod = np.asarray(prod).ravel()

    # Drop NaNs
    ref = ref[~np.isnan(ref)]
    prod = prod[~np.isnan(prod)]

    if ref.size == 0 or prod.size == 0:
        raise ValueError("Empty ref or prod array after NaN filtering.")

    if use_deciles:
        # PSI-style: bin boundaries from ref quantiles
        quantiles = np.linspace(0.0, 1.0, num_bins + 1)
        bin_edges = np.quantile(ref, quantiles)
        # Ensure strictly increasing edges (guard against constant values)
        bin_edges = np.unique(bin_edges)
        if bin_edges.size - 1 < num_bins:
            # Not enough unique edges; fall back to uniform bins
            min_val = min(ref.min(), prod.min())
            max_val = max(ref.max(), prod.max())
            bin_edges = np.linspace(min_val, max_val, num_bins + 1)
    else:
        min_val = min(ref.min(), prod.min())
        max_val = max(ref.max(), prod.max())
        bin_edges = np.linspace(min_val, max_val, num_bins + 1)

    # Compute histogram counts
    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    prod_counts, _ = np.histogram(prod, bins=bin_edges)

    # Convert counts to probabilities
    ref_probs = ref_counts.astype(float) / ref_counts.sum()
    prod_probs = prod_counts.astype(float) / prod_counts.sum()

    # Add epsilon to avoid zeros in KL/PSI, then renormalize
    ref_probs = ref_probs + epsilon
    prod_probs = prod_probs + epsilon
    ref_probs = ref_probs / ref_probs.sum()
    prod_probs = prod_probs / prod_probs.sum()

    return ref_probs, prod_probs, bin_edges

def wasserstein_1d_from_hist(p: np.ndarray, q: np.ndarray, bin_edges: np.ndarray) -> float:
    """
    1D Wasserstein-1 distance for two discrete distributions over the same bins.

    p, q: shape (K,) probabilities for each bin (must sum to 1).
    bin_edges: shape (K+1,) edges of the bins.
    Uses the CDF difference integral approximation that is exact for piecewise-constant densities.
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    bin_edges = np.asarray(bin_edges, dtype=np.float64).ravel()

    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape.")
    if bin_edges.size != p.size + 1:
        raise ValueError("bin_edges must have length len(p)+1.")
    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("p and q must be nonnegative.")
    if not np.isclose(p.sum(), 1.0):
        p = p / p.sum()
    if not np.isclose(q.sum(), 1.0):
        q = q / q.sum()

    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)

    widths = np.diff(bin_edges)  # (K,)
    # Integral of |CDF difference| across each bin width
    return float(np.sum(np.abs(cdf_p - cdf_q) * widths))

# ------------------------
#   Metrics: KL, PSI, JS
# ------------------------

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    KL(P || Q) using discrete distributions p, q over same bins.
    """
    return float(entropy(p, q))  # scipy.stats.entropy


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Jensen–Shannon divergence:

    JS(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M),
    where M = 0.5 * (P + Q)
    """
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def wasserstein_1d(ref: np.ndarray, prod: np.ndarray) -> float:
    """
    1D Wasserstein distance (Earth Mover's Distance) between ref and prod.

    This works fine even when ref and prod have different sample sizes.
    """
    ref = np.asarray(ref).ravel()
    prod = np.asarray(prod).ravel()

    ref = ref[~np.isnan(ref)]
    prod = prod[~np.isnan(prod)]

    if ref.size == 0 or prod.size == 0:
        raise ValueError("Empty ref or prod array after NaN filtering.")

    return float(wasserstein_distance(ref, prod))


# ------------------------
#     High-level driver
# ------------------------

@dataclass
class DriftMetrics:
    kl: float
    psi: float
    js: float
    wasserstein: float


def compute_drift_for_feature(
    ref: np.ndarray,
    prod: np.ndarray,
    num_bins: int = 10,
) -> DriftMetrics:
    """
    Compute KL, PSI, JS, Wasserstein for a single scalar feature.
    """
    # p is ref (expected), q is prod (actuals)
    p, q, _ = compute_hist_probs(ref, prod, num_bins=num_bins, use_deciles=True)
    return DriftMetrics(
        kl=kl_divergence(p, q),
        psi=psi(p, q),
        js=js_divergence(p, q),
        wasserstein=wasserstein_1d(ref, prod),
    )


def compute_drift_for_embeddings(
    ref_df: pd.DataFrame,
    prod_df: pd.DataFrame,
    num_bins: int = 10,
) -> Tuple[DriftMetrics, Dict[str, DriftMetrics]]:
    """
    Compute drift metrics for all embedding_* columns, and return both:

    - macro-average metrics across dimensions
    - per-dimension metrics
    """
    emb_cols = [c for c in ref_df.columns if c.startswith("embedding")]
    if not emb_cols:
        raise ValueError("No embedding_* columns found in reference dataframe.")

    per_dim: Dict[str, DriftMetrics] = {}

    for col in emb_cols:
        ref_vals = ref_df[col].to_numpy()
        prod_vals = prod_df[col].to_numpy()
        per_dim[col] = compute_drift_for_feature(ref_vals, prod_vals, num_bins=num_bins)

    # Macro-average across dimensions
    kl_vals = np.array([m.kl for m in per_dim.values()])
    psi_vals = np.array([m.psi for m in per_dim.values()])
    js_vals = np.array([m.js for m in per_dim.values()])
    wass_vals = np.array([m.wasserstein for m in per_dim.values()])

    avg_metrics = DriftMetrics(
        kl=float(kl_vals.mean()),
        psi=float(psi_vals.mean()),
        js=float(js_vals.mean()),
        wasserstein=float(wass_vals.mean()),
    )

    return avg_metrics, per_dim


# ------------------------
#         CLI
# ------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute KL, PSI, JS, Wasserstein for MNIST monitoring example."
    )
    parser.add_argument(
        "--ref",
        type=str,
        required=False,
        default="../data/mnist_reference.parquet",
        help="Path to reference parquet file (e.g. mnist_reference.parquet)",
    )
    parser.add_argument(
        "--prod",
        type=str,
        required=False,
        default="../data/mnist_production.parquet",
        help="Path to production parquet file (e.g. mnist_production.parquet)",
    )
    parser.add_argument(
        "--feature",
        type=str,
        default="pred_conf",
        help="Scalar feature column name (e.g. pred_conf). "
             "If not provided and --feature-type=embedding, metrics will be "
             "computed for embedding_* columns.",
    )
    parser.add_argument(
        "--feature-type",
        type=str,
        choices=["scalar", "embedding"],
        default="scalar",
        help="Type of feature: 'scalar' for one column (e.g. pred_conf), "
             "'embedding' for all embedding_* cols.",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=10,
        help="Number of bins for histogram-based metrics (KL/PSI/JS).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    ref_df = pd.read_parquet(args.ref)
    prod_df = pd.read_parquet(args.prod)

    print(f"Loaded reference: {args.ref}, shape={ref_df.shape}")
    print(f"Loaded production: {args.prod}, shape={prod_df.shape}")

    if args.feature_type == "scalar":
        if args.feature is None:
            raise SystemExit("When feature-type=scalar, you must provide --feature.")
        if args.feature not in ref_df.columns:
            raise SystemExit(f"Feature '{args.feature}' not in reference dataframe.")
        if args.feature not in prod_df.columns:
            raise SystemExit(f"Feature '{args.feature}' not in production dataframe.")

        ref_vals = ref_df[args.feature].to_numpy()
        prod_vals = prod_df[args.feature].to_numpy()

        metrics = compute_drift_for_feature(ref_vals, prod_vals, num_bins=args.num_bins)

        print(f"\n=== Drift metrics for scalar feature '{args.feature}' ===")
        print(f"KL(P||Q):           {metrics.kl:.6f}")
        print(f"PSI:                {metrics.psi:.6f}")
        print(f"JS(P,Q):            {metrics.js:.6f}")
        print(f"Wasserstein (1D):   {metrics.wasserstein:.6f}")

    elif args.feature_type == "embedding":
        avg_metrics, per_dim = compute_drift_for_embeddings(
            ref_df, prod_df, num_bins=args.num_bins
        )

        print("\n=== Macro-averaged drift metrics over all embedding_* dimensions ===")
        print(f"KL(P||Q):           {avg_metrics.kl:.6f}")
        print(f"PSI:                {avg_metrics.psi:.6f}")
        print(f"JS(P,Q):            {avg_metrics.js:.6f}")
        print(f"Wasserstein (1D):   {avg_metrics.wasserstein:.6f}")

        # Optionally print a few per-dim metrics
        print("\nSample per-dimension metrics (first 5 embedding dims):")
        for col in sorted(per_dim.keys())[:5]:
            m = per_dim[col]
            print(
                f"  {col}: KL={m.kl:.4f}, PSI={m.psi:.4f}, "
                f"JS={m.js:.4f}, W1={m.wasserstein:.4f}"
            )

    else:
        raise SystemExit(f"Unknown feature_type: {args.feature_type}")


if __name__ == "__main__":
    main()
