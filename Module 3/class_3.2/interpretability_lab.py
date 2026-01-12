# © 2026 Ankur Mohan (adapt as needed)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ---------------------------
# Data generators
# ---------------------------

def make_nonlinear_regression(n=3000, seed=0):
    """
    y = sin(x1) + 0.5*x2 + noise
    -> PDP for x1 should recover a sine shape for flexible models.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-3, 3, size=n)
    x2 = rng.normal(0, 1, size=n)
    X = np.c_[x1, x2]
    y = np.sin(x1) + 0.5 * x2 + rng.normal(0, 0.2, size=n)
    feature_names = ["x1", "x2"]
    return X, y, feature_names


def make_interaction_regression(n=3000, seed=1):
    """
    y = x1*x2 + 0.2*x1 + noise
    -> ICE for x1 varies with x2 (heterogeneous effect).
    """
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-2, 2, size=n)
    x2 = rng.uniform(-2, 2, size=n)
    X = np.c_[x1, x2]
    y = x1 * x2 + 0.2 * x1 + rng.normal(0, 0.2, size=n)
    feature_names = ["x1", "x2"]
    return X, y, feature_names


def make_correlated_regression(n=4000, rho=0.9, seed=2):
    """
    x2 is highly correlated with x3 (x3 ~= x2).
    True target depends on x1 and x3, NOT on x2.
    PDP(x2) breaks the x2-x3 relationship -> off-manifold -> can look like x2 has an effect.
    ALE(x2) is more robust because it uses local differences where x2-x3 coupling holds.
    """
    eps = 0.05
    rng = np.random.default_rng(seed)

    x1 = rng.normal(0, 1, size=n)
    x2 = rng.normal(0, 1, size=n)
    # Make x3 almost deterministically tied to x2
    x3 = rho * x2 + np.sqrt(1 - rho**2) * rng.normal(0, 1, size=n)

    X = np.c_[x1, x2, x3]

    # True function ignores x2 directly
    y = 2.0 * np.tanh(x1) + 1.5 * np.sin(2 * x3) + eps * rng.normal(0, 1, size=n)

    feature_names = ["x1", "x2", "x3"]
    return X, y, feature_names


# ---------------------------
# PDP / ICE utilities
# ---------------------------

def ice_curves(model, X, feature_idx, grid, pred_fn=None):
    """
    Returns ICE matrix of shape (n_samples, len(grid)).
    pred_fn(model, X) -> predictions. If None uses model.predict.
    """
    if pred_fn is None:
        pred_fn = lambda m, A: m.predict(A)

    X = np.asarray(X)
    n = X.shape[0]
    curves = np.zeros((n, len(grid)))
    X_tmp = X.copy()

    for k, z in enumerate(grid):
        X_tmp[:, feature_idx] = z
        curves[:, k] = pred_fn(model, X_tmp)

    return curves


def pdp_from_ice(ice):
    return ice.mean(axis=0)


# ---------------------------
# 1D ALE (simple, dependency-free)
# ---------------------------

def ale_1d(model, X, feature_idx, bins=20, pred_fn=None):
    """
    Simple 1D ALE implementation using binwise local differences.
    Returns (bin_centers, ale_values_centered).
    """
    if pred_fn is None:
        pred_fn = lambda m, A: m.predict(A)

    X = np.asarray(X)
    xj = X[:, feature_idx]
    edges = np.quantile(xj, np.linspace(0, 1, bins + 1))

    # Remove duplicate edges if feature has ties
    edges = np.unique(edges)
    if len(edges) < 3:
        raise ValueError("Not enough unique bin edges for ALE.")

    K = len(edges) - 1
    deltas = np.zeros(K)
    counts = np.zeros(K)

    X_lo = X.copy()
    X_hi = X.copy()

    for k in range(K):
        lo, hi = edges[k], edges[k + 1]
        mask = (xj >= lo) & (xj <= hi if k == K - 1 else xj < hi)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue

        X_lo[idx, feature_idx] = lo
        X_hi[idx, feature_idx] = hi
        f_hi = pred_fn(model, X_hi[idx])
        f_lo = pred_fn(model, X_lo[idx])

        deltas[k] = np.mean(f_hi - f_lo)
        counts[k] = len(idx)

    # Accumulate
    ale = np.cumsum(deltas)

    # Map to bin centers
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Center ALE to have mean 0 over the empirical distribution
    # (weighted by counts for stability)
    if counts.sum() > 0:
        ale_centered = ale - np.average(ale, weights=np.maximum(counts, 1e-12))
    else:
        ale_centered = ale - ale.mean()

    return centers, ale_centered


# ---------------------------
# Plot helpers
# ---------------------------

def plot_pdp_ice_ale(model, X, feature_idx, feature_name, title, grid=None):
    xj = X[:, feature_idx]
    if grid is None:
        lo, hi = np.percentile(xj, [2, 98])
        grid = np.linspace(lo, hi, 60)

    ice = ice_curves(model, X, feature_idx, grid)
    pdp = pdp_from_ice(ice)
    centers, ale_vals = ale_1d(model, X, feature_idx, bins=20)

    plt.figure()
    # ICE: plot a subset for readability
    n = ice.shape[0]
    subset = np.linspace(0, n - 1, min(n, 80), dtype=int)
    for i in subset:
        plt.plot(grid, ice[i], alpha=0.15)

    plt.plot(grid, pdp, linewidth=2, label="PDP (avg ICE)")
    plt.plot(centers, ale_vals, linewidth=2, label="ALE (centered)")
    plt.xlabel(feature_name)
    plt.ylabel("prediction")
    plt.title(title)
    plt.legend()
    plt.tight_layout()


def demo_one_dataset(X, y, feature_names, task="regression", seed=0):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed)

    if task == "regression":
        lin = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
        gbr = GradientBoostingRegressor(random_state=seed, max_depth=3)
        lin.fit(Xtr, ytr)
        gbr.fit(Xtr, ytr)

        # Model-agnostic permutation importance (teaches "importance vs effect")
        imp_lin = permutation_importance(lin, Xte, yte, n_repeats=10, random_state=seed)
        imp_gbr = permutation_importance(gbr, Xte, yte, n_repeats=10, random_state=seed)

        print("\nPermutation importance (Ridge):")
        for j, name in enumerate(feature_names):
            print(f"  {name}: {imp_lin.importances_mean[j]:.4f} ± {imp_lin.importances_std[j]:.4f}")

        print("\nPermutation importance (GBR):")
        for j, name in enumerate(feature_names):
            print(f"  {name}: {imp_gbr.importances_mean[j]:.4f} ± {imp_gbr.importances_std[j]:.4f}")

        # Plot interpretability for feature 0 and 1
        for j, name in enumerate(feature_names):
            plot_pdp_ice_ale(gbr, Xte, j, name, title=f"GBR: PDP/ICE/ALE for {name}")

    else:
        # Assume y is {0,1}
        logreg = Pipeline([("scaler", StandardScaler()),
                           ("clf", LogisticRegression(max_iter=2000))])
        rf = RandomForestClassifier(n_estimators=300, random_state=seed)
        logreg.fit(Xtr, ytr)
        rf.fit(Xtr, ytr)

        # Use probability for interpretability curves
        pred_fn = lambda m, A: m.predict_proba(A)[:, 1]

        # Permutation importance (probability space)
        imp_log = permutation_importance(logreg, Xte, yte, n_repeats=10, random_state=seed)
        imp_rf = permutation_importance(rf, Xte, yte, n_repeats=10, random_state=seed)

        print("\nPermutation importance (LogReg):")
        for j, name in enumerate(feature_names):
            print(f"  {name}: {imp_log.importances_mean[j]:.4f} ± {imp_log.importances_std[j]:.4f}")

        print("\nPermutation importance (RF):")
        for j, name in enumerate(feature_names):
            print(f"  {name}: {imp_rf.importances_mean[j]:.4f} ± {imp_rf.importances_std[j]:.4f}")

        # ICE/PDP/ALE on probability scale
        for j, name in enumerate(feature_names):
            xj = Xte[:, j]
            lo, hi = np.percentile(xj, [2, 98])
            grid = np.linspace(lo, hi, 60)

            ice = ice_curves(rf, Xte, j, grid, pred_fn=pred_fn)
            pdp = pdp_from_ice(ice)
            centers, ale_vals = ale_1d(rf, Xte, j, bins=20, pred_fn=pred_fn)

            plt.figure()
            subset = np.linspace(0, len(Xte) - 1, min(len(Xte), 80), dtype=int)
            for i in subset:
                plt.plot(grid, ice[i], alpha=0.15)
            plt.plot(grid, pdp, linewidth=2, label="PDP (avg ICE)")
            plt.plot(centers, ale_vals, linewidth=2, label="ALE (centered)")
            plt.xlabel(name)
            plt.ylabel("P(y=1)")
            plt.title(f"RF: PDP/ICE/ALE for {name}")
            plt.legend()
            plt.tight_layout()


def main():
    # 1) Nonlinear main effect
    X, y, names = make_nonlinear_regression()
    print("=== Synthetic: nonlinear main effect ===")
    # demo_one_dataset(X, y, names, task="regression")

    # 2) Interaction: ICE will fan out
    X, y, names = make_interaction_regression()
    print("\n=== Synthetic: interaction ===")
    # demo_one_dataset(X, y, names, task="regression")

    # 3) Correlation: PDP pitfalls vs ALE
    X, y, names = make_correlated_regression(rho=0.9)
    print("\n=== Synthetic: correlated features (PDP pitfalls) ===")
    demo_one_dataset(X, y, names, task="regression")

    plt.show()


if __name__ == "__main__":
    main()
