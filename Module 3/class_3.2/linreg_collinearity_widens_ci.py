# linreg_collinearity_widens_ci.py
# © 2025 Ankur Mohan

import numpy as np

def ols_ci_widths(X, y, alpha=0.05):
    n = X.shape[0]
    Xb = np.concatenate([np.ones((n, 1)), X], axis=1)
    p = Xb.shape[1]
    XtX_inv = np.linalg.inv(Xb.T @ Xb)
    beta = XtX_inv @ (Xb.T @ y)

    resid = y - Xb @ beta
    dof = n - p
    sigma2 = (resid.T @ resid) / dof
    se = np.sqrt(np.diag(sigma2 * XtX_inv))

    try:
        from scipy.stats import t
        tcrit = t.ppf(1.0 - alpha / 2.0, dof)
    except Exception:
        tcrit = 1.96

    ci_halfwidth = tcrit * se
    return beta, se, ci_halfwidth

def main(seed=0):
    rng = np.random.default_rng(seed)
    n = 600

    # Base feature
    x1 = rng.normal(size=n)

    # Case A: x2 independent
    x2_indep = rng.normal(size=n)

    # Case B: x2 highly correlated with x1
    x2_corr = x1 + 0.02 * rng.normal(size=n)

    # True model depends on x1 + x2 (both matter)
    # In correlated case, the model can’t "decide" how to split credit between x1 and x2
    beta_true = np.array([1.0, 1.0])

    # Generate y using the correlated version (you can switch to indep too)
    y_indep = (np.stack([x1, x2_indep], axis=1) @ beta_true) + 0.8 * rng.normal(size=n)
    y_corr  = (np.stack([x1, x2_corr], axis=1)  @ beta_true) + 0.8 * rng.normal(size=n)

    betaA, seA, hwA = ols_ci_widths(np.stack([x1, x2_indep], axis=1), y_indep)
    betaB, seB, hwB = ols_ci_widths(np.stack([x1, x2_corr], axis=1), y_corr)

    names = ["intercept", "x1", "x2"]

    print("=== Independent features ===")
    for j, name in enumerate(names):
        print(f"{name:>9s}: beta={betaA[j]: .4f}  SE={seA[j]: .4f}  95% halfwidth={hwA[j]: .4f}")

    print("\n=== Highly correlated features ===")
    for j, name in enumerate(names):
        print(f"{name:>9s}: beta={betaB[j]: .4f}  SE={seB[j]: .4f}  95% halfwidth={hwB[j]: .4f}")

    print("\nTeaching point: multicollinearity -> (X^T X)^(-1) has large diagonals -> big SE -> wide CIs.")
    print("Predictions may still be good, but the 'story' (individual coefficients) becomes uncertain.")

if __name__ == "__main__":
    main()
