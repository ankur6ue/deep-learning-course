import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

# ---- 1) Synthetic data with correlation + nonlinearity + interaction
def make_demo_data(n=6000, seed=0, rho_noise=0.08):
    rng = np.random.default_rng(seed)

    x1 = rng.uniform(-3, 3, n)
    # x2 is highly correlated (redundant) proxy for x1
    x2 = x1 + rho_noise * rng.normal(size=n)

    x3 = rng.uniform(-2, 2, n)
    x4 = rng.normal(0, 1, n)

    # Nonlinear main effect of x1
    nonlinear = 2.5 * np.sin(1.4 * x1)

    # Interaction: effect of x3 depends on sign of x1
    interaction = 2.0 * x3 * (x1 > 0).astype(float) - 1.0 * x3 * (x1 <= 0).astype(float)

    # Another nonlinear effect
    other = 0.8 * (x4 ** 2)

    y = nonlinear + interaction + other + 0.2 * rng.normal(size=n)

    X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4})
    return X, y


X, y = make_demo_data()
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)

# ---- 2) Fit a nonlinear model (tree-based)
model = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.08, random_state=0)
model.fit(Xtr, ytr)

pred = model.predict(Xte)
print("R2:", round(r2_score(yte, pred), 3))

# ---- 3) SHAP (global + local)
# If import shap fails in your environment: pip install shap numba (and avoid incompatible coverage packages)
import shap

# Use a background sample for expected value / baseline
background = Xtr.sample(400, random_state=0)

# Tree models -> TreeExplainer under the hood
explainer = shap.Explainer(model, background)
sv = explainer(Xte)   # sv is a shap.Explanation

# ========== GLOBAL ==========
# (A) Global importance (mean |SHAP|): "what features move predictions most?"
plt.figure()
shap.plots.bar(sv, max_display=10, show=False)  # bar plot of mean(|phi|)
plt.title("Global SHAP importance (mean |contribution|)")
plt.gcf().set_size_inches(10, 7)   # make it taller
plt.tight_layout()

# (B) Beeswarm summary: shows direction + spread + heterogeneity
plt.figure()
shap.plots.beeswarm(sv, max_display=10, show=False)
plt.title("SHAP summary (direction + magnitude + heterogeneity)")
plt.gcf().set_size_inches(10, 7)   # make it taller
plt.tight_layout()

# ========== NONLINEARITY ==========
# Dependence plot: SHAP value vs feature value reveals nonlinear functional form.
# (In this dataset: x1 should show a sine-like pattern.)
plt.figure()
shap.plots.scatter(sv[:, "x1"], color=sv[:, "x3"], show=False)
plt.title("Dependence: x1 (nonlinear pattern). Color by x3")
plt.gcf().set_size_inches(10, 7)   # make it taller
plt.tight_layout()
# ========== INTERACTION ==========
# Interaction shows up as "different SHAP(x3) at the same x3 value depending on x1"
plt.figure()
shap.plots.scatter(sv[:, "x3"], color=sv[:, "x1"], show=False)
plt.title("Dependence: x3 (interaction). Color by x1")
plt.gcf().set_size_inches(10, 7)   # make it taller
plt.tight_layout()

# ========== CORRELATION / REDUNDANCY ==========
# x1 and x2 are redundant; SHAP often splits credit across them.
# A stable way to "deal with" correlated features for global reporting is to GROUP them.
def grouped_global_importance(explanation, groups):
    # mean absolute group contribution: mean_i | sum_{j in group} phi_ij |
    phi = explanation.values  # (n_samples, n_features)
    cols = list(explanation.feature_names)
    out = {}
    for gname, feats in groups.items():
        idx = [cols.index(f) for f in feats]
        out[gname] = np.mean(np.abs(phi[:, idx].sum(axis=1)))
    return pd.Series(out).sort_values(ascending=False)

groups = {
    "group(x1,x2)": ["x1", "x2"],   # correlated pair
    "x3": ["x3"],
    "x4": ["x4"],
}

print("\nGrouped global importance (mean |sum SHAP|):")
print(grouped_global_importance(sv, groups))

# ========== LOCAL (single prediction) ==========
# For one point: baseline + contributions = prediction
i = 0
plt.figure()
shap.plots.waterfall(sv[i], max_display=10)
plt.title("Local explanation (one point): baseline + feature contributions")

plt.show()
