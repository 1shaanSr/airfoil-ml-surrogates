"""
Train and evaluate the ML surrogate models.

Two targets (Cl, Cd), each modeled separately (simpler to report and
explain than a joint multi-output model) with two methods:
  - Linear regression: the simplest possible baseline.
  - Random forest: the main surrogate, a standard, basic, well-understood
    ensemble-of-decision-trees regressor (no probabilistic/Gaussian
    machinery, no manual tuning beyond a couple of obvious hyperparameters).

Evaluated on two held-out sets, deliberately kept separate:
  - test split of the main grid (interpolation: same parameter ranges as
    training, just different specific points)
  - extrapolation_dataset.csv (points outside the training parameter
    ranges entirely -- never seen in any form during training)

Metrics: R^2, RMSE, and a relative error defined as MAE / mean(|y_true|)
(robust near-zero-value points, unlike pointwise percent error).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

FEATURES = ["m", "p", "t", "alpha"]


def relative_error(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred) / np.mean(np.abs(y_true))


def evaluate(model, X, y_true, label):
    y_pred = model.predict(X)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    relerr = relative_error(y_true, y_pred)
    print(f"  {label:22s}  R2={r2:7.4f}   RMSE={rmse:.5f}   relative_error={relerr*100:6.2f}%")
    return dict(label=label, r2=r2, rmse=rmse, relative_error=relerr)


if __name__ == "__main__":
    df = pd.read_csv("../data/main_dataset.csv")
    df_extrap = pd.read_csv("../data/extrapolation_dataset.csv")

    X = df[FEATURES].values
    results = []

    for target in ["Cl", "Cd"]:
        print(f"\n=== Target: {target} ===")
        y = df[target].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        lin = LinearRegression().fit(X_train, y_train)
        rf = RandomForestRegressor(
            n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
        ).fit(X_train, y_train)

        print(" -- held-out test split (interpolation) --")
        results.append(dict(target=target, split="test", **evaluate(lin, X_test, y_test, "linear regression")))
        results.append(dict(target=target, split="test", **evaluate(rf, X_test, y_test, "random forest")))

        X_ex = df_extrap[FEATURES].values
        y_ex = df_extrap[target].values
        print(" -- extrapolation set (outside training ranges) --")
        results.append(dict(target=target, split="extrapolation", **evaluate(lin, X_ex, y_ex, "linear regression")))
        results.append(dict(target=target, split="extrapolation", **evaluate(rf, X_ex, y_ex, "random forest")))

        # break down extrapolation by type (alpha vs shape) for the random forest
        for etype in df_extrap["extrap_type"].unique():
            mask = df_extrap["extrap_type"] == etype
            r = evaluate(rf, X_ex[mask.values], y_ex[mask.values], f"RF, extrap type={etype}")
            results.append(dict(target=target, split=f"extrapolation_{etype}", **r))

    pd.DataFrame(results).to_csv("../data/surrogate_results.csv", index=False)
    print("\nSaved summary to data/surrogate_results.csv")
