"""
Does the random forest 'know' when it is extrapolating?

The simplest self-reported uncertainty signal from a forest is the spread
(standard deviation) of the predictions of its individual trees at a query
point.  We compare that spread, and the actual error, between:
  - in-range test points (seed-42 split of the main dataset),
  - angle-of-attack extrapolation points,
  - shape extrapolation points.
We also count how often linear regression predicts a physically impossible
negative drag coefficient on the extrapolation set, and how often a trivial
input-range check would have flagged the query.
Outputs: ../data/reliability_results.csv and printed summary.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

F = ["m", "p", "t", "alpha"]
df = pd.read_csv("../data/main_dataset.csv"); ex = pd.read_csv("../data/extrapolation_dataset.csv")
X = df[F].values
tr, te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)

lo, hi = X[tr].min(axis=0), X[tr].max(axis=0)          # training range of each input
def out_of_range(Xq): return np.any((Xq < lo) | (Xq > hi), axis=1)

groups = {"in-range test split": (X[te], {t: df[t].values[te] for t in ["Cl", "Cd"]})}
for k in ["alpha", "shape"]:
    m = (ex.extrap_type == k).values
    groups[f"extrapolation ({k})"] = (ex[F].values[m], {t: ex[t].values[m] for t in ["Cl", "Cd"]})

rows = []
ref_spread = {}
for tgt in ["Cl", "Cd"]:
    y = df[tgt].values
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1).fit(X[tr], y[tr])
    lr = LinearRegression().fit(X[tr], y[tr])
    for g, (Xq, yq) in groups.items():
        per_tree = np.column_stack([t.predict(Xq) for t in rf.estimators_])
        spread = per_tree.std(axis=1)
        err = np.abs(rf.predict(Xq) - yq[tgt])
        if g == "in-range test split": ref_spread[tgt] = np.percentile(spread, 95)
        rows.append(dict(target=tgt, group=g, n=len(Xq), mean_tree_spread=spread.mean(),
                         mean_abs_error=err.mean(), spread_over_error=spread.mean() / err.mean(),
                         frac_above_inrange_p95=np.mean(spread > ref_spread[tgt]),
                         frac_flagged_by_range_check=out_of_range(Xq).mean(),
                         lr_negative_pred_frac=(np.mean(lr.predict(Xq) < 0) if tgt == "Cd" else np.nan)))
r = pd.DataFrame(rows); r.to_csv("../data/reliability_results.csv", index=False)
pd.set_option("display.width", 200); print(r.round(5).to_string(index=False))
