"""
Additional statistical analysis of the surrogate models.  Every number that
appears in the extended paper is written to a CSV in ../data by this script.

Analyses (all basic, standard):
  A. Repeated random 80/20 splits (30 seeds): mean +/- std of R^2 and
     relative error.
  B. The same, but splitting by AIRFOIL SHAPE (all angles of attack of a given
     (m,p,t) go entirely to train or entirely to test): predicting genuinely
     new shapes instead of new angles of known shapes.
  C. Learning curve: test error versus number of training rows.
  D. Permutation feature importance (random forest).
  E. Residuals of the random forest by angle of attack.
  F. Two-stage 'structure-informed' model for Cd: predict Cl first, then map
     Cl -> Cd with a small linear regression on [Cl^2, t, t^2].
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance

FEATURES = ["m", "p", "t", "alpha"]
DATA = "../data/"


def metrics(y, pred):
    return r2_score(y, pred), mean_absolute_error(y, pred) / np.mean(np.abs(y))


def make_rf(seed):
    return RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)


df = pd.read_csv(DATA + "main_dataset.csv")
ex = pd.read_csv(DATA + "extrapolation_dataset.csv")
X = df[FEATURES].values
groups = df.groupby(["m", "p", "t"]).ngroup().values
print("number of distinct airfoil shapes:", len(np.unique(groups)))

# ---------------- A & B: repeated splits ----------------
rows = []
N_SEEDS = 30
for seed in range(N_SEEDS):
    idx = np.arange(len(df))
    tr_A, te_A = train_test_split(idx, test_size=0.2, random_state=seed)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_B, te_B = next(gss.split(X, groups=groups))
    for proto, tr, te in [("random rows", tr_A, te_A), ("held-out shapes", tr_B, te_B)]:
        for target in ["Cl", "Cd"]:
            y = df[target].values
            for name, mdl in [("linear regression", LinearRegression()),
                              ("random forest", make_rf(seed))]:
                mdl.fit(X[tr], y[tr])
                r2, rel = metrics(y[te], mdl.predict(X[te]))
                rows.append(dict(protocol=proto, target=target, model=name,
                                 seed=seed, r2=r2, rel=rel))
rep = pd.DataFrame(rows)
rep.to_csv(DATA + "repeated_splits_raw.csv", index=False)
summ = rep.groupby(["protocol", "target", "model"]).agg(
    r2_mean=("r2", "mean"), r2_std=("r2", "std"),
    rel_mean=("rel", "mean"), rel_std=("rel", "std")).reset_index()
summ.to_csv(DATA + "repeated_splits_summary.csv", index=False)
print("\n=== A/B: repeated splits, mean +/- std over", N_SEEDS, "seeds ===")
print(summ.to_string(index=False))

# ---------------- C: learning curve ----------------
sizes = [50, 100, 200, 400, 800, 1339]
lc_rows = []
for seed in range(20):
    tr, te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=seed)
    rng = np.random.RandomState(seed)
    tr = rng.permutation(tr)
    for n in sizes:
        sub = tr[:n]
        for target in ["Cl", "Cd"]:
            y = df[target].values
            for name, mdl in [("linear regression", LinearRegression()),
                              ("random forest", make_rf(seed))]:
                mdl.fit(X[sub], y[sub])
                r2, rel = metrics(y[te], mdl.predict(X[te]))
                lc_rows.append(dict(n_train=len(sub), target=target, model=name,
                                    seed=seed, r2=r2, rel=rel))
lc = pd.DataFrame(lc_rows)
lc_s = lc.groupby(["target", "model", "n_train"]).agg(
    r2_mean=("r2", "mean"), r2_std=("r2", "std"),
    rel_mean=("rel", "mean"), rel_std=("rel", "std")).reset_index()
lc_s.to_csv(DATA + "learning_curve_summary.csv", index=False)
print("\n=== C: learning curve (20 seeds) ===")
print(lc_s.to_string(index=False))

# ---------------- D: permutation importance ----------------
tr, te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)
imp_rows = []
for target in ["Cl", "Cd"]:
    y = df[target].values
    rf = make_rf(42).fit(X[tr], y[tr])
    pi = permutation_importance(rf, X[te], y[te], n_repeats=30, random_state=0, scoring="r2")
    for f, mu, sd in zip(FEATURES, pi.importances_mean, pi.importances_std):
        imp_rows.append(dict(target=target, feature=f, r2_drop_mean=mu, r2_drop_std=sd))
imp = pd.DataFrame(imp_rows)
imp.to_csv(DATA + "permutation_importance.csv", index=False)
print("\n=== D: permutation importance (drop in test R^2 when a feature is shuffled) ===")
print(imp.to_string(index=False))

# ---------------- E: residuals by alpha (main split, seed 42) ----------------
res_rows = []
for target in ["Cl", "Cd"]:
    y = df[target].values
    rf = make_rf(42).fit(X[tr], y[tr])
    pred = rf.predict(X[te])
    for a, yt, yp in zip(df["alpha"].values[te], y[te], pred):
        res_rows.append(dict(target=target, alpha=a, true=yt, pred=yp, resid=yp - yt))
res = pd.DataFrame(res_rows)
res.to_csv(DATA + "rf_test_residuals.csv", index=False)
mae_alpha = res.groupby(["target", "alpha"]).resid.apply(lambda r: np.mean(np.abs(r))).reset_index(name="mae")
mae_alpha.to_csv(DATA + "rf_mae_by_alpha.csv", index=False)
print("\n=== E: RF mean absolute residual by alpha ===")
print(mae_alpha.pivot(index="alpha", columns="target", values="mae").round(5).to_string())

# ---------------- F: two-stage structure-informed Cd model ----------------
def zfeat(cl, t):
    return np.column_stack([cl ** 2, t, t ** 2])

yCl, yCd = df["Cl"].values, df["Cd"].values
stage2 = LinearRegression().fit(zfeat(yCl[tr], df["t"].values[tr]), yCd[tr])   # fit with TRUE Cl
print("\nstage-2 fitted k (coef on Cl^2) =", round(stage2.coef_[0], 5), " (generator used k = 0.01)")

lr_cl = LinearRegression().fit(X[tr], yCl[tr])
rf_cl = make_rf(42).fit(X[tr], yCl[tr])
lr_cd = LinearRegression().fit(X[tr], yCd[tr])
rf_cd = make_rf(42).fit(X[tr], yCd[tr])

def eval_sets():
    yield "test split", X[te], df["t"].values[te], yCd[te]
    Xe = ex[FEATURES].values
    yield "extrap: all", Xe, ex["t"].values, ex["Cd"].values
    for k in ["alpha", "shape"]:
        mk = (ex["extrap_type"] == k).values
        yield f"extrap: {k} only", Xe[mk], ex["t"].values[mk], ex["Cd"].values[mk]

f_rows = []
for label, Xs, ts, ys in eval_sets():
    cands = {
        "direct linear regression": lr_cd.predict(Xs),
        "direct random forest": rf_cd.predict(Xs),
        "two-stage (linear Cl -> quadratic)": stage2.predict(zfeat(lr_cl.predict(Xs), ts)),
        "two-stage (forest Cl -> quadratic)": stage2.predict(zfeat(rf_cl.predict(Xs), ts)),
    }
    for name, pred in cands.items():
        r2, rel = metrics(ys, pred)
        f_rows.append(dict(set=label, model=name, r2=r2, rel=rel))
fr = pd.DataFrame(f_rows)
fr.to_csv(DATA + "two_stage_results.csv", index=False)
print("\n=== F: Cd, direct vs two-stage ===")
print(fr.to_string(index=False))
