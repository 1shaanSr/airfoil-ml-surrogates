"""
All figures for the extended paper, in one consistent style.
Model identity is fixed across figures:  physics model = black, random
forest = blue dashed, linear regression = vermilion dash-dot,
two-stage model = green dotted (colorblind-safe palette; line style is a
second encoding so figures also read in grayscale).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from naca_geometry import naca4_coordinates
from panel_method import solve_panel_method, _panel_geometry
from drag_model import profile_drag_coefficient

OUT = "../figures/"
DATA = "../data/"
FEATURES = ["m", "p", "t", "alpha"]
C_RF, C_LR, C_TS, C_TRUE = "#0072B2", "#D55E00", "#009E73", "black"
TRAIN_A = (-6, 10)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 8, "lines.linewidth": 1.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def physics(m, p, t, a):
    x, y = naca4_coordinates(m, p, t, n_panels=160)
    r = solve_panel_method(x, y, a)
    _, _, S, _ = _panel_geometry(x, y)
    return r["Cl"], profile_drag_coefficient(t, np.sum(S), r["Cl"])


# ---- models (same split as the main results: seed 42) ----
df = pd.read_csv(DATA + "main_dataset.csv")
X = df[FEATURES].values
tr, te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)
yCl, yCd = df["Cl"].values, df["Cd"].values
rf_cl = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1).fit(X[tr], yCl[tr])
rf_cd = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1).fit(X[tr], yCd[tr])
lr_cl = LinearRegression().fit(X[tr], yCl[tr])
lr_cd = LinearRegression().fit(X[tr], yCd[tr])
zf = lambda cl, t: np.column_stack([cl ** 2, t, t ** 2])
stage2 = LinearRegression().fit(zf(yCl[tr], df["t"].values[tr]), yCd[tr])

# ---------------- Fig 1: predicted vs true (test split) ----------------
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
for ax, tgt, mdl, y in zip(axes, ["$C_l$", "$C_d$"], [rf_cl, rf_cd], [yCl, yCd]):
    yp = mdl.predict(X[te]); lo, hi = min(y[te].min(), yp.min()), max(y[te].max(), yp.max())
    ax.plot([lo, hi], [lo, hi], color="0.4", lw=1, ls="-", label="perfect prediction")
    ax.scatter(y[te], yp, s=9, alpha=0.6, color=C_RF, edgecolor="none", label="random forest")
    ax.set_xlabel(f"true {tgt}"); ax.set_ylabel(f"predicted {tgt}"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT + "fig1.png"); plt.close(fig)

# ---------------- Fig 2: alpha sweep, truth vs RF vs LR ----------------
alphas = np.linspace(-16, 20, 73); m, p, t = 2, 4, 12
tru = np.array([physics(m, p, t, a) for a in alphas])
Xs = np.column_stack([np.full_like(alphas, m), np.full_like(alphas, p),
                      np.full_like(alphas, t), alphas])
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
for ax, name, col, rf, lr in [(axes[0], "$C_l$", 0, rf_cl, lr_cl), (axes[1], "$C_d$", 1, rf_cd, lr_cd)]:
    ax.axvspan(*TRAIN_A, color="0.9", label="training range of $\\alpha$")
    ax.plot(alphas, tru[:, col], color=C_TRUE, label="physics model")
    ax.plot(alphas, rf.predict(Xs), color=C_RF, ls="--", label="random forest")
    ax.plot(alphas, lr.predict(Xs), color=C_LR, ls="-.", label="linear regression")
    ax.set_xlabel("angle of attack $\\alpha$ (deg)"); ax.set_ylabel(name)
    ax.set_title(f"NACA 2412: {name} versus $\\alpha$")
axes[0].legend(frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(OUT + "fig2.png"); plt.close(fig)

# ---------------- Fig 3: shapes and Cp ----------------
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [1.15, 1]})
ax = axes[0]
for (mm, pp, tt), ls in [((0, 0, 12), "-"), ((2, 4, 12), "--"), ((6, 4, 12), ":")]:
    x, y = naca4_coordinates(mm, pp, tt, n_panels=200)
    ax.plot(x, y, color="black", ls=ls, label=f"NACA {mm}{pp}{tt:02d}")
ax.set_aspect("equal"); ax.set_xlabel("$x/c$"); ax.set_ylabel("$y/c$")
ax.set_ylim(-0.12, 0.2); ax.set_title("Airfoil sections")
ax.legend(frameon=False, loc="upper center", ncol=3, fontsize=7, columnspacing=1.0, handlelength=1.8)
ax = axes[1]
x, y = naca4_coordinates(2, 4, 12, n_panels=160)
r = solve_panel_method(x, y, 4.0); h = len(r["xc"]) // 2
ax.plot(r["xc"][:h], r["Cp"][:h], color="black", ls="-", label="upper surface")
ax.plot(r["xc"][h:], r["Cp"][h:], color="0.45", ls="--", label="lower surface")
ax.invert_yaxis(); ax.set_xlabel("$x/c$"); ax.set_ylabel("pressure coefficient $C_p$")
ax.set_title("NACA 2412, $\\alpha=4^\\circ$"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT + "fig3.png"); plt.close(fig)

# ---------------- Fig 4: Cl-alpha for several cambers; drag polars ----------------
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
aa = np.arange(-6, 11, 2)
blues = plt.cm.Blues(np.linspace(0.4, 0.95, 4))
for c, mm in zip(blues, [0, 2, 4, 6]):
    cl = [physics(mm, 4 if mm else 0, 12, a)[0] for a in aa]
    axes[0].plot(aa, cl, color=c, marker="o", ms=3.5, label=f"$m={mm}$")
axes[0].set_xlabel("$\\alpha$ (deg)"); axes[0].set_ylabel("$C_l$")
axes[0].set_title("Lift curves ($p=4$, $t=12$)"); axes[0].legend(frameon=False)
oranges = plt.cm.Oranges(np.linspace(0.4, 0.95, 3))
for c, tt in zip(oranges, [6, 12, 18]):
    pts = np.array([physics(2, 4, tt, a) for a in aa])
    axes[1].plot(pts[:, 1], pts[:, 0], color=c, marker="o", ms=3.5, label=f"$t={tt}\\%$")
axes[1].set_xlabel("$C_d$"); axes[1].set_ylabel("$C_l$")
axes[1].set_title("Drag polars ($m=2$, $p=4$)"); axes[1].legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT + "fig4.png"); plt.close(fig)

# ---------------- Fig 5: learning curves ----------------
lc = pd.read_csv(DATA + "learning_curve_summary.csv")
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
for ax, tgt, nm in zip(axes, ["Cl", "Cd"], ["$C_l$", "$C_d$"]):
    for model, col, ls, lab in [("random forest", C_RF, "--", "random forest"),
                                ("linear regression", C_LR, "-.", "linear regression")]:
        s = lc[(lc.target == tgt) & (lc.model == model)]
        ax.errorbar(s.n_train, 100 * s.rel_mean, yerr=100 * s.rel_std, color=col, ls=ls,
                    marker="o", ms=3.5, capsize=2, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("number of training rows"); ax.set_ylabel("relative error (%)")
    ax.set_title(f"{nm}: test error versus training-set size"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT + "fig5.png"); plt.close(fig)

# ---------------- Fig 6: permutation importance ----------------
imp = pd.read_csv(DATA + "permutation_importance.csv")
labels = {"m": "camber $m$", "p": "camber position $p$", "t": "thickness $t$", "alpha": "angle of attack $\\alpha$"}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharex=True)
for ax, tgt, nm in zip(axes, ["Cl", "Cd"], ["$C_l$", "$C_d$"]):
    s = imp[imp.target == tgt].set_index("feature").loc[["m", "p", "t", "alpha"]]
    ypos = np.arange(len(s))[::-1]
    ax.barh(ypos, s.r2_drop_mean, xerr=s.r2_drop_std, color=C_RF, height=0.55, capsize=2)
    ax.set_yticks(ypos); ax.set_yticklabels([labels[f] for f in s.index])
    ax.set_xlabel("drop in test $R^2$ when shuffled"); ax.set_title(nm)
    ax.grid(axis="y", alpha=0)
fig.tight_layout(); fig.savefig(OUT + "fig6.png"); plt.close(fig)

# ---------------- Fig 7: residuals by alpha ----------------
res = pd.read_csv(DATA + "rf_test_residuals.csv")
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
for ax, tgt, nm in zip(axes, ["Cl", "Cd"], ["$C_l$", "$C_d$"]):
    s = res[res.target == tgt]; al = sorted(s.alpha.unique())
    ax.boxplot([s[s.alpha == a].resid.values for a in al], positions=al, widths=1.2,
               flierprops=dict(marker=".", markersize=3, markeredgecolor="0.4"),
               medianprops=dict(color=C_RF), boxprops=dict(color="0.3"),
               whiskerprops=dict(color="0.3"), capprops=dict(color="0.3"))
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xticks(al); ax.set_xticklabels([str(a) for a in al])
    ax.set_xlabel("$\\alpha$ (deg)"); ax.set_ylabel(f"residual (predicted $-$ true) {nm}")
    ax.set_title(f"Random forest residuals, {nm}")
fig.tight_layout(); fig.savefig(OUT + "fig7.png"); plt.close(fig)

# ---------------- Fig 8: two-stage Cd model, alpha sweep ----------------
fig, ax = plt.subplots(figsize=(4.6, 3.3))
ax.axvspan(*TRAIN_A, color="0.9", label="training range of $\\alpha$")
ax.plot(alphas, tru[:, 1], color=C_TRUE, label="physics model")
ax.plot(alphas, rf_cd.predict(Xs), color=C_RF, ls="--", label="direct random forest")
ax.plot(alphas, lr_cd.predict(Xs), color=C_LR, ls="-.", label="direct linear regression")
ax.plot(alphas, stage2.predict(zf(lr_cl.predict(Xs), Xs[:, 2])), color=C_TS, ls=":", lw=2.2,
        label="two-stage (linear $C_l$, then quadratic)")
ax.set_xlabel("angle of attack $\\alpha$ (deg)"); ax.set_ylabel("$C_d$")
ax.set_title("NACA 2412: $C_d$ versus $\\alpha$"); ax.legend(frameon=False, fontsize=7, loc="upper center")
fig.tight_layout(); fig.savefig(OUT + "fig8.png"); plt.close(fig)
print("figures written to", OUT)
