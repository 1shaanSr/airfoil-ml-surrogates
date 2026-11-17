"""
Generate every table in the paper as a LaTeX tabular, straight from computed
values / saved CSVs, so no number is typed by hand.  Output: a dict written to
../data/tables_generated.json (name -> LaTeX tabular string).
"""
import json
import numpy as np
import pandas as pd
from scipy.integrate import quad
from naca_geometry import naca4_coordinates, camber_line
from panel_method import solve_panel_method

D = "../data/"


def cl_of(m, p, t, a, n=160):
    x, y = naca4_coordinates(m, p, t, n_panels=n)
    return solve_panel_method(x, y, a)["Cl"]


def tab(colspec, header, rows):
    out = [r"\begin{tabular}{@{}" + colspec + r"@{}}", r"\toprule", header + r" \\", r"\midrule"]
    out += [r + r" \\" for r in rows]
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


T = {}

# ---- zero-lift angle: thin-airfoil theory vs panel method ----
def alpha_L0_thin(m_pct, p_tenths):
    m, p = m_pct / 100, p_tenths / 10
    def f(th):
        x = np.array([(1 - np.cos(th)) / 2]); _, d = camber_line(x, m, p)
        return d[0] * (np.cos(th) - 1)
    return np.degrees(-quad(f, 0, np.pi, limit=200)[0] / np.pi)

def alpha_L0_panel(m, p, t):
    al = np.array([-6, -4, -2, 0, 2]); c = [cl_of(m, p, t, a) for a in al]
    f = np.polyfit(al, c, 1); return -f[1] / f[0]

rows = []
for (m, p, t) in [(2, 4, 12), (4, 4, 12), (6, 4, 12), (2, 4, 6)]:
    rows.append(f"NACA {m}{p}{t:02d} & ${alpha_L0_thin(m,p):+.3f}$ & ${alpha_L0_panel(m,p,t):+.3f}$".replace("+", "").replace("$-", "$-"))
T["alphaL0"] = tab("lcc", r"section & thin-airfoil theory & panel method", rows)

# ---- force check: circulation vs pressure integration ----
def force_check(m, p, t, a, n=160):
    x, y = naca4_coordinates(m, p, t, n_panels=n); r = solve_panel_method(x, y, a)
    th, S, Cp = r["theta"], r["S"], r["Cp"]
    Cx = -np.sum(Cp * np.sin(th) * S); Cy = np.sum(Cp * np.cos(th) * S); al = np.radians(a)
    return r["Cl"], Cy * np.cos(al) - Cx * np.sin(al), Cx * np.cos(al) + Cy * np.sin(al)

rows = []
for (m, p, t, a) in [(0, 0, 12, 0), (0, 0, 12, 4), (2, 4, 12, 0), (2, 4, 12, 4),
                     (4, 4, 12, 8), (4, 4, 15, -4), (6, 4, 9, 6)]:
    c1, c2, dd = force_check(m, p, t, a)
    rows.append(f"NACA {m}{p}{t:02d} & ${a}^\\circ$ & ${c1:.4f}$ & ${c2:.4f}$ & ${dd:+.5f}$")
T["forcecheck"] = tab("lcccc", r"section & $\alpha$ & $C_l$ (circulation) & $C_l$ (pressure integral) & pressure drag", rows)

# ---- thickness trend & panel convergence ----
rows = []
for t in [12, 6, 3, 1]:
    a3 = np.array([-1, 0, 1]); c3 = [cl_of(0, 0, t, a, 200) for a in a3]
    rows.append(f"{t} & ${np.polyfit(a3, c3, 1)[0]*180/np.pi:.3f}$")
T["thin"] = tab("cc", r"thickness $t$ (\% chord) & slope (per rad)", rows)
rows = []
for n in [40, 80, 160, 320]:
    a4 = np.array([-2, 0, 2]); c4 = [cl_of(0, 0, 12, a, n) for a in a4]
    rows.append(f"{n} & ${np.polyfit(a4, c4, 1)[0]*180/np.pi:.3f}$")
T["conv"] = tab("cc", r"panels & slope (per rad)", rows)

# ---- repeated splits ----
s = pd.read_csv(D + "repeated_splits_summary.csv")
def g(proto, tgt, mdl, col): return s[(s.protocol == proto) & (s.target == tgt) & (s.model == mdl)][col].values[0]
rows = []
for tgt, nm in [("Cl", "$C_l$"), ("Cd", "$C_d$")]:
    for mdl in ["linear regression", "random forest"]:
        cells = []
        for proto in ["random rows", "held-out shapes"]:
            cells.append(f"${g(proto,tgt,mdl,'r2_mean'):.4f}\\pm{g(proto,tgt,mdl,'r2_std'):.4f}$")
        for proto in ["random rows", "held-out shapes"]:
            cells.append(f"${100*g(proto,tgt,mdl,'rel_mean'):.2f}\\pm{100*g(proto,tgt,mdl,'rel_std'):.2f}$")
        rows.append(f"{nm} & {mdl} & " + " & ".join(cells))
T["repeated"] = tab("llcccc", r"target & model & $R^2$ (rows) & $R^2$ (shapes) & error \% (rows) & error \% (shapes)", rows)

# ---- learning curve, selected sizes ----
lc = pd.read_csv(D + "learning_curve_summary.csv")
rows = []
for n in [50, 100, 200, 400, 800, 1339]:
    cells = []
    for tgt in ["Cl", "Cd"]:
        for mdl in ["linear regression", "random forest"]:
            r = lc[(lc.target == tgt) & (lc.model == mdl) & (lc.n_train == n)].iloc[0]
            cells.append(f"${100*r.rel_mean:.2f}\\pm{100*r.rel_std:.2f}$")
    rows.append(f"{n} & " + " & ".join(cells))
T["learning"] = tab("ccccc", r"training rows & $C_l$ linear & $C_l$ forest & $C_d$ linear & $C_d$ forest", rows)

# ---- residuals by alpha: MAE, mean, median ----
rr = pd.read_csv(D + "rf_test_residuals.csv")
g_ = rr.groupby(["target", "alpha"]).resid
st = pd.DataFrame({"mae": g_.apply(lambda v: np.mean(np.abs(v))), "mean": g_.mean(), "median": g_.median()}).reset_index()
rows = []
for a_ in sorted(st.alpha.unique()):
    c = st[(st.target == "Cl") & (st.alpha == a_)].iloc[0]; d_ = st[(st.target == "Cd") & (st.alpha == a_)].iloc[0]
    rows.append(f"${int(a_)}$ & ${c.mae:.4f}$ & ${c['mean']:+.4f}$ & ${c['median']:+.4f}$ & ${d_.mae:.5f}$ & ${d_['mean']:+.5f}$")
T["maealpha"] = tab("cccccc", r"$\alpha$ (deg) & $C_l$ MAE & $C_l$ mean & $C_l$ median & $C_d$ MAE & $C_d$ mean", rows)

# ---- reliability: does the forest warn when extrapolating? ----
rl = pd.read_csv(D + "reliability_results.csv")
rows = []
GL = {"in-range test split": "in-range test split", "extrapolation (alpha)": "extrapolation, $\\alpha$", "extrapolation (shape)": "extrapolation, shape"}
for tgt, nm in [("Cl", "$C_l$"), ("Cd", "$C_d$")]:
    for grp in GL:
        q = rl[(rl.target == tgt) & (rl.group == grp)].iloc[0]
        rows.append(f"{nm} & {GL[grp]} & {int(q.n)} & ${q.mean_tree_spread:.5f}$ & ${q.mean_abs_error:.5f}$ & {100*q.frac_above_inrange_p95:.0f}\\% & {100*q.frac_flagged_by_range_check:.0f}\\%")
T["reliability"] = tab("llccccc", r"target & query set & $n$ & tree spread & abs.\ error & spread $>$ p95 & range check", rows)

# ---- exact-solution check (Karman-Trefftz) ----
kt = pd.read_csv(D + "kt_exact_check.csv")
rows = [f"${q.n:.1f}$ & ${q.eps:.2f}$ & ${q.t_over_c:.3f}$ & ${q.exact:.4f}$ & ${q.panel_160:.4f}$ & ${q.err_160_pct:+.2f}$ & ${q.max_abs_err_pct:.2f}$" for q in kt.itertuples()]
T["kt"] = tab("ccccccc", r"$n$ & $\varepsilon$ & $t/c$ & exact slope & panel slope (160) & error (\%) at 160 & max $|$error$|$ (\%), 80--640", rows)

# ---- extrapolation (seed-42 models) ----
r = pd.read_csv(D + "surrogate_results.csv")
def gr(tgt, split, label):
    q = r[(r.target == tgt) & (r.split == split) & (r.label == label)].iloc[0]; return q.r2, q.relative_error
rows = []
for tgt, nm in [("Cl", "$C_l$"), ("Cd", "$C_d$")]:
    for split, sl, lab in [("extrapolation", "all 48 points", "linear regression"),
                           ("extrapolation", "all 48 points", "random forest"),
                           ("extrapolation_alpha", "angle of attack only (32)", None),
                           ("extrapolation_shape", "shape only (16)", None)]:
        if lab is None:
            r2, re = gr(tgt, split, r[(r.target == tgt) & (r.split == split)].label.values[0]); mod = "random forest"
        else:
            r2, re = gr(tgt, split, lab); mod = lab
        r2s = f"${r2:.4f}$".replace("$-", "$-")
        rows.append(f"{nm} & {mod} & {sl} & {r2s} & {100*re:.2f}\\%")
T["extrap"] = tab("lllcc", r"target & model & subset & $R^2$ & rel.\ error", rows)

# ---- two-stage ----
f = pd.read_csv(D + "two_stage_results.csv")
sets = ["test split", "extrap: all", "extrap: alpha only", "extrap: shape only"]
rows = []
for mdl in ["direct linear regression", "direct random forest",
            "two-stage (linear Cl -> quadratic)", "two-stage (forest Cl -> quadratic)"]:
    cells = []
    for st in sets:
        q = f[(f.set == st) & (f.model == mdl)].iloc[0]
        cells.append(f"${q.r2:.4f}$ ({100*q.rel:.1f}\\%)")
    nm = (mdl.replace("Cl -> quadratic", "$C_l$, then quadratic")
             .replace("(linear ", "(linear ").replace("(forest ", "(forest "))
    rows.append(nm + " & " + " & ".join(cells))
T["twostage"] = tab("lcccc", r"$C_d$ model & test split & extrap.\ all & extrap.\ $\alpha$ only & extrap.\ shape only", rows)

json.dump(T, open(D + "tables_generated.json", "w"), indent=1)
for k, v in T.items():
    print("-----", k); print(v)
