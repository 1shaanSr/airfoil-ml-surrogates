"""
Re-runs every physics check quoted in the paper's Section 2.2 against the
CURRENT code, and asserts the paper's stated thresholds.  Run:  python3 verify_physics_claims.py
"""
import numpy as np
from scipy.integrate import quad
from naca_geometry import naca4_coordinates, camber_line
from panel_method import solve_panel_method, _panel_geometry, _influence_integrals

fails = []
def check(name, cond, detail):
    print(("PASS  " if cond else "FAIL  ") + f"{name}: {detail}")
    if not cond: fails.append(name)

# 1. symmetry: NACA 0012 at 0 deg has Cl ~ 0   (paper: about 5e-17)
x, y = naca4_coordinates(0, 0, 12, 160)
cl0 = solve_panel_method(x, y, 0.0)["Cl"]
check("symmetry Cl(0012, 0 deg)", abs(cl0) < 1e-12, f"{cl0:.2e}")

# 2. influence coefficients vs brute-force numerical integration of the point-source AND point-vortex fields
#    (paper: max abs diff ~9e-16, max rel diff ~1.5e-13 over 120 random pairs; relative error uses a denominator floored at 1e-3)
rng = np.random.RandomState(1)
w_abs = w_rel = 0.0; npairs = 0
for (m, p, t) in [(2, 4, 12), (0, 0, 12), (6, 3, 9)]:
    x, y = naca4_coordinates(m, p, t, 40)
    xc, yc, S, th = _panel_geometry(x, y)
    I, J = _influence_integrals(x, y, xc, yc, S, th)
    n = len(xc)
    for _ in range(40):
        i, j = rng.randint(0, n), rng.randint(0, n)
        if i == j: continue
        npairs += 1
        def integ(f):
            def g(s_):
                xs, ys = x[j] + s_ * np.cos(th[j]), y[j] + s_ * np.sin(th[j])
                dx, dy = xc[i] - xs, yc[i] - ys
                return f(dx, dy) / (dx * dx + dy * dy)
            return quad(g, 0, S[j], limit=300)[0]
        us, vs = integ(lambda a, b: a), integ(lambda a, b: b)        # unit source panel
        uv, vv = integ(lambda a, b: -b), integ(lambda a, b: a)       # unit vortex panel (point-vortex field)
        ni = (np.sin(th[i]), -np.cos(th[i])); ti = (np.cos(th[i]), np.sin(th[i]))
        pairs_ = [(I[i, j], us * ni[0] + vs * ni[1]), (J[i, j], us * ti[0] + vs * ti[1]),   # source normal / tangential
                  (-J[i, j], uv * ni[0] + vv * ni[1]), (I[i, j], uv * ti[0] + vv * ti[1])]  # vortex normal / tangential
        for a_, b_ in pairs_:
            w_abs = max(w_abs, abs(a_ - b_)); w_rel = max(w_rel, abs(a_ - b_) / max(abs(b_), 1e-3))
check(f"source+vortex influence coefficients vs numerical integration ({npairs} pairs)",
      w_abs < 2e-15 and w_rel < 1e-12, f"max abs diff {w_abs:.1e}, max rel diff {w_rel:.1e}")

# 3. zero-lift angle: thin-airfoil theory vs panel method (Table 1)
def a_thin(m_pct, p_t):
    m, p = m_pct / 100, p_t / 10
    def f(th):
        d = camber_line(np.array([(1 - np.cos(th)) / 2]), m, p)[1][0]
        return d * (np.cos(th) - 1)
    return np.degrees(-quad(f, 0, np.pi, limit=200)[0] / np.pi)
def a_panel(m, p, t):
    al = np.array([-6, -4, -2, 0, 2]); c = [solve_panel_method(*naca4_coordinates(m, p, t, 160), a)["Cl"] for a in al]
    f = np.polyfit(al, c, 1); return -f[1] / f[0]
for (m, p, t), (th_ref, pn_ref) in {(2, 4, 12): (-2.077, -2.145), (4, 4, 12): (-4.154, -4.276),
                                    (6, 4, 12): (-6.232, -6.386), (2, 4, 6): (-2.077, -2.102)}.items():
    a1, a2 = a_thin(m, p), a_panel(m, p, t)
    check(f"alpha_L0 NACA {m}{p}{t:02d}", abs(a1 - th_ref) < 6e-4 and abs(a2 - pn_ref) < 6e-4 and abs(a2 - a1) / abs(a1) < 0.04,
          f"thin {a1:+.3f}, panel {a2:+.3f} (Table 1: {th_ref:+.3f}, {pn_ref:+.3f})")

# 4. lift from pressure integration vs circulation; pressure drag ~ 0 (Table 2)
maxdl, maxd, mincl_rel = 0.0, 0.0, 1.0
for (m, p, t, a) in [(0, 0, 12, 0), (0, 0, 12, 4), (2, 4, 12, 0), (2, 4, 12, 4), (4, 4, 12, 8), (4, 4, 15, -4), (6, 4, 9, 6)]:
    x, y = naca4_coordinates(m, p, t, 160); r = solve_panel_method(x, y, a)
    th_, S_, Cp = r["theta"], r["S"], r["Cp"]
    Cx = -np.sum(Cp * np.sin(th_) * S_); Cy = np.sum(Cp * np.cos(th_) * S_); al = np.radians(a)
    L, D = Cy * np.cos(al) - Cx * np.sin(al), Cx * np.cos(al) + Cy * np.sin(al)
    maxdl = max(maxdl, abs(L - r["Cl"])); maxd = max(maxd, abs(D))
    if r["Cl"] > 0.7: mincl_rel = min(mincl_rel, 0) ; mincl_rel = max(mincl_rel, abs(L - r["Cl"]) / r["Cl"])
check("max |Cl(pressure) - Cl(circulation)| <= 0.013", maxdl <= 0.0131, f"{maxdl:.4f}")
check("relative difference < 1% when Cl > 0.7", mincl_rel < 0.01, f"{100*mincl_rel:.2f}%")
check("pressure drag magnitude < 0.001", maxd < 0.001, f"{maxd:.5f}")
x, y = naca4_coordinates(2, 4, 12, 160)
check("max Cp = 0.989 (NACA 2412, 4 deg)", abs(solve_panel_method(x, y, 4.0)["Cp"].max() - 0.989) < 5e-4, f"{solve_panel_method(x, y, 4.0)['Cp'].max():.3f}")

# 5. lift-curve slope numbers quoted in the text
al = np.array([-4, -2, 0, 2, 4, 6]); c = [solve_panel_method(*naca4_coordinates(0, 0, 12, 160), a)["Cl"] for a in al]
s = np.polyfit(al, c, 1)[0] * 180 / np.pi
check("NACA 0012 slope 6.86/rad, 9.2% above 2*pi", abs(s - 6.86) < 0.005 and abs((s / (2 * np.pi) - 1) * 100 - 9.2) < 0.05, f"{s:.4f}, {(s/(2*np.pi)-1)*100:.2f}%")

print("\nALL PHYSICS CLAIMS VERIFIED" if not fails else f"\nFAILED: {fails}")
