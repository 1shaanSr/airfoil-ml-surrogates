"""
Check the panel method against an EXACT potential-flow solution: the symmetric
Karman-Trefftz airfoil (a conformal map of a circle, with a finite
trailing-edge angle).  See Appendix A of the paper for the derivation.

  (z - n b)/(z + n b) = ((zeta - b)/(zeta + b))^n ,   circle: centre -eps*b, radius R = b(1+eps)
  Kutta-Joukowski on the circle:  Gamma = 4 pi U R sin(alpha)
  => Cl = 2 Gamma/(U c) = 8 pi R sin(alpha)/c  => exact slope 8 pi R/c per radian,
     with the chord c measured from the mapped shape.
Outputs ../data/kt_exact_check.csv and asserts the accuracy claims in the paper.
"""
import numpy as np, pandas as pd
from panel_method import solve_panel_method


def kt_shape(eps, n, N):
    b = 1.0; R = b * (1 + eps); zc = -eps * b
    h = N // 2
    v = np.linspace(0, np.pi, h + 1); ph = np.pi * (1 - np.cos(v)) / 2      # clustered at TE and LE
    phi = np.concatenate([ph, 2 * np.pi - ph[::-1][1:]])                    # TE -> upper -> LE -> lower -> TE
    zeta = zc + R * np.exp(1j * phi)
    w = ((zeta - b) / (zeta + b)) ** n
    z = (n * b * (1 + w) / (1 - w))[:-1]                                    # drop duplicate closing point
    c = z.real.max() - z.real.min()
    return (z.real - z.real.min()) / c, z.imag / c, c, R


def exact_slope(eps, n):
    x, y, c, R = kt_shape(eps, n, 2000)
    return 8 * np.pi * R / c, y.max() - y.min()


def panel_slope(eps, n, N):
    x, y, _, _ = kt_shape(eps, n, N)
    al = np.array([-2.0, 0.0, 2.0]); cl = [solve_panel_method(x, y, a)["Cl"] for a in al]
    return np.polyfit(al, cl, 1)[0] * 180 / np.pi


# sanity: n -> 2 must reproduce the closed-form Joukowski slope 8 pi (1+eps)/(3+2eps+1/(1+2eps))
e0 = 0.1
assert abs(exact_slope(e0, 1.999999)[0] - 8 * np.pi * (1 + e0) / (3 + 2 * e0 + 1 / (1 + 2 * e0))) < 1e-3

rows = []
for n in [1.9, 1.8]:
    for eps in [0.05, 0.10, 0.15]:
        se, tc = exact_slope(eps, n)
        errs = {N: 100 * (panel_slope(eps, n, N) / se - 1) for N in (80, 160, 320, 640)}
        rows.append(dict(n=n, eps=eps, t_over_c=tc, exact=se, panel_160=panel_slope(eps, n, 160),
                         err_160_pct=errs[160], max_abs_err_pct=max(abs(v) for v in errs.values())))
r = pd.DataFrame(rows); r.to_csv("../data/kt_exact_check.csv", index=False)
print(r.round(4).to_string(index=False))
assert r.err_160_pct.abs().max() < 0.5 and r.max_abs_err_pct.max() < 1.3
print("PASS: |error| < 0.5% at 160 panels; < 1.3% over 80-640 panels")

# small-thickness Joukowski approximation 2*pi*(1 + 0.77 t/c), compared with our NACA 00tt slopes (Table 3)
for t, ours in [(12, 6.850), (6, 6.592)]:
    print(f"2pi(1+0.77*{t/100:.2f}) = {2*np.pi*(1+0.77*t/100):.3f}   vs NACA 00{t:02d} panel slope {ours:.3f}")
