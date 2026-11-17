"""
Scripts behind two claims in the paper:
  (a) open vs closed trailing edge:  slope changes by +1.1%, alpha_L0(2412) by 0.005 deg; TE gap 0.0025c at t=12%.
  (b) linear regression predicts Cd < 0 for NACA 2412 below alpha ~ -10.3 deg, and for 5 of 32 alpha-extrapolation points.
"""
import numpy as np, pandas as pd
import naca_geometry as ng
from panel_method import solve_panel_method
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def slopes(coef):
    orig = ng.thickness
    ng.thickness = lambda x, t: 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - coef * x**4)
    try:
        al = np.array([-4, -2, 0, 2, 4, 6]); c = [solve_panel_method(*ng.naca4_coordinates(0, 0, 12, 160), a)["Cl"] for a in al]
        s = np.polyfit(al, c, 1)[0] * 180 / np.pi
        al2 = np.array([-6, -4, -2, 0, 2]); c2 = [solve_panel_method(*ng.naca4_coordinates(2, 4, 12, 160), a)["Cl"] for a in al2]
        f = np.polyfit(al2, c2, 1)
    finally:
        ng.thickness = orig
    return s, -f[1] / f[0]

so, ao = slopes(0.1015); sc, ac = slopes(0.1036)
x, y = ng.naca4_coordinates(0, 0, 12, 160)
gap = abs(y[0] - y[-1])
print(f"TE gap (NACA 0012) = {gap:.5f} c;  slope open {so:.4f} -> closed {sc:.4f} ({100*(sc/so-1):+.2f}%);  alpha_L0(2412) {ao:.3f} -> {ac:.3f} (diff {abs(ac-ao):.3f} deg)")
assert abs(gap - 0.0025) < 5e-5 and abs(100 * (sc / so - 1) - 1.1) < 0.05 and abs(abs(ac - ao) - 0.005) < 5e-4

df = pd.read_csv("../data/main_dataset.csv"); ex = pd.read_csv("../data/extrapolation_dataset.csv")
X = df[["m", "p", "t", "alpha"]].values
tr, _ = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)
lr = LinearRegression().fit(X[tr], df.Cd.values[tr])
al = np.linspace(-30, 0, 3001)
pred = lr.predict(np.column_stack([np.full_like(al, 2), np.full_like(al, 4), np.full_like(al, 12), al]))
cross = al[np.where(pred < 0)[0].max()]
neg = int(np.sum(lr.predict(ex[ex.extrap_type == "alpha"][["m", "p", "t", "alpha"]].values) < 0))
print(f"LR Cd<0 for NACA 2412 below alpha = {cross:.2f} deg;  negative predictions on alpha-extrapolation set: {neg} of 32")
assert abs(cross - (-10.3)) < 0.05 and neg == 5
print("PASS")
