"""
Generate the training/test dataset by running the (now validated) panel
method + drag model over a grid of NACA 4-digit shape parameters and angles
of attack.

Two datasets are produced, deliberately:

1. main_dataset.csv -- the "normal operating range" grid, used for the
   standard train/test split (interpolation performance).

2. extrapolation_dataset.csv -- points deliberately OUTSIDE the main grid's
   parameter ranges (higher angle of attack, higher camber, higher
   thickness), used later specifically to show where the ML surrogate's
   predictions degrade under extrapolation. This is evaluated separately
   and is never mixed into training.

IMPORTANT SCOPE NOTE (stated here and repeated in the paper): the "ground
truth" for this whole project is OUR OWN panel method + empirical drag
model, not experiment or CFD. Neither model has any representation of
stall (the panel method is inviscid/potential flow; the drag model is a
smooth parabola with no separation mechanism). So "where the ML model
fails at extreme angles of attack" in this project means: where the ML
surrogate fails to reproduce OUR physics model outside its training range
-- a statement about ML interpolation/extrapolation, not about real-world
stall aerodynamics. This distinction matters and must not get blurred.
"""
import numpy as np
import pandas as pd
from naca_geometry import naca4_coordinates
from panel_method import solve_panel_method, _panel_geometry
from drag_model import profile_drag_coefficient


def compute_case(m, p, t, alpha, n_panels=160):
    x, y = naca4_coordinates(m, p, t, n_panels=n_panels)
    res = solve_panel_method(x, y, alpha)
    _, _, S, _ = _panel_geometry(x, y)
    wetted = np.sum(S)
    cd = profile_drag_coefficient(t, wetted, res["Cl"])
    return res["Cl"], cd


def build_grid(m_vals, p_vals, t_vals, alpha_vals):
    rows = []
    for m in m_vals:
        p_list = [0] if m == 0 else p_vals
        for p in p_list:
            for t in t_vals:
                for a in alpha_vals:
                    cl, cd = compute_case(m, p, t, a)
                    rows.append(dict(m=m, p=p, t=t, alpha=a, Cl=cl, Cd=cd))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # --- Main dataset: the "normal operating range" ---
    m_vals = [0, 1, 2, 3, 4, 5, 6]
    p_vals = [2, 3, 4, 5, 6]
    t_vals = [6, 8, 10, 12, 15, 18]
    alpha_vals = [-6, -4, -2, 0, 2, 4, 6, 8, 10]

    df_main = build_grid(m_vals, p_vals, t_vals, alpha_vals)
    print(f"Main dataset: {len(df_main)} rows")
    print(df_main.describe())
    df_main.to_csv("../data/main_dataset.csv", index=False)

    # --- Extrapolation dataset: deliberately outside the main ranges ---
    rows = []
    # (a) angle-of-attack extrapolation, at a handful of representative shapes
    shapes = [(0, 0, 12), (2, 4, 12), (4, 4, 10), (2, 4, 18)]
    extreme_alphas = [-12, -10, -8, 12, 14, 16, 18, 20]
    for (m, p, t) in shapes:
        for a in extreme_alphas:
            cl, cd = compute_case(m, p, t, a)
            rows.append(dict(m=m, p=p, t=t, alpha=a, Cl=cl, Cd=cd, extrap_type="alpha"))
    # (b) shape extrapolation: camber/thickness beyond the training range,
    #     at ordinary (non-extreme) angles of attack
    extreme_shapes = [(8, 4, 12), (6, 8, 12), (2, 4, 21), (2, 4, 4)]
    ordinary_alphas = [-4, 0, 4, 8]
    for (m, p, t) in extreme_shapes:
        for a in ordinary_alphas:
            cl, cd = compute_case(m, p, t, a)
            rows.append(dict(m=m, p=p, t=t, alpha=a, Cl=cl, Cd=cd, extrap_type="shape"))

    df_extrap = pd.DataFrame(rows)
    print(f"\nExtrapolation dataset: {len(df_extrap)} rows")
    df_extrap.to_csv("../data/extrapolation_dataset.csv", index=False)

    print("\nSaved: data/main_dataset.csv, data/extrapolation_dataset.csv")
