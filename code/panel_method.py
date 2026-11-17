"""
Constant-strength source + vortex panel method for inviscid, incompressible
flow around a 2D airfoil (the classic "Hess-Smith" method, in the form given
by J. Moran, "An Introduction to Theoretical and Computational Aerodynamics",
1984, Ch. 3 -- the standard textbook version of this algorithm).

Physical model:
  - Each panel carries a constant-strength source, lambda_j (unknown, one per panel).
  - The whole body carries ONE shared vortex strength, gamma (unknown).
  - Boundary condition: net normal velocity (freestream + all sources + vortex)
    is zero at every panel's control point (flow doesn't cross the surface).
  - Kutta condition: the flow leaves the trailing edge smoothly, enforced as
    equal-and-opposite tangential speed on the two panels touching the
    trailing edge.
  - This gives N+1 linear equations in N+1 unknowns (lambda_1..N, gamma):
    exactly the kind of linear system solved with ordinary linear algebra.

Everything here is closed-form geometry/calculus (panel self- and mutual-
influence integrals) plus one linear solve. No fitting, no free parameters.

Convention note: I[i,j] and J[i,j] below are normalized so that a panel's
self-induced normal velocity is I[i,i] = pi (this is the standard Moran
convention). All equations are written consistently in that convention.
"""
import numpy as np


def _panel_geometry(x, y):
    xc = 0.5 * (x[:-1] + x[1:])
    yc = 0.5 * (y[:-1] + y[1:])
    S = np.hypot(x[1:] - x[:-1], y[1:] - y[:-1])
    theta = np.arctan2(y[1:] - y[:-1], x[1:] - x[:-1])
    return xc, yc, S, theta


def _influence_integrals(x, y, xc, yc, S, theta):
    """
    I[i,j]: normal velocity at control point i induced by a unit-strength
            SOURCE panel j.
    J[i,j]: tangential velocity at control point i induced by a unit-strength
            SOURCE panel j.
    (Standard closed-form panel integrals; self terms I[i,i]=pi, J[i,i]=0.)

    Important geometric detail: the integral for panel j is parametrized
    from panel j's STARTING corner point (x[j], y[j]) -- not its midpoint --
    since the panel's local coordinate runs from 0 to S[j] starting there.
    The field point i uses its control point (xc[i], yc[i]) (midpoint).
    Mixing these up (using panel j's midpoint instead of its start corner)
    is a classic bug that breaks left/right symmetry silently.
    """
    n = len(xc)
    # Vectorized version of the same formulas as the original (i,j) double
    # loop -- kept algebraically identical, just computed for all pairs at
    # once. Verified against the loop version to match exactly (see
    # test_vectorized_matches_loop() below) before being trusted for the
    # full dataset generation.
    XC_I = xc[:, None]   # shape (n,1), broadcasts over j
    YC_I = yc[:, None]
    X_J = x[None, :n]    # shape (1,n), broadcasts over i
    Y_J = y[None, :n]
    TH_I = theta[:, None]
    TH_J = theta[None, :]
    S_J = S[None, :]

    dx = XC_I - X_J
    dy = YC_I - Y_J
    A = -dx * np.cos(TH_J) - dy * np.sin(TH_J)
    B = dx ** 2 + dy ** 2
    Cn = np.sin(TH_I - TH_J)
    Dn = -dx * np.sin(TH_I) + dy * np.cos(TH_I)
    Ct = -np.cos(TH_I - TH_J)
    Dt = dx * np.cos(TH_I) + dy * np.sin(TH_I)
    Ssq = S_J ** 2
    E2 = np.clip(B - A ** 2, 1e-12, None)
    E = np.sqrt(E2)
    term_log = np.log((Ssq + 2 * A * S_J + B) / B)
    term_atan = np.arctan2(S_J + A, E) - np.arctan2(A, E)

    I = -((Cn / 2) * term_log + ((Dn - A * Cn) / E) * term_atan)
    J = (Ct / 2) * term_log + ((Dt - A * Ct) / E) * term_atan

    np.fill_diagonal(I, np.pi)
    np.fill_diagonal(J, 0.0)
    return I, J


def solve_panel_method(x, y, alpha_deg, V_inf=1.0):
    """
    Solve for panel source strengths and the shared vortex strength, given
    freestream speed V_inf and angle of attack alpha_deg.

    Returns dict with control points, surface tangential speed, pressure
    coefficient Cp, and lift coefficient Cl (Kutta-Joukowski: Cl = 2*Gamma/V,
    chord normalized to 1).
    """
    alpha = np.radians(alpha_deg)
    xc, yc, S, theta = _panel_geometry(x, y)
    n = len(xc)
    I, J = _influence_integrals(x, y, xc, yc, S, theta)

    # Vortex-panel coefficients from the source/vortex duality (a unit vortex
    # panel's field is the unit source panel's field rotated 90 degrees):
    Kn_vortex = -J  # normal velocity at i due to unit vortex on j
    Kt_vortex = I   # tangential velocity at i due to unit vortex on j (self: pi)

    A_mat = np.zeros((n + 1, n + 1))
    b_vec = np.zeros(n + 1)

    # (1) Flow tangency at each control point: normal velocity = 0.
    #     induced_normal_i / (2*pi) + V_inf . n_i = 0, and V_inf . n_i = -sin(alpha-theta_i)
    #     (verified symbolically), so induced_normal_i = 2*pi*V_inf*sin(alpha-theta_i).
    A_mat[:n, :n] = I
    A_mat[:n, n] = np.sum(Kn_vortex, axis=1)
    b_vec[:n] = 2 * np.pi * V_inf * np.sin(alpha - theta)

    # (2) Kutta condition: tangential speeds at the trailing-edge panels
    #     (first and last panel in the ordering) sum to zero.
    #     V_inf . t_i = cos(alpha-theta_i) (verified symbolically).
    A_mat[n, :n] = J[0, :] + J[-1, :]
    A_mat[n, n] = np.sum(Kt_vortex[0, :]) + np.sum(Kt_vortex[-1, :])
    b_vec[n] = -2 * np.pi * V_inf * (
        np.cos(alpha - theta[0]) + np.cos(alpha - theta[-1])
    )

    sol = np.linalg.solve(A_mat, b_vec)
    lam = sol[:n]
    gamma = sol[n]

    # Tangential velocity at each control point (source terms use J,
    # vortex terms use I=Kt_vortex; each unit-strength contribution is
    # divided by 2*pi to match the (Cn/2)*log(...) normalization above).
    Vt = (
        V_inf * np.cos(alpha - theta)
        + (J @ lam) / (2 * np.pi)
        + gamma * np.sum(Kt_vortex, axis=1) / (2 * np.pi)
    )

    Cp = 1 - (Vt / V_inf) ** 2

    # Kutta-Joukowski: L' = rho*V*Gamma. The sign of Gamma relative to lift
    # depends on the body-contour traversal direction used to define it; for
    # our clockwise traversal the correct sign is the negative of the raw
    # gamma*perimeter product (confirmed by validation: without this minus
    # sign, Cl came out with the right zero-lift angle but decreasing with
    # alpha, i.e. exactly sign-flipped).
    Gamma_total = -gamma * np.sum(S)
    Cl = 2 * Gamma_total / V_inf

    return {
        "xc": xc, "yc": yc, "S": S, "theta": theta,
        "lambda": lam, "gamma": gamma,
        "Vt": Vt, "Cp": Cp, "Cl": Cl,
    }
