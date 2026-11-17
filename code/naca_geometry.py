"""
NACA 4-digit airfoil geometry.

A NACA 4-digit airfoil is specified by three numbers, e.g. NACA 2412:
  m = 2  -> max camber = 2% of chord (first digit / 100)
  p = 4  -> location of max camber = 40% of chord (second digit / 10)
  t = 12 -> max thickness = 12% of chord (last two digits / 100)

All formulas below are the standard closed-form NACA 4-digit equations
(see e.g. Abbott & Von Doenhoff, "Theory of Wing Sections", 1959).
Nothing here is fit to data -- it's the analytic definition of the shape.
"""
import numpy as np


def camber_line(x, m, p):
    """Mean camber line y_c(x) and its slope dy_c/dx, for x in [0,1] (chord units)."""
    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)
    if m == 0 or p == 0:
        return yc, dyc_dx  # symmetric airfoil: camber line is flat
    front = x < p
    back = ~front
    yc[front] = (m / p**2) * (2 * p * x[front] - x[front]**2)
    dyc_dx[front] = (2 * m / p**2) * (p - x[front])
    yc[back] = (m / (1 - p)**2) * ((1 - 2 * p) + 2 * p * x[back] - x[back]**2)
    dyc_dx[back] = (2 * m / (1 - p)**2) * (p - x[back])
    return yc, dyc_dx


def thickness(x, t):
    """Symmetric thickness distribution y_t(x), half-thickness perpendicular to chord."""
    return 5 * t * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )


def naca4_coordinates(m_pct, p_tenths, t_pct, n_panels=160):
    """
    Build closed-loop panel coordinates for a NACA 4-digit airfoil.

    m_pct: max camber in percent chord (e.g. 2 for NACA 2412)
    p_tenths: camber position in tenths of chord (e.g. 4 for NACA 2412)
    t_pct: max thickness in percent chord (e.g. 12 for NACA 2412)
    n_panels: number of panels on the full closed body (upper+lower)

    Returns x, y arrays of length n_panels+1, tracing the airfoil surface
    starting at the trailing edge, going over the top, around the leading
    edge, along the bottom, and back to the trailing edge (closed loop).
    This point ordering is what the panel method in panel_method.py assumes.
    """
    m = m_pct / 100.0
    p = p_tenths / 10.0
    t = t_pct / 100.0

    n_side = n_panels // 2 + 1
    # cosine spacing clusters points near the leading/trailing edges,
    # where curvature is highest -- standard practice, not a free choice.
    beta = np.linspace(0, np.pi, n_side)
    xc = (1 - np.cos(beta)) / 2

    yc, dyc_dx = camber_line(xc, m, p)
    yt = thickness(xc, t)
    theta = np.arctan(dyc_dx)

    xu = xc - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = xc + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    # trailing edge -> upper surface (reversed, TE to LE) -> lower surface (LE to TE)
    x = np.concatenate([xu[::-1], xl[1:]])
    y = np.concatenate([yu[::-1], yl[1:]])
    return x, y


if __name__ == "__main__":
    # quick sanity check: NACA 0012 should be symmetric (yc == 0 everywhere)
    x, y = naca4_coordinates(0, 0, 12, n_panels=40)
    print("NACA 0012 max |y| (should be close to 0.06, i.e. 6% half-thickness):", np.max(np.abs(y)))
    x2, y2 = naca4_coordinates(2, 4, 12, n_panels=40)
    print("NACA 2412 generated, n points:", len(x2))
