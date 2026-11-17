"""
Simple profile-drag model for a 2D airfoil section, at a fixed Reynolds
number. This is NOT a viscous CFD solve -- it is the classical "component
buildup" / equivalent-skin-friction method used for preliminary aircraft
design (see e.g. Raymer, "Aircraft Design: A Conceptual Approach", or
Anderson, "Fundamentals of Aerodynamics", drag-polar chapter). It is a
standard, well-documented engineering approximation, explicitly not a
substitute for a real Navier-Stokes/boundary-layer solution.

Model:
  Cd = Cd0(t) + k * Cl^2

  Cd0(t) = Cf * FF(t) * (Swet / c)
    Cf      : flat-plate turbulent skin-friction coefficient (Prandtl
              1/7-power-law formula), at a fixed Reynolds number.
    FF(t)   : thickness-dependent form factor (Hoerner's classical formula),
              which is the standard way this method accounts for how a
              thicker section has more pressure drag than a flat plate of
              the same wetted length.
    Swet/c  : wetted perimeter of the actual airfoil shape, per unit chord
              -- computed directly from the panel geometry, so this term
              is shape-specific, not a constant.

  k * Cl^2 : the classical parabolic drag-polar term (profile drag rises
             roughly with the square of the lift coefficient away from
             the drag-minimum condition). k is fixed at a typical published
             value for smooth, low-speed airfoil sections, not fit to any
             data used later in this project.

Known limitations (stated here, and again in the paper):
  - Fixed Reynolds number: no Re dependence is modeled.
  - No stall: this model has no mechanism for the real drag rise from flow
    separation at high angle of attack -- it will underpredict Cd there.
  - The drag-polar minimum is assumed at Cl=0; real cambered airfoils have
    their minimum-drag point at a small positive Cl (near the design Cl).
"""
import numpy as np

RE_FIXED = 1.0e6  # fixed Reynolds number for the whole dataset (typical small-UAV/wind-tunnel value)
K_POLAR = 0.01     # typical published parabolic drag-polar curvature for smooth low-speed airfoils


def flat_plate_cf(re):
    """Prandtl 1/7-power-law turbulent flat-plate skin-friction coefficient."""
    return 0.074 / re ** 0.2


def form_factor(t_pct):
    """Hoerner's classical thickness form factor, t_pct = thickness in percent chord."""
    tc = t_pct / 100.0
    return 1 + 2.7 * tc + 100 * tc ** 4


def profile_drag_coefficient(t_pct, wetted_length, cl, re=RE_FIXED, k=K_POLAR):
    """
    t_pct: thickness in percent chord (e.g. 12 for NACA xx12)
    wetted_length: total airfoil perimeter per unit chord (sum of panel
                   lengths from the panel-method geometry -- chord is 1,
                   so this is already "per unit chord")
    cl: lift coefficient (from the panel method)
    """
    cf = flat_plate_cf(re)
    ff = form_factor(t_pct)
    cd0 = cf * ff * wetted_length
    return cd0 + k * cl ** 2


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from naca_geometry import naca4_coordinates
    from panel_method import solve_panel_method, _panel_geometry

    print(f"Flat-plate Cf at Re={RE_FIXED:.0e}: {flat_plate_cf(RE_FIXED):.5f}")
    print()
    print("Sanity check: Cd0 (at Cl=0, alpha such that Cl~0) for a few thicknesses")
    print("(published smooth-airfoil Cd0 at this Reynolds number is typically ~0.004-0.01)")
    for t in [6, 9, 12, 15, 18]:
        x, y = naca4_coordinates(0, 0, t, n_panels=160)
        _, _, S, _ = _panel_geometry(x, y)
        wetted = np.sum(S)
        cd0 = profile_drag_coefficient(t, wetted, cl=0.0)
        print(f"  t={t:2d}%   wetted length/c={wetted:.4f}   form factor={form_factor(t):.4f}   Cd0={cd0:.5f}")

    print()
    print("Drag polar shape check: NACA 2412, Cd vs alpha (should be U-shaped in Cl, smooth)")
    for a in [-4, -2, 0, 2, 4, 6, 8]:
        x, y = naca4_coordinates(2, 4, 12, n_panels=160)
        res = solve_panel_method(x, y, a)
        _, _, S, _ = _panel_geometry(x, y)
        wetted = np.sum(S)
        cd = profile_drag_coefficient(12, wetted, res["Cl"])
        print(f"  alpha={a:+3d} deg   Cl={res['Cl']:+.4f}   Cd={cd:.5f}   L/D={res['Cl']/cd:+.2f}")
