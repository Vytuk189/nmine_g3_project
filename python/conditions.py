"""Edit the initial profiles and fixed boundary states here for each scenario."""

import numpy as np
from solver_functions import primitive_to_q, q_to_primitive


def p_rule(x):
    if 4 < x < 6:
        return 500e5    
    else:
        return 10e5

def u_rule(x):
    return 0.0

def t_g_rule(x):
    return 293.15

def t_l_rule(x):
    return 293.15

def alpha_g_rule(x):
    return 0.1

def g3_initial_u(x):
    from domain_rules import g3_nozzle_area
    u0 = 5.0
    return g3_nozzle_area(0) * u0 / g3_nozzle_area(x)

def g3_initial_p(x):
    return 10e5

def g3_initial_t_g(x):
    return 293.15

def g3_initial_t_l(x):  
    return 293.15   

def g3_initial_alpha_g(x):
    return 0.9999

def g3_inlet_state():
    """Keep the inlet at the initial primitive values evaluated at x = 0."""
    x = 0.0
    return primitive_to_q(g3_initial_p(x), g3_initial_u(x), g3_initial_t_g(x), g3_initial_t_l(x), g3_initial_alpha_g(x))

def g3_outlet_pressure():
    """Prescribed outlet pressure [Pa]. Will be lowered for subsequent cases."""
    return 10e5


def fixed_state(side):
    """Returns the conservative state prescribed on a fixed boundary."""
    if side == "left":
        p, u, t_g, t_l, alpha_g = 1e5, 1.0, 293.15, 293.15, 0.5
    else:
        p, u, t_g, t_l, alpha_g = 1e5, 0.0, 293.15, 293.15, 0.5
    return primitive_to_q(p, u, t_g, t_l, alpha_g)

def initialize_state(domain, area_rule, p_rule, t_g_rule, t_l_rule, u_rule, alpha_g_rule, n=100):
    """
    Initialize conservative cell averages using composite midpoint quadrature.
    Ghost cells copy the nearest physical cell. n is a positive integer.
    Rules return SI scalars; temperatures/pressure > 0 and 0 < alpha_g < 1.
    """
    q = np.zeros((5, domain["n_total"]))

    for j in domain["physical_ind"]:
        h = domain["dx"][j] / n
        # No endpoint samples: a jump exactly on a face cannot pollute the average.
        x = domain["x_faces"][j] + (np.arange(n) + 0.5)*h
        area_q = np.zeros((n, 5))
        area_samples = np.zeros(n)

        for i, xi in enumerate(x):
            area_samples[i] = area_rule(xi)
            area_q[i] = area_samples[i]*primitive_to_q(p_rule(xi), u_rule(xi), t_g_rule(xi), t_l_rule(xi), alpha_g_rule(xi))

        # Use consistent quadrature for the integral and its normalization.
        q[:, j] = np.sum(area_q, axis=0) / np.sum(area_samples)


    # zero-gradient ghost cells
    first = domain["physical_ind"][0]
    last = domain["physical_ind"][-1]
    q[:, :first] = q[:, first:first + 1]
    q[:, last + 1:] = q[:, last:last + 1]

    return q


def apply_boundary_conditions(domain, q, case, side, check=False, rtol=1e-3):
    """
    q is (5, n_total); side is "left"/"right"; use the case names below.
    At least one ghost per side, and no more reflective ghosts than physical cells.
    check=True retains the optional fixed-boundary/initial-state diagnostic.
    fixed: fill ghosts with fixed_state(side); optionally check the initial edge state.
    zerograd: all ghosts copy the nearest physical cell.
    periodic: ghosts copy successive physical cells from the opposite side.
    reflective: mirror successive physical cells and reverse momentum.
    """
    # A dictionary allows the driver to prescribe a different pressure each stage.
    pressure = case.get("pressure") if isinstance(case, dict) else None
    case = case["type"] if isinstance(case, dict) else case
    physical = domain["physical_ind"]

    if side == "left":
        edge = physical[0]
        ghosts = domain["ghost_left_ind"][::-1]  # Nearest ghost first.
    else:
        edge = physical[-1]
        ghosts = domain["ghost_right_ind"]
        
    if case == "g3_inlet":
        u = 5.0  
        x_inlet = domain["x_faces"][physical[0]]
        p = g3_initial_p(x_inlet)
        t_g = g3_initial_t_g(x_inlet)
        t_l = g3_initial_t_l(x_inlet)
        alpha_g = g3_initial_alpha_g(x_inlet)

        # Recalculate momentum and energy consistently with the new velocity.
        q_bc = primitive_to_q(p, u, t_g, t_l, alpha_g)

        for ghost in ghosts:
            q[:, ghost] = q_bc

        return q

    if case == "g3_outlet":
        # Recover the primitive values in the last physical cell.
        primitives = q_to_primitive(q[:, edge])

        if primitives["u"] >= primitives["c"]:
            # Supersonic outflow: information travels out of the domain.
            # Extrapolate all variables instead of imposing back pressure.
            q_bc = q[:, edge].copy()
        else:
            # Subsonic outflow: impose pressure and copy the remaining
            # primitive variables from the adjacent physical cell.
            p = g3_outlet_pressure() if pressure is None else pressure
            u = primitives["u"]
            t_g = primitives["t_g"]
            t_l = primitives["t_l"]
            alpha_g = primitives["alpha_g"]

            # Recalculate densities and energy consistently with this pressure.
            q_bc = primitive_to_q(p, u, t_g, t_l, alpha_g)

        for ghost in ghosts:
            q[:, ghost] = q_bc

        return q

    if side == "left":
        edge = physical[0]
        ghosts = domain["ghost_left_ind"][::-1]  # Nearest ghost first
    else:
        edge = physical[-1]
        ghosts = domain["ghost_right_ind"]       # Nearest ghost first

    if case == "fixed":
        q_bc = fixed_state(side)
        if check and not np.allclose(q_bc, q[:, edge], rtol=rtol, atol=0):
            raise ValueError(f"{side}: fixed state does not match the initial edge state.")
        for ghost in ghosts:
            q[:, ghost] = q_bc
        return q

    for i, ghost in enumerate(ghosts):
        if case == "zerograd":
            q[:, ghost] = q[:, edge]

        elif case == "periodic":
            source = physical[(-1 - i) % len(physical)] if side == "left" else physical[i % len(physical)]
            q[:, ghost] = q[:, source]

        elif case == "reflective":
            source = physical[i] if side == "left" else physical[-1 - i]
            q[:, ghost] = q[:, source]
            q[2, ghost] = -q[2, source]

    return q
