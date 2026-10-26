"""Edit the initial profiles and fixed boundary states here for each scenario."""

import numpy as np
from solver_functions import primitive_to_q


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
    Ghost cells copy the nearest physical cell.
    """
    if not isinstance(n, (int, np.integer)) or n <= 0: raise ValueError("n must be a positive integer.")
    q = np.zeros((5, domain["n_total"]))

    for j in domain["physical_ind"]:
        h = domain["dx"][j] / n
        # No endpoint samples: a jump exactly on a face cannot pollute the average.
        x = domain["x_faces"][j] + (np.arange(n) + 0.5)*h
        area_q = np.zeros((n, 5))

        for i, xi in enumerate(x):
            area_q[i] = area_rule(xi)*primitive_to_q(p_rule(xi), u_rule(xi), t_g_rule(xi), t_l_rule(xi), alpha_g_rule(xi))
        
        q[:, j] = h*np.sum(area_q, axis=0) / domain["volumes"][j]


    # zero-gradient ghost cells
    first = domain["physical_ind"][0]
    last = domain["physical_ind"][-1]
    q[:, :first] = q[:, first:first + 1]
    q[:, last + 1:] = q[:, last:last + 1]

    return q


def apply_boundary_conditions(domain, q, case, side, check=False, rtol=1e-3):
    """
    fixed: fill ghosts with fixed_state(side); optionally check the initial edge state.
    zerograd: all ghosts copy the nearest physical cell.
    periodic: ghosts copy successive physical cells from the opposite side.
    reflective: mirror successive physical cells and reverse momentum.
    """
    if side not in ("left", "right"):
        raise ValueError(f"Unknown side '{side}'.")

    if case not in ("fixed", "zerograd", "reflective", "periodic"):
        raise ValueError(f"Unknown boundary case '{case}'.")

    physical = domain["physical_ind"]

    if side == "left":
        edge = physical[0]
        ghosts = domain["ghost_left_ind"][::-1]  # Nearest ghost first
    else:
        edge = physical[-1]
        ghosts = domain["ghost_right_ind"]       # Nearest ghost first

    if len(ghosts) == 0:
        raise ValueError(f"'{case}' requires at least one ghost cell on the {side} side.")

    if case == "fixed":
        q_bc = fixed_state(side)
        if check and not np.allclose(q_bc, q[:, edge], rtol=rtol, atol=0):
            raise ValueError(f"{side}: fixed state does not match the initial edge state.")
        for ghost in ghosts:
            q[:, ghost] = q_bc
        return q

    if case == "reflective" and len(ghosts) > len(physical):
        raise ValueError("Not enough physical cells to mirror all ghost cells.")

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
