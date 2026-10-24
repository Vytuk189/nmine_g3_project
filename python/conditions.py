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
    Initialize conservative cell averages using the trapezoidal rule.
    Ghost cells copy the nearest physical cell.
    """
    q = np.zeros((5, domain["n_total"]))

    for j in domain["physical_ind"]:
        x = np.linspace(domain["x_faces"][j], domain["x_faces"][j + 1], n + 1)
        h = domain["dx"][j] / n
        
        area_q = np.zeros((n + 1, 5))

        for i, xi in enumerate(x):
            area_q[i] = area_rule(xi)*primitive_to_q(p_rule(xi), u_rule(xi), t_g_rule(xi), t_l_rule(xi), alpha_g_rule(xi))
        
        traps = 0.5*(area_q[:-1] + area_q[1:])*h
        q[:, j] = np.sum(traps, axis=0) / domain["volumes"][j]


    # zero-gradient ghost cells
    first = domain["physical_ind"][0]
    last = domain["physical_ind"][-1]
    q[:, :first] = q[:, first:first + 1]
    q[:, last + 1:] = q[:, last:last + 1]

    return q


def apply_boundary_conditions(domain, q, case, side, check=False, rtol=1e-3):
    """
    fixed: no ghost cells; optionally check the initial boundary state.
    zerograd: all ghosts copy the nearest physical cell.
    periodic: all ghosts copy the opposite edge physical cell.
    reflective: mirror successive physical cells and reverse momentum.
    """
    if side not in ("left", "right"):
        raise ValueError(f"Unknown side '{side}'.")

    if case not in ("fixed", "zerograd", "reflective", "periodic"):
        raise ValueError(f"Unknown boundary case '{case}'.")

    physical = domain["physical_ind"]

    if side == "left":
        edge = physical[0]
        opposite = physical[-1]
        ghosts = domain["ghost_left_ind"][::-1]  # Nearest ghost first
    else:
        edge = physical[-1]
        opposite = physical[0]
        ghosts = domain["ghost_right_ind"]       # Nearest ghost first

    if case == "fixed":
        if len(ghosts) != 0:
            raise ValueError(
                f"'fixed' requires no ghost cells on the {side} side."
            )

        q_bc = fixed_state(side)

        if check and not np.allclose(q_bc, q[:, edge], rtol=rtol, atol=0):
            raise ValueError(
                f"{side}: fixed state does not match the initial condition.\n"
                f"Q_bc = {q_bc}\nQ_cell = {q[:, edge]}"
            )

        return q

    if len(ghosts) == 0:
        raise ValueError(f"'{case}' requires ghost cells on the {side} side.")

    if case == "reflective" and len(ghosts) > len(physical):
        raise ValueError("Not enough physical cells to mirror all ghost cells.")

    for i, ghost in enumerate(ghosts):
        if case == "zerograd":
            q[:, ghost] = q[:, edge]

        elif case == "periodic":
            q[:, ghost] = q[:, opposite]

        elif case == "reflective":
            source = physical[i] if side == "left" else physical[-1 - i]
            q[:, ghost] = q[:, source]
            q[2, ghost] = -q[2, source]

    return q
