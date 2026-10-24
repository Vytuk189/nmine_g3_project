import numpy as np

from conditions import apply_boundary_conditions
from solver_functions import compute_residual


def fixed_point_euler(q_initial, tau, domain, bc_settings, eps_tol=1e-6, limit=1000, relax_factor=0.25, time=0.0):
    """Solve one backward Euler step; time is used only in progress output.

    eps_tol bounds each cell relative fixed-point defect; limit caps iterations.
    relax_factor scales each correction. Returns the converged full state.
    """
    if not 0 < relax_factor <= 1: raise ValueError("relax_factor must be in (0, 1].")

    physical = domain["physical_ind"]
    q_n = q_initial.copy()
    q_n = apply_boundary_conditions(domain, q_n, bc_settings["left"], "left")
    q_n = apply_boundary_conditions(domain, q_n, bc_settings["right"], "right")
    q_m = q_n.copy()

    for counter in range(1, limit + 1):
        r_m = compute_residual(q_m, domain)
        q_target = q_n[:, physical] + tau*r_m
        correction = q_target - q_m[:, physical]

        cell_errors = np.zeros(len(physical))

        for i, j in enumerate(physical):
            difference = np.linalg.norm(correction[:, i], 2)
            magnitude = np.linalg.norm(q_m[:, j], 2)
            cell_errors[i] = difference/max(magnitude, np.finfo(float).tiny)

        if not np.all(np.isfinite(cell_errors)):
            print()
            raise RuntimeError("Fixed-point iteration produced a non-finite error.")

        max_error = np.max(cell_errors)
        print(f"\rImplicit Euler | Time: t = {time:.6g} s | Fixed-point iteration: {counter}/{limit}, max error: {max_error:.3e}".ljust(120), end="", flush=True)

        if np.all(cell_errors < eps_tol):
            return q_m

        q_next = q_m.copy()
        q_next[:, physical] = q_m[:, physical] + relax_factor*correction
        q_next = apply_boundary_conditions(domain, q_next, bc_settings["left"], "left")
        q_next = apply_boundary_conditions(domain, q_next, bc_settings["right"], "right")
        q_m = q_next

    print()
    raise RuntimeError(f"Fixed-point iteration did not converge after {limit} iterations; last checked maximum cell error = {max_error:.3e}.")

def fixed_point_CN(q_initial, tau, domain, bc_settings, eps_tol=1e-5, limit=1000):
    physical = domain["physical_ind"]
    q_initial = q_initial.copy()
    q_initial = apply_boundary_conditions(domain, q_initial, bc_settings["left"], "left")
    q_initial = apply_boundary_conditions(domain, q_initial, bc_settings["right"], "right")

    r_initial = compute_residual(q_initial, domain)
    q_m = q_initial.copy()

    for counter in range(1, limit + 1):
        print(f"\rPerforming fixed-point iteration: {counter}/{limit}", end="", flush=True)
        r_m = compute_residual(q_m, domain)

        q_m_next = q_m.copy()
        q_m_next[:, physical] = q_initial[:, physical] + tau/2*(r_initial + r_m)
        q_m_next = apply_boundary_conditions(domain, q_m_next, bc_settings["left"], "left")
        q_m_next = apply_boundary_conditions(domain, q_m_next, bc_settings["right"], "right")

        cell_errors = np.zeros(len(physical))

        for i, j in enumerate(physical):
            difference = np.linalg.norm(q_m_next[:, j] - q_m[:, j], 2)
            magnitude = np.linalg.norm(q_m_next[:, j], 2)
            cell_errors[i] = difference/max(magnitude, np.finfo(float).tiny)

        if not np.all(np.isfinite(cell_errors)):
            raise RuntimeError("Fixed-point iteration produced a non-finite error.")

        q_m = q_m_next
        print(f"\rFixed-point iteration: {counter}/{limit}, max error: {np.max(cell_errors):.3e}".ljust(120), end="", flush=True)

        if np.all(cell_errors < eps_tol):
            # FPI is done to find residual, not Q
            return compute_residual(q_m, domain)

    raise RuntimeError(f"Fixed-point iteration did not converge after {limit} iterations; maximum cell error = {np.max(cell_errors):.3e}.")
