import numpy as np

from conditions import apply_boundary_conditions
from solver_functions import compute_residual


def fixed_point_euler(q_initial, tau, domain, bc_settings, eps_tol=1e-6, limit=1000, relax_factor=0.25, time=0.0):
    """Solve one backward Euler step using relaxed fixed-point iteration.

    Convergence is checked using the relative change between successive
    iterates in every physical cell. time is used only for progress output.
    """
    if not np.isfinite(tau) or tau <= 0: raise ValueError("tau must be finite and positive.")
    if not np.isfinite(relax_factor) or not 0 < relax_factor <= 1: raise ValueError("relax_factor must be in (0, 1].")
    if not np.isfinite(eps_tol) or eps_tol <= 0: raise ValueError("eps_tol must be finite and positive.")
    if not isinstance(limit, (int, np.integer)) or limit <= 0: raise ValueError("limit must be a positive integer.")

    physical = domain["physical_ind"]

    # Old time-level state: fixed throughout the iteration.
    q_n = q_initial.copy()
    q_n = apply_boundary_conditions(domain, q_n, bc_settings["left"], "left")
    q_n = apply_boundary_conditions(domain, q_n, bc_settings["right"], "right")

    # Initial guess for the new time level.
    q_m = q_n.copy()

    for counter in range(1, limit + 1):
        r_m = compute_residual(q_m, domain)

        # Relaxed backward Euler fixed-point update.
        q_next = q_m.copy()
        q_next[:, physical] = (1 - relax_factor)*q_m[:, physical] + relax_factor*(q_n[:, physical] + tau*r_m)

        q_next = apply_boundary_conditions(domain, q_next, bc_settings["left"], "left")
        q_next = apply_boundary_conditions(domain, q_next, bc_settings["right"], "right")

        # Relative change of the five-component vector in each cell.
        cell_errors = np.zeros(len(physical))

        for i, j in enumerate(physical):
            difference = np.linalg.norm(q_next[:, j] - q_m[:, j], 2)
            magnitude = np.linalg.norm(q_next[:, j], 2)
            cell_errors[i] = difference/max(magnitude, np.finfo(float).tiny)

        if not np.all(np.isfinite(cell_errors)):
            print()
            raise RuntimeError("Fixed-point iteration produced a non-finite error.")

        max_error = np.max(cell_errors)
        print(f"\rImplicit Euler | Time: t = {time:.6g} s | Fixed-point iteration: {counter}/{limit}, max error: {max_error:.3e}".ljust(120), end="", flush=True)

        if np.all(cell_errors < eps_tol):
            return q_next

        q_m = q_next

    print()
    raise RuntimeError(f"Fixed-point iteration did not converge after {limit} iterations; maximum cell error = {max_error:.3e}.")

def fixed_point_CN(q_initial, tau, domain, bc_settings, eps_tol=1e-6, limit=1000, relax_factor=0.25, time=0.0):
    """Solve one Crank-Nicolson step using relaxed fixed-point iteration.

    Convergence is checked using the relative change between successive
    iterates in every physical cell. time is used only for progress output.
    """
    if not np.isfinite(tau) or tau <= 0: raise ValueError("tau must be finite and positive.")
    if not np.isfinite(relax_factor) or not 0 < relax_factor <= 1: raise ValueError("relax_factor must be in (0, 1].")
    if not np.isfinite(eps_tol) or eps_tol <= 0: raise ValueError("eps_tol must be finite and positive.")
    if not isinstance(limit, (int, np.integer)) or limit <= 0: raise ValueError("limit must be a positive integer.")

    physical = domain["physical_ind"]

    # Old time-level state: fixed throughout the iteration.
    q_n = q_initial.copy()
    q_n = apply_boundary_conditions(domain, q_n, bc_settings["left"], "left")
    q_n = apply_boundary_conditions(domain, q_n, bc_settings["right"], "right")
    r_n = compute_residual(q_n, domain)

    # Initial guess for the new time level.
    q_m = q_n.copy()

    for counter in range(1, limit + 1):
        r_m = compute_residual(q_m, domain)

        # Relaxed Crank-Nicolson fixed-point update.
        q_next = q_m.copy()
        q_next[:, physical] = (1 - relax_factor)*q_m[:, physical] + relax_factor*(q_n[:, physical] + tau/2*(r_m+r_n))

        q_next = apply_boundary_conditions(domain, q_next, bc_settings["left"], "left")
        q_next = apply_boundary_conditions(domain, q_next, bc_settings["right"], "right")

        # Relative change of the five-component vector in each cell.
        cell_errors = np.zeros(len(physical))

        for i, j in enumerate(physical):
            difference = np.linalg.norm(q_next[:, j] - q_m[:, j], 2)
            magnitude = np.linalg.norm(q_next[:, j], 2)
            cell_errors[i] = difference/max(magnitude, np.finfo(float).tiny)

        if not np.all(np.isfinite(cell_errors)):
            print()
            raise RuntimeError("Fixed-point iteration produced a non-finite error.")

        max_error = np.max(cell_errors)
        print(f"\rCrank-Nicolson | Time: t = {time:.6g} s | Fixed-point iteration: {counter}/{limit}, max error: {max_error:.3e}".ljust(120), end="", flush=True)

        if np.all(cell_errors < eps_tol):
            return q_next

        q_m = q_next

    print()
    raise RuntimeError(f"Fixed-point iteration did not converge after {limit} iterations; maximum cell error = {max_error:.3e}.")
