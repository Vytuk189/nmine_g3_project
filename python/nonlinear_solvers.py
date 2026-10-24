"""Fixed-point and Newton solvers for individual implicit time steps."""

import numpy as np
from scipy.sparse import csc_matrix, eye
from scipy.sparse.linalg import splu

from conditions import apply_boundary_conditions
from solver_functions import compute_residual, get_primitives


def reference_state(q, domain, q_reference=None):
    """Calculate five reference scales for the Jacobian perturbations."""
    if q_reference is not None:
        return np.asarray(q_reference, dtype=float)

    physical = domain["physical_ind"]
    primitives = get_primitives(q)

    reference = np.max(np.abs(q[:, physical]), axis=1)

    # Momentum needs a nonzero scale even when the initial velocity is zero.
    reference[2] = np.max(primitives["rho"][physical]*(np.abs(primitives["u"][physical]) + primitives["c"][physical]))

    # Natural scale of the dimensionless gas volume fraction.
    reference[4] = 1.0

    return reference


def residual_jacobian(q, r, domain, bc_settings, reference, eps_j=1e-6):
    """Approximate J_R using perturbations of the physical conservative variables."""
    physical = domain["physical_ind"]
    n_cells = len(physical)
    periodic = "periodic" in bc_settings.values()

    # Five-colour groups for nonperiodic boundaries.
    # With periodic boundaries, perturb cells individually.
    if periodic:
        groups = [np.array([j]) for j in range(n_cells)]
    else:
        groups = [np.arange(k, n_cells, 5) for k in range(5)]

    rows = []
    columns = []
    values = []

    for component in range(5):
        for group in groups:
            if len(group) == 0:
                continue

            q_perturbed = q.copy()
            perturbations = np.zeros(len(group))

            # Perturb the same component in every cell of this group.
            for k, j in enumerate(group):
                cell = physical[j]
                change = eps_j*max(abs(q[component, cell]), reference[component])

                # Keep the volume-fraction perturbation inside (0, 1).
                if component == 4:
                    change = min(change, 0.25)

                    if q[component, cell] >= 0.5:
                        change = -change

                q_perturbed[component, cell] += change

                # Actual change after floating-point addition.
                perturbations[k] = q_perturbed[component, cell] - q[component, cell]

            q_perturbed = apply_boundary_conditions(domain, q_perturbed, bc_settings["left"], "left")
            q_perturbed = apply_boundary_conditions(domain, q_perturbed, bc_settings["right"], "right")

            r_perturbed = compute_residual(q_perturbed, domain)
            residual_change = r_perturbed - r

            # Extract the Jacobian column belonging to each perturbed cell.
            for k, j in enumerate(group):
                if periodic:
                    affected_cells = range(n_cells)
                else:
                    affected_cells = range(max(0, j - 2), min(n_cells, j + 3))

                column = 5*j + component

                for i in affected_cells:
                    for equation in range(5):
                        row = 5*i + equation
                        derivative = residual_change[equation, i]/perturbations[k]

                        rows.append(row)
                        columns.append(column)
                        values.append(derivative)

    return csc_matrix((values, (rows, columns)), shape=(5*n_cells, 5*n_cells))


def fixed_point_euler(q_initial, tau, domain, bc_settings, eps_tol=1e-6, limit=1000, relax_factor=0.25, time=0.0):
    """Solve one backward Euler step using relaxed fixed-point iteration."""
    physical = domain["physical_ind"]
    n_cells = len(physical)

    # Known state at the beginning of the time step.
    q_n = np.array(q_initial, dtype=float, copy=True)
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

        # Relative change of the five-component vector in every physical cell.
        cell_errors = np.zeros(n_cells)

        for i, j in enumerate(physical):
            difference = np.linalg.norm(q_next[:, j] - q_m[:, j], 2)
            magnitude = np.linalg.norm(q_next[:, j], 2)
            cell_errors[i] = difference/max(magnitude, np.finfo(float).tiny)

        if not np.all(np.isfinite(cell_errors)):
            print()
            raise RuntimeError("Fixed-point iteration produced a non-finite error.")

        max_error = np.max(cell_errors)
        print(f"\rIE | t={time:.4e} | FP {counter}/{limit} | err={max_error:.3e}".ljust(76), end="", flush=True)

        if np.all(cell_errors < eps_tol):
            return q_next

        q_m = q_next

    print()
    raise RuntimeError(f"Fixed-point iteration did not converge after {limit} iterations.")


def fixed_point_CN(q_initial, tau, domain, bc_settings, eps_tol=1e-6, limit=1000, relax_factor=0.25, time=0.0):
    """Solve one Crank-Nicolson step using relaxed fixed-point iteration."""
    physical = domain["physical_ind"]
    n_cells = len(physical)

    # Known state and residual at the beginning of the time step.
    q_n = np.array(q_initial, dtype=float, copy=True)
    q_n = apply_boundary_conditions(domain, q_n, bc_settings["left"], "left")
    q_n = apply_boundary_conditions(domain, q_n, bc_settings["right"], "right")
    r_n = compute_residual(q_n, domain)

    # Initial guess for the new time level.
    q_m = q_n.copy()

    for counter in range(1, limit + 1):
        r_m = compute_residual(q_m, domain)

        # Relaxed Crank-Nicolson fixed-point update.
        q_next = q_m.copy()
        q_next[:, physical] = (1 - relax_factor)*q_m[:, physical] + relax_factor*(q_n[:, physical] + tau/2*(r_n + r_m))

        q_next = apply_boundary_conditions(domain, q_next, bc_settings["left"], "left")
        q_next = apply_boundary_conditions(domain, q_next, bc_settings["right"], "right")

        # Relative change of the five-component vector in every physical cell.
        cell_errors = np.zeros(n_cells)

        for i, j in enumerate(physical):
            difference = np.linalg.norm(q_next[:, j] - q_m[:, j], 2)
            magnitude = np.linalg.norm(q_next[:, j], 2)
            cell_errors[i] = difference/max(magnitude, np.finfo(float).tiny)

        if not np.all(np.isfinite(cell_errors)):
            print()
            raise RuntimeError("Fixed-point iteration produced a non-finite error.")

        max_error = np.max(cell_errors)
        print(f"\rCN | t={time:.4e} | FP {counter}/{limit} | err={max_error:.3e}".ljust(76), end="", flush=True)

        if np.all(cell_errors < eps_tol):
            return q_next

        q_m = q_next

    print()
    raise RuntimeError(f"Fixed-point iteration did not converge after {limit} iterations.")


def newton_euler(q_initial, tau, domain, bc_settings, eps_tol=1e-6, limit=1000, relax_factor=1.0, time=0.0, eps_j=1e-6, q_reference=None, jacobian_every=3):
    """Solve one backward Euler step, periodically rebuilding the Jacobian."""
    if not isinstance(jacobian_every, (int, np.integer)) or jacobian_every <= 0:
        raise ValueError("jacobian_every must be a positive integer.")

    physical = domain["physical_ind"]
    n_cells = len(physical)

    # Known state at the beginning of the time step.
    q_n = np.array(q_initial, dtype=float, copy=True)
    q_n = apply_boundary_conditions(domain, q_n, bc_settings["left"], "left")
    q_n = apply_boundary_conditions(domain, q_n, bc_settings["right"], "right")

    # Initial guess and reference scales.
    q_m = q_n.copy()
    reference = reference_state(q_n, domain, q_reference)

    for counter in range(1, limit + 1):
        # Recalculate the residual and equation defect at EVERY iteration.
        r_m = compute_residual(q_m, domain)
        g_m = q_m[:, physical] - q_n[:, physical] - tau*r_m

        # Rebuild on iterations 1, 1+jacobian_every, 1+2*jacobian_every, ...
        if (counter - 1) % jacobian_every == 0:
            jacobian_r = residual_jacobian(q_m, r_m, domain, bc_settings, reference, eps_j)
            jacobian_g = eye(5*n_cells, format="csc") - tau*jacobian_r

            # Store the factorisation until the next Jacobian update.
            factorisation = splu(jacobian_g)

        # Solve using the current defect and the stored factorisation.
        correction = factorisation.solve(-g_m.flatten(order="F"))
        correction = correction.reshape((5, n_cells), order="F")

        q_next = q_m.copy()
        q_next[:, physical] = q_m[:, physical] + relax_factor*correction

        q_next = apply_boundary_conditions(domain, q_next, bc_settings["left"], "left")
        q_next = apply_boundary_conditions(domain, q_next, bc_settings["right"], "right")

        # Original convergence test: relative vector change in every cell.
        cell_errors = np.zeros(n_cells)

        for i, j in enumerate(physical):
            difference = np.linalg.norm(q_next[:, j] - q_m[:, j], 2)
            magnitude = np.linalg.norm(q_next[:, j], 2)
            cell_errors[i] = difference/max(magnitude, np.finfo(float).tiny)

        if not np.all(np.isfinite(cell_errors)):
            print()
            raise RuntimeError("Newton iteration produced a non-finite error.")

        max_error = np.max(cell_errors)
        print(f"\rIE | t={time:.4e} | NWT {counter}/{limit} | err={max_error:.3e}".ljust(76), end="", flush=True)

        if np.all(cell_errors < eps_tol):
            return q_next

        q_m = q_next

    print()
    raise RuntimeError(f"Newton iteration did not converge after {limit} iterations.")


def newton_CN(q_initial, tau, domain, bc_settings, eps_tol=1e-6, limit=1000, relax_factor=1.0, time=0.0, eps_j=1e-6, q_reference=None, jacobian_every=3):
    """Solve one Crank-Nicolson step, periodically rebuilding the Jacobian."""
    if not isinstance(jacobian_every, (int, np.integer)) or jacobian_every <= 0:
        raise ValueError("jacobian_every must be a positive integer.")

    physical = domain["physical_ind"]
    n_cells = len(physical)

    # Known state and residual at the beginning of the time step.
    q_n = np.array(q_initial, dtype=float, copy=True)
    q_n = apply_boundary_conditions(domain, q_n, bc_settings["left"], "left")
    q_n = apply_boundary_conditions(domain, q_n, bc_settings["right"], "right")
    r_n = compute_residual(q_n, domain)

    # Initial guess and reference scales.
    q_m = q_n.copy()
    reference = reference_state(q_n, domain, q_reference)

    for counter in range(1, limit + 1):
        # Recalculate the residual and equation defect at EVERY iteration.
        r_m = compute_residual(q_m, domain)
        g_m = q_m[:, physical] - q_n[:, physical] - tau/2*(r_n + r_m)

        # Rebuild on iterations 1, 1+jacobian_every, 1+2*jacobian_every, ...
        if (counter - 1) % jacobian_every == 0:
            jacobian_r = residual_jacobian(q_m, r_m, domain, bc_settings, reference, eps_j)
            jacobian_g = eye(5*n_cells, format="csc") - tau/2*jacobian_r

            # Store the factorisation until the next Jacobian update.
            factorisation = splu(jacobian_g)

        # Solve using the current defect and the stored factorisation.
        correction = factorisation.solve(-g_m.flatten(order="F"))
        correction = correction.reshape((5, n_cells), order="F")

        q_next = q_m.copy()
        q_next[:, physical] = q_m[:, physical] + relax_factor*correction

        q_next = apply_boundary_conditions(domain, q_next, bc_settings["left"], "left")
        q_next = apply_boundary_conditions(domain, q_next, bc_settings["right"], "right")

        # Original convergence test: relative vector change in every cell.
        cell_errors = np.zeros(n_cells)

        for i, j in enumerate(physical):
            difference = np.linalg.norm(q_next[:, j] - q_m[:, j], 2)
            magnitude = np.linalg.norm(q_next[:, j], 2)
            cell_errors[i] = difference/max(magnitude, np.finfo(float).tiny)

        if not np.all(np.isfinite(cell_errors)):
            print()
            raise RuntimeError("Newton iteration produced a non-finite error.")

        max_error = np.max(cell_errors)
        print(f"\rCN | t={time:.4e} | NWT {counter}/{limit} | err={max_error:.3e}".ljust(76), end="", flush=True)

        if np.all(cell_errors < eps_tol):
            return q_next

        q_m = q_next

    print()
    raise RuntimeError(f"Newton iteration did not converge after {limit} iterations.")
   