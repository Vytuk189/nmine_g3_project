"""Fixed-point and Newton solvers for individual implicit time steps."""

import numpy as np
from scipy.sparse import csc_matrix, eye
from scipy.sparse.linalg import splu

from conditions import apply_boundary_conditions
from solver_functions import compute_residual, get_primitives, validate_state

class NonlinearConvergenceError(RuntimeError):
    """The current implicit-step attempt did not converge."""

def reference_state(q, domain, q_reference=None):
    """Calculate component scales; automatic scales also normalise convergence.

    q_reference overrides Jacobian perturbations only, not the stopping test.
    """
    if q_reference is not None:
        return np.asarray(q_reference, dtype=float)

    physical = domain["physical_ind"]
    primitives = get_primitives(q)

    reference = np.max(np.abs(q[:, physical]), axis=1)
    reference[0] = max(reference[0], 1e-3)  # kg/m3
    reference[1] = max(reference[1], 1e-3)  # kg/m3
    reference[3] = max(reference[3], 1.0)   # J/m3; avoid a zero energy scale.

    # Momentum needs a nonzero scale even when the initial velocity is zero.
    reference[2] = np.max(primitives["rho"][physical]*(np.abs(primitives["u"][physical]) + primitives["c"][physical]))

    # Natural scale of the dimensionless gas volume fraction.
    reference[4] = 1.0

    return reference


def residual_jacobian(q, r, domain, bc_settings, reference, eps_j=1e-6, reconstruction="muscl", pressure_fallback="first_order"):
    """Approximate J_R using perturbations of the physical conservative variables."""
    physical = domain["physical_ind"]
    n_cells = len(physical)
    periodic = "periodic" in bc_settings.values()

    # A perturbation affects at most five residual cells (j-2 through j+2).
    # Periodic supports wrap at the ends; only disjoint supports share a group.
    if periodic:
        groups = []
        group_rows = []
        for j in range(n_cells):
            affected = {(j + offset) % n_cells for offset in range(-2, 3)}
            for group, occupied in zip(groups, group_rows):
                if affected.isdisjoint(occupied):
                    group.append(j)
                    occupied.update(affected)
                    break
            else:
                groups.append([j])
                group_rows.append(affected.copy())
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
            validate_state(q_perturbed, domain)
            r_perturbed = compute_residual(q_perturbed, domain, reconstruction=reconstruction, pressure_fallback=pressure_fallback)
            residual_change = r_perturbed - r

            # Extract the Jacobian column belonging to each perturbed cell.
            for k, j in enumerate(group):
                if periodic:
                    affected_cells = sorted({(j + offset) % n_cells for offset in range(-2, 3)})
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


def convergence_error(q_next, q_m, correction, scale, convergence_check):
    """Return the existing stopping error, without changing either definition.

    scaled: largest abs(unrelaxed correction)/component scale in physical cells.
    teacher: matrix 2-norm of the applied update divided by that of the new state,
    including ghosts. This is the spectral norm, not the flattened vector norm.
    """
    if convergence_check == "scaled":
        return np.max(np.abs(correction)/scale[:, None])
    difference = np.linalg.norm(q_next - q_m, 2)
    magnitude = np.linalg.norm(q_next, 2)
    return difference/magnitude if magnitude > 0 else (0.0 if difference == 0 else np.inf)


def update_info(info, errors):
    """Record iteration count, error and recent error contraction factor.

    rate is the geometric mean of up to three successive error ratios.
    Below one means shrinking errors; above one means growing errors.
    One iteration has no measured ratio (NaN). A final exact zero gives rate=0.
    This is an observed diagnostic, not the formal order of Newton convergence.
    """
    recent = errors[-4:]
    ratios = [new/old for old, new in zip(recent[:-1], recent[1:]) if old > 0]
    rate = np.exp(np.mean(np.log(ratios))) if ratios and min(ratios) > 0 else (0.0 if ratios else np.nan)
    info.update(iterations=len(errors), error=errors[-1], rate=rate)


def fixed_point_euler(q_initial, tau, domain, bc_settings, eps_tol=1e-6, limit=1000, relax_factor=1.0, time=0.0, reconstruction="muscl", pressure_fallback="first_order", convergence_check="scaled", info=None):
    """Solve one backward Euler step; return full Q including updated ghosts.

    tau and eps_tol are positive; limit is a positive integer; 0 < relax_factor <= 1.
    Choose convergence_check="scaled" or "teacher", reconstruction="muscl" or
    "first_order", and pressure_fallback="first_order" or "gradual".
    Optional info={} receives iterations, error, rate and Jacobian build count.
    time labels progress only; it does not enter the equation.
    """
    physical = domain["physical_ind"]
    info = {} if info is None else info
    info.clear()
    info["jacobian_builds"] = 0
    errors = []

    q_n = np.array(q_initial, dtype=float, copy=True)
    q_n = apply_boundary_conditions(domain, q_n, bc_settings["left"], "left")
    q_n = apply_boundary_conditions(domain, q_n, bc_settings["right"], "right")
    validate_state(q_n, domain)
    q_m = q_n.copy()
    scale = reference_state(q_n, domain) if convergence_check == "scaled" else None

    for counter in range(1, limit + 1):
        # Always evaluate the CURRENT residual, even when reusing a Jacobian.
        r_m = compute_residual(q_m, domain, reconstruction=reconstruction, pressure_fallback=pressure_fallback)
        # The convergence test in scaled mode uses this UNRELAXED correction.
        correction = q_n[:, physical] + tau*r_m - q_m[:, physical]
        q_next = q_m.copy()
        q_next[:, physical] += relax_factor*correction
        validate_state(q_next, domain)
        q_next = apply_boundary_conditions(domain, q_next, bc_settings["left"], "left")
        q_next = apply_boundary_conditions(domain, q_next, bc_settings["right"], "right")

        error = convergence_error(q_next, q_m, correction, scale, convergence_check)
        if not np.isfinite(error):
            raise NonlinearConvergenceError("Nonlinear iteration produced a non-finite error.")
        errors.append(error)
        update_info(info, errors)
        print(f"\rIE t={time:.3e} | FP {counter}/{limit} | err={error:.2e} | r={info['rate']:.2g}".ljust(76), end="", flush=True)

        if error < eps_tol:
            return q_next
        q_m = q_next

    raise NonlinearConvergenceError(f"Fixed-point iteration did not converge after {limit} iterations; last error={error:.3e}.")


def fixed_point_CN(q_initial, tau, domain, bc_settings, eps_tol=1e-6, limit=1000, relax_factor=1.0, time=0.0, reconstruction="muscl", pressure_fallback="first_order", convergence_check="scaled", info=None):
    """Solve one Crank-Nicolson step; return full Q including updated ghosts.

    tau and eps_tol are positive; limit is a positive integer; 0 < relax_factor <= 1.
    Choose convergence_check="scaled" or "teacher", reconstruction="muscl" or
    "first_order", and pressure_fallback="first_order" or "gradual".
    Optional info={} receives iterations, error, rate and Jacobian build count.
    time labels progress only; it does not enter the equation.
    """
    physical = domain["physical_ind"]
    info = {} if info is None else info
    info.clear()
    info["jacobian_builds"] = 0
    errors = []

    q_n = np.array(q_initial, dtype=float, copy=True)
    q_n = apply_boundary_conditions(domain, q_n, bc_settings["left"], "left")
    q_n = apply_boundary_conditions(domain, q_n, bc_settings["right"], "right")
    validate_state(q_n, domain)
    r_n = compute_residual(q_n, domain, reconstruction=reconstruction, pressure_fallback=pressure_fallback)
    q_m = q_n.copy()
    scale = reference_state(q_n, domain) if convergence_check == "scaled" else None

    for counter in range(1, limit + 1):
        # Always evaluate the CURRENT residual, even when reusing a Jacobian.
        r_m = compute_residual(q_m, domain, reconstruction=reconstruction, pressure_fallback=pressure_fallback)
        # The convergence test in scaled mode uses this UNRELAXED correction.
        correction = q_n[:, physical] + tau/2*(r_n + r_m) - q_m[:, physical]
        q_next = q_m.copy()
        q_next[:, physical] += relax_factor*correction
        validate_state(q_next, domain)
        q_next = apply_boundary_conditions(domain, q_next, bc_settings["left"], "left")
        q_next = apply_boundary_conditions(domain, q_next, bc_settings["right"], "right")

        error = convergence_error(q_next, q_m, correction, scale, convergence_check)
        if not np.isfinite(error):
            raise NonlinearConvergenceError("Nonlinear iteration produced a non-finite error.")
        errors.append(error)
        update_info(info, errors)
        print(f"\rCN t={time:.3e} | FP {counter}/{limit} | err={error:.2e} | r={info['rate']:.2g}".ljust(76), end="", flush=True)

        if error < eps_tol:
            return q_next
        q_m = q_next

    raise NonlinearConvergenceError(f"Fixed-point iteration did not converge after {limit} iterations; last error={error:.3e}.")


def newton_euler(q_initial, tau, domain, bc_settings, eps_tol=1e-6, limit=1000, relax_factor=1.0, time=0.0, eps_j=1e-6, q_reference=None, jacobian_every=3, reconstruction="muscl", pressure_fallback="first_order", convergence_check="scaled", info=None, jacobian_cache=None):
    """Solve one backward Euler step; return full Q including updated ghosts.

    tau and eps_tol are positive; limit is a positive integer; 0 < relax_factor <= 1.
    Choose convergence_check="scaled" or "teacher", reconstruction="muscl" or
    "first_order", and pressure_fallback="first_order" or "gradual".
    Optional info={} receives iterations, error, rate and Jacobian build count.
    time labels progress only; it does not enter the equation.
    jacobian_every is a positive integer: rebuild inside the solve on iterations
    1, 1+jacobian_every, ...; iteration 1 can reuse the outer loop's cached matrix.
    eps_j > 0 controls finite differences; q_reference is None or five positive
    scales, ordered like Q. Cache belongs to one run/method/mesh only; the outer
    loop clears it on scheduled refreshes, failures and time-step changes.
    """
    physical = domain["physical_ind"]
    n_cells = len(physical)
    info = {} if info is None else info
    info.clear()
    info["jacobian_builds"] = 0
    errors = []

    q_n = np.array(q_initial, dtype=float, copy=True)
    q_n = apply_boundary_conditions(domain, q_n, bc_settings["left"], "left")
    q_n = apply_boundary_conditions(domain, q_n, bc_settings["right"], "right")
    validate_state(q_n, domain)
    q_m = q_n.copy()
    scale = reference_state(q_n, domain) if convergence_check == "scaled" else None
    reference = reference_state(q_n, domain, q_reference)
    cache = {} if jacobian_cache is None else jacobian_cache
    if cache.get("tau") != tau or cache.get("method") != "IE":
        cache.clear()  # I - tau*J_R (or I - tau/2*J_R) changed.

    for counter in range(1, limit + 1):
        # Always evaluate the CURRENT residual, even when reusing a Jacobian.
        r_m = compute_residual(q_m, domain, reconstruction=reconstruction, pressure_fallback=pressure_fallback)
        g_m = q_m[:, physical] - q_n[:, physical] - tau*r_m

        if "factorisation" not in cache or (counter > 1 and (counter - 1) % jacobian_every == 0):
            jacobian_r = residual_jacobian(q_m, r_m, domain, bc_settings, reference, eps_j, reconstruction=reconstruction, pressure_fallback=pressure_fallback)
            jacobian_g = eye(5*n_cells, format="csc") - tau*jacobian_r
            try:
                cache["factorisation"] = splu(jacobian_g)
            except RuntimeError as error:
                raise NonlinearConvergenceError("Newton matrix factorisation failed.") from error
            cache.update(tau=tau, method="IE")
            info["jacobian_builds"] += 1

        correction = cache["factorisation"].solve(-g_m.flatten(order="F"))
        correction = correction.reshape((5, n_cells), order="F")
        q_next = q_m.copy()
        q_next[:, physical] += relax_factor*correction
        validate_state(q_next, domain)
        q_next = apply_boundary_conditions(domain, q_next, bc_settings["left"], "left")
        q_next = apply_boundary_conditions(domain, q_next, bc_settings["right"], "right")

        error = convergence_error(q_next, q_m, correction, scale, convergence_check)
        if not np.isfinite(error):
            raise NonlinearConvergenceError("Nonlinear iteration produced a non-finite error.")
        errors.append(error)
        update_info(info, errors)
        print(f"\rIE t={time:.3e} | NWT {counter}/{limit} | err={error:.2e} | r={info['rate']:.2g}".ljust(76), end="", flush=True)

        if error < eps_tol:
            return q_next
        q_m = q_next

    raise NonlinearConvergenceError(f"Newton iteration did not converge after {limit} iterations; last error={error:.3e}.")


def newton_CN(q_initial, tau, domain, bc_settings, eps_tol=1e-6, limit=1000, relax_factor=1.0, time=0.0, eps_j=1e-6, q_reference=None, jacobian_every=3, reconstruction="muscl", pressure_fallback="first_order", convergence_check="scaled", info=None, jacobian_cache=None):
    """Solve one Crank-Nicolson step; return full Q including updated ghosts.

    tau and eps_tol are positive; limit is a positive integer; 0 < relax_factor <= 1.
    Choose convergence_check="scaled" or "teacher", reconstruction="muscl" or
    "first_order", and pressure_fallback="first_order" or "gradual".
    Optional info={} receives iterations, error, rate and Jacobian build count.
    time labels progress only; it does not enter the equation.
    jacobian_every is a positive integer: rebuild inside the solve on iterations
    1, 1+jacobian_every, ...; iteration 1 can reuse the outer loop's cached matrix.
    eps_j > 0 controls finite differences; q_reference is None or five positive
    scales, ordered like Q. Cache belongs to one run/method/mesh only; the outer
    loop clears it on scheduled refreshes, failures and time-step changes.
    """
    physical = domain["physical_ind"]
    n_cells = len(physical)
    info = {} if info is None else info
    info.clear()
    info["jacobian_builds"] = 0
    errors = []

    q_n = np.array(q_initial, dtype=float, copy=True)
    q_n = apply_boundary_conditions(domain, q_n, bc_settings["left"], "left")
    q_n = apply_boundary_conditions(domain, q_n, bc_settings["right"], "right")
    validate_state(q_n, domain)
    r_n = compute_residual(q_n, domain, reconstruction=reconstruction, pressure_fallback=pressure_fallback)
    q_m = q_n.copy()
    scale = reference_state(q_n, domain) if convergence_check == "scaled" else None
    reference = reference_state(q_n, domain, q_reference)
    cache = {} if jacobian_cache is None else jacobian_cache
    if cache.get("tau") != tau or cache.get("method") != "CN":
        cache.clear()  # I - tau*J_R (or I - tau/2*J_R) changed.

    for counter in range(1, limit + 1):
        # Always evaluate the CURRENT residual, even when reusing a Jacobian.
        r_m = compute_residual(q_m, domain, reconstruction=reconstruction, pressure_fallback=pressure_fallback)
        g_m = q_m[:, physical] - q_n[:, physical] - tau/2*(r_n + r_m)

        if "factorisation" not in cache or (counter > 1 and (counter - 1) % jacobian_every == 0):
            jacobian_r = residual_jacobian(q_m, r_m, domain, bc_settings, reference, eps_j, reconstruction=reconstruction, pressure_fallback=pressure_fallback)
            jacobian_g = eye(5*n_cells, format="csc") - tau/2*jacobian_r
            try:
                cache["factorisation"] = splu(jacobian_g)
            except RuntimeError as error:
                raise NonlinearConvergenceError("Newton matrix factorisation failed.") from error
            cache.update(tau=tau, method="CN")
            info["jacobian_builds"] += 1

        correction = cache["factorisation"].solve(-g_m.flatten(order="F"))
        correction = correction.reshape((5, n_cells), order="F")
        q_next = q_m.copy()
        q_next[:, physical] += relax_factor*correction
        validate_state(q_next, domain)
        q_next = apply_boundary_conditions(domain, q_next, bc_settings["left"], "left")
        q_next = apply_boundary_conditions(domain, q_next, bc_settings["right"], "right")

        error = convergence_error(q_next, q_m, correction, scale, convergence_check)
        if not np.isfinite(error):
            raise NonlinearConvergenceError("Nonlinear iteration produced a non-finite error.")
        errors.append(error)
        update_info(info, errors)
        print(f"\rCN t={time:.3e} | NWT {counter}/{limit} | err={error:.2e} | r={info['rate']:.2g}".ljust(76), end="", flush=True)

        if error < eps_tol:
            return q_next
        q_m = q_next

    raise NonlinearConvergenceError(f"Newton iteration did not converge after {limit} iterations; last error={error:.3e}.")

