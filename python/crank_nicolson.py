"""Crank-Nicolson with Newton or relaxed fixed-point iteration."""

import numpy as np

from conditions import apply_boundary_conditions
from solver_functions import get_primitives, validate_state, InvalidStateError
from miscellaneous import conservation_check
from nonlinear_solvers import fixed_point_CN, newton_CN, NonlinearConvergenceError


def crank_nicolson(q_initial, domain, final_time, bc_settings, tau, save_every=100, eps_tol=1e-6, limit=1000, relax_factor=1.0, nonlinear_method="newton", eps_j=1e-6, q_reference=None, jacobian_every=3, reconstruction="muscl", pressure_fallback="first_order", convergence_check="scaled", min_tau=1e-12, max_retries=12):
    """Integrate to final_time; shorten the last step when necessary.

    Newton is the default; set nonlinear_method="fpi" for fixed point.
    jacobian_every controls Jacobian rebuilds within each Newton solve.
    Returns the final state and primitive history, including ghost cells.
    """
    if not isinstance(save_every, (int, np.integer)) or save_every <= 0: raise ValueError("save_every must be a positive integer.")
    if not np.isfinite(final_time) or final_time < 0: raise ValueError("final_time must be finite and nonnegative.")
    if not np.isfinite(tau) or tau <= 0: raise ValueError("tau must be finite and positive.")
    if not np.isfinite(eps_tol) or eps_tol <= 0: raise ValueError("eps_tol must be finite and positive.")
    if not isinstance(limit, (int, np.integer)) or limit <= 0: raise ValueError("limit must be a positive integer.")
    if nonlinear_method not in ("newton", "fpi"): raise ValueError("Choose nonlinear_method='newton' or 'fpi'.")
    if not np.isfinite(min_tau) or min_tau <= 0 or min_tau > tau: raise ValueError("min_tau must be finite, positive and no larger than tau.")
    if not isinstance(max_retries, (int, np.integer)) or max_retries < 0: raise ValueError("max_retries must be a nonnegative integer.")

    if not np.isfinite(relax_factor) or not 0 < relax_factor <= 1: raise ValueError("relax_factor must be in (0, 1].")
    if nonlinear_method == "newton" and (not np.isfinite(eps_j) or eps_j <= 0): raise ValueError("eps_j must be finite and positive.")

    if convergence_check not in ("scaled", "teacher"): raise ValueError("convergence_check must be scaled or teacher.")

    iteration_label = "NWT" if nonlinear_method == "newton" else "FP"

    t = 0.0
    counter = 0
    q_current = q_initial.copy()
    physical = domain["physical_ind"]

    q_current = apply_boundary_conditions(domain, q_current, bc_settings["left"], "left")
    q_current = apply_boundary_conditions(domain, q_current, bc_settings["right"], "right")
    validate_state(q_current, domain)
    # Save the initial state, including ghost cells.
    primitives = get_primitives(q_current)
    history = {"times": [t], "primitives": {}}

    for name in primitives:
        history["primitives"][name] = [primitives[name].copy()]

       # Retain a smaller successful time step after a rejected attempt.
    current_tau = tau

    while t < final_time:
        remaining = final_time - t
        step = min(current_tau, remaining)
        q_start = q_current.copy()
        retries = 0

        while True:
            if not np.isfinite(step) or step <= 0 or t + step == t:
                raise RuntimeError("Time step is too small to advance time.")

            try:
                if nonlinear_method == "newton":
                    q_candidate = newton_CN(q_start, step, domain, bc_settings, eps_tol=eps_tol, limit=limit, relax_factor=relax_factor, time=t, eps_j=eps_j, q_reference=q_reference, jacobian_every=jacobian_every, reconstruction=reconstruction, pressure_fallback=pressure_fallback, convergence_check=convergence_check)
                else:
                    q_candidate = fixed_point_CN(q_start, step, domain, bc_settings, eps_tol=eps_tol, limit=limit, relax_factor=relax_factor, time=t, reconstruction=reconstruction, pressure_fallback=pressure_fallback, convergence_check=convergence_check)

                validate_state(q_candidate, domain)
                break

            except (InvalidStateError, NonlinearConvergenceError) as error:
                smaller_step = step/2

                if retries >= max_retries or smaller_step < min_tau or t + smaller_step == t:
                    raise RuntimeError(f"CN failed at t={t:.6g} s, dt={step:.6g} s after {retries} retries. Last accepted state retained; last failure: {error}") from error

                retries += 1
                print(f"\rCN | t={t:.4e} | retry {retries}/{max_retries} | dt={smaller_step:.3e}".ljust(76), flush=True)
                print(f"    {error}", flush=True)

                step = smaller_step
                # Restart from q_start. Neither time nor history advances.

        # Reached only after a successful nonlinear solve.
        q_current = q_candidate
        current_tau = min(current_tau, step)

        # Recalculate this after retries, since step may have changed.
        last_step = step >= remaining or t + step >= final_time
        t = final_time if last_step else t + step
        counter += 1

        print(f"\rCN | t={t:.4e} | {iteration_label} converged".ljust(76), end="", flush=True)

        primitives = get_primitives(q_current)

        if counter % save_every == 0 or last_step:
            history["times"].append(t)

            for name in primitives:
                history["primitives"][name].append(primitives[name].copy())

            min_p = np.min(primitives["p"][physical])
            max_p = np.max(primitives["p"][physical])
            max_speed = np.max(np.abs(primitives["u"][physical]))

            print(f"\rIteration: {counter} | t={t:.3e} | dt={step:.2e} | min_p={min_p:.3e} | max_p={max_p:.3e} | max_u={max_speed:.3e}", flush=True)

    history["times"] = np.array(history["times"])

    for name in history["primitives"]:
        history["primitives"][name] = np.array(history["primitives"][name])

    conservation_check(q_initial, q_current, domain)
    return q_current, history
