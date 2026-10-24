"""Crank-Nicolson with a prescribed time step and relaxed fixed-point iteration."""

import numpy as np

from conditions import apply_boundary_conditions
from solver_functions import get_primitives
import miscellaneous
from nonlinear_solvers import fixed_point_CN


def crank_nicolson(q_initial, domain, final_time, bc_settings, tau, save_every=100, eps_tol=1e-6, limit=1000, relax_factor=0.25):
    """Integrate to final_time; tau is in seconds and the final step is shortened.

    eps_tol, limit and relax_factor are forwarded to fixed_point_CN.
    save_every controls saved states and printed time-step summaries.
    Returns the final conservative state and primitive history, including ghosts.
    """
    if not isinstance(save_every, (int, np.integer)) or save_every <= 0: raise ValueError("save_every must be a positive integer.")
    if not np.isfinite(final_time) or final_time < 0: raise ValueError("final_time must be finite and nonnegative.")
    if not np.isfinite(tau) or tau <= 0: raise ValueError("tau must be finite and positive.")
    if not np.isfinite(eps_tol) or eps_tol <= 0: raise ValueError("eps_tol must be finite and positive.")
    if not isinstance(limit, (int, np.integer)) or limit <= 0: raise ValueError("limit must be a positive integer.")

    if not np.isfinite(relax_factor) or not 0 < relax_factor <= 1: raise ValueError("relax_factor must be in (0, 1].")

    t = 0.0
    counter = 0
    q_current = q_initial.copy()
    physical = domain["physical_ind"]
    q_current = apply_boundary_conditions(domain, q_current, bc_settings["left"], "left")
    q_current = apply_boundary_conditions(domain, q_current, bc_settings["right"], "right")

    # Save the initial state, including ghost cells.
    primitives = get_primitives(q_current)
    history = {"times": [t], "primitives": {}}

    for name in primitives:
        history["primitives"][name] = [primitives[name].copy()]

    while t < final_time:
        step = tau

        remaining = final_time - t
        last_step = step >= remaining or t + step >= final_time
        step = min(step, remaining)

        if not np.isfinite(step) or step <= 0 or t + step == t: raise ValueError("Invalid time step: time cannot advance.")

        # The nonlinear solver returns the new state with boundary conditions applied.
        q_current = fixed_point_CN(q_current, step, domain, bc_settings, eps_tol=eps_tol, limit=limit, relax_factor=relax_factor, time=t)

        t = final_time if last_step else t + step
        counter += 1
        print(f"\rCrank-Nicolson | Time: t = {t:.6g} s | Fixed-point iteration: converged".ljust(120), end="", flush=True)

        primitives = get_primitives(q_current)

        if counter % save_every == 0 or last_step:
            history["times"].append(t)

            for name in primitives:
                history["primitives"][name].append(primitives[name].copy())

            max_p = np.max(primitives["p"][physical])
            max_speed = np.max(np.abs(primitives["u"][physical]))

            print(f"\rIteration: {counter}, t: {t:.6g} s, tau: {step:.6g} s, max p: {max_p:.6g} Pa, max |u|: {max_speed:.6g} m/s".ljust(120))

    # Each primitive array has shape (number of saved times, N_total).
    history["times"] = np.array(history["times"])

    for name in history["primitives"]:
        history["primitives"][name] = np.array(history["primitives"][name])

    miscellaneous.conservation_check(q_initial, q_current, domain)
    return q_current, history
