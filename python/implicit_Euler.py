"""Backward Euler with Newton or relaxed fixed-point iteration."""

import numpy as np

from conditions import apply_boundary_conditions
from miscellaneous import conservation_check
from solver_functions import get_primitives, validate_state, InvalidStateError
from nonlinear_solvers import fixed_point_euler, newton_euler, NonlinearConvergenceError


def implicit_euler(q_initial, domain, final_time, bc_settings, tau, save_every=100, eps_tol=1e-6, limit=1000, relax_factor=1.0, nonlinear_method="newton", eps_j=1e-6, q_reference=None, jacobian_every=3, reconstruction="muscl", pressure_fallback="first_order", convergence_check="scaled", min_tau=1e-12, max_retries=12, jacobian_step_every=1, adaptive_tau=False, growth_after=3, growth_iterations=8, growth_rate=0.5):
    """Advance full Q to final_time, returning (q_final, history).

    Inputs use SI units. tau > 0 is the requested step and the growth ceiling;
    final_time >= 0. Choose nonlinear_method="newton" or "fpi". eps_tol > 0,
    limit/save_every are positive integers and 0 < relax_factor <= 1.
    reconstruction: "muscl" or "first_order"; pressure_fallback: "first_order"
    or "gradual"; convergence_check: "scaled" or "teacher" (see nonlinear_solvers).

    Newton only: eps_j > 0, q_reference=None or five positive component scales.
    jacobian_every controls INNER iteration rebuilds. jacobian_step_every is a
    positive integer: refresh at outer steps 1, 1+K, ...; otherwise reuse the
    previous solve's last matrix. Step changes and failures force a refresh.

    Failed attempts restart from the last accepted Q with half the step, subject
    to min_tau > 0 and max_retries >= 0. Default adaptive_tau=False retains the
    smaller step for the rest of the run. True doubles it (at most back to tau)
    after growth_after consecutive easy steps: no rejection, at most
    growth_iterations nonlinear iterations and observed rate <= growth_rate.
    Use positive integer growth counts and 0 < growth_rate < 1. This controls
    nonlinear difficulty, NOT temporal truncation error or guaranteed stability.

    CFL is measured using the START state and the actual attempted step.
    history["diagnostics"] stores one row per accepted step, independently of
    save_every. Initial/final primitive snapshots always include ghosts.
    """
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

    # Cache is local to this run: never shared with another case or method.
    jacobian_cache = {}
    easy_steps = 0
    history["diagnostics"] = {name: [] for name in ("times", "dt", "cfl", "iterations", "error", "rate", "retries", "jacobian_builds")}
    current_tau = tau

    while t < final_time:
        remaining = final_time - t
        step = min(current_tau, remaining)
        # Absorb only endpoint roundoff, instead of taking a near-zero extra step.
        if remaining - step <= 8*np.finfo(float).eps*max(abs(t), abs(final_time)):
            step = remaining
        q_start = q_current.copy()
        retries = 0
        failed_attempt = False
        builds = 0
        # CFL is diagnostic only; tau remains the time-step control.
        wave_rate = np.max((np.abs(primitives["u"][physical]) + primitives["c"][physical])/domain["dx"][physical])
        if counter % jacobian_step_every == 0:
            jacobian_cache.clear()

        while True:
            if not np.isfinite(step) or step <= 0 or t + step == t:
                raise RuntimeError("Time step is too small to advance time.")

            cfl = step*wave_rate
            # A different step requires a different Newton matrix/factorisation.
            if jacobian_cache.get("tau") != step:
                jacobian_cache.clear()
            reused_matrix = bool(jacobian_cache)
            info = {}
            try:
                if nonlinear_method == "newton":
                    q_candidate = newton_euler(q_start, step, domain, bc_settings, eps_tol=eps_tol, limit=limit, relax_factor=relax_factor, time=t, eps_j=eps_j, q_reference=q_reference, jacobian_every=jacobian_every, reconstruction=reconstruction, pressure_fallback=pressure_fallback, convergence_check=convergence_check, info=info, jacobian_cache=jacobian_cache)
                else:
                    q_candidate = fixed_point_euler(q_start, step, domain, bc_settings, eps_tol=eps_tol, limit=limit, relax_factor=relax_factor, time=t, reconstruction=reconstruction, pressure_fallback=pressure_fallback, convergence_check=convergence_check, info=info)

                validate_state(q_candidate, domain)
                builds += info["jacobian_builds"]
                break

            except (InvalidStateError, NonlinearConvergenceError) as error:
                failed_attempt = True
                builds += info.get("jacobian_builds", 0)
                jacobian_cache.clear()
                easy_steps = 0
                if reused_matrix and info.get("jacobian_builds", 0) == 0:
                    # First distinguish a stale matrix from a genuinely hard step.
                    print(f"\rIE t={t:.3e} | fresh Jacobian retry | CFL={cfl:.3f}".ljust(76), flush=True)
                    print(f"    {error}", flush=True)
                    continue
                smaller_step = step/2

                if retries >= max_retries or smaller_step < min_tau or t + smaller_step == t:
                    raise RuntimeError(f"IE failed at t={t:.6g} s, dt={step:.6g} s after {retries} retries. Last accepted state retained; last failure: {error}") from error

                retries += 1
                print(f"\rIE | t={t:.4e} | retry {retries}/{max_retries} | dt={smaller_step:.2e} | CFL={smaller_step*wave_rate:.3f}".ljust(76), flush=True)
                print(f"    {error}", flush=True)

                step = smaller_step
                # Restart from q_start. Neither time nor history advances.

        # Reached only after a successful nonlinear solve.
        q_current = q_candidate
        current_tau = min(current_tau, step)
        # A one-iteration solve has no ratio but is already easy.
        easy = not failed_attempt and info["iterations"] <= growth_iterations and (info["iterations"] == 1 or info["rate"] <= growth_rate)
        easy_steps = easy_steps + 1 if easy else 0
        if adaptive_tau and easy_steps >= growth_after:
            current_tau = min(tau, 2*current_tau)
            easy_steps = 0

        # Recalculate this after retries, since step may have changed.
        last_step = step >= remaining or t + step >= final_time
        t = final_time if last_step else t + step
        counter += 1

        print(f"\rIE | t={t:.4e} | {iteration_label} OK | CFL={cfl:.3f} | r={info['rate']:.2g}".ljust(76), end="", flush=True)

        diagnostics = {"times": t, "dt": step, "cfl": cfl, "iterations": info["iterations"], "error": info["error"], "rate": info["rate"], "retries": retries, "jacobian_builds": builds}
        for name, value in diagnostics.items():
            history["diagnostics"][name].append(value)

        primitives = get_primitives(q_current)

        if counter % save_every == 0 or last_step:
            history["times"].append(t)

            for name in primitives:
                history["primitives"][name].append(primitives[name].copy())

            min_p = np.min(primitives["p"][physical])
            max_p = np.max(primitives["p"][physical])
            max_speed = np.max(np.abs(primitives["u"][physical]))

            print(f"\rIteration: {counter} | t={t:.3e} | dt={step:.2e} | CFL={cfl:.3f} | min_p={min_p:.3e} | max_p={max_p:.3e} | max_u={max_speed:.3e}", flush=True)

    history["times"] = np.array(history["times"])

    for name in history["primitives"]:
        history["primitives"][name] = np.array(history["primitives"][name])

    for name in history["diagnostics"]:
        history["diagnostics"][name] = np.array(history["diagnostics"][name])

    conservation_check(q_initial, q_current, domain)
    return q_current, history
