"""Explicit Euler integration with saved rate-of-change diagnostics."""

import numpy as np
from conditions import apply_boundary_conditions
from solver_functions import get_primitives, compute_residual
from miscellaneous import conservation_check, open_conservation_check
from data_io import save_snapshot


def get_time_step(u, c, dx, cfl=0.5):
    """Return the smallest cell time step allowed by the CFL condition."""
    return cfl*np.min(dx / (np.abs(u) + c))


def explicit_euler(q_initial, domain, final_time, bc_settings, cfl=0.5, save_every=100, reconstruction="muscl", pressure_fallback="first_order", results_folder=None, start_time=0.0, start_iteration=0, save_initial=True, open_problem=False):
    """Integrate from start_time to absolute final_time.

    save_every counts time steps. Initial and final states are also saved.
    results_folder enables writing snapshots during the calculation.
    open_problem enables the open-domain conservation balance.

    Rez compares consecutive states using volume-weighted RMS changes,
    divided by the actual time step and the previous component maximum.
    The reference time is fixed at 1 second. This diagnostic does not stop
    the calculation.

    For continuation, pass the previous final state and last_iteration.
    Use save_initial=False to avoid saving the shared endpoint twice.
    """
    t = start_time
    counter = start_iteration
    q_current = q_initial.copy()

    physical = domain["physical_ind"]
    volumes = domain["volumes"][physical]
    total_volume = np.sum(volumes)

    q_current = apply_boundary_conditions(domain, q_current, bc_settings["left"], "left")
    q_current = apply_boundary_conditions(domain, q_current, bc_settings["right"], "right")

    primitives = get_primitives(q_current)

    # The initial state has no preceding time step, so its Rez is undefined.
    history = {"times": [t], "primitives": {}, "rez": [np.full(5, np.nan)]}

    for name in primitives:
        history["primitives"][name] = [primitives[name].copy()]

    if results_folder is not None and save_initial:
        save_snapshot(primitives, t, domain, results_folder, counter, rez=history["rez"][-1])

    if open_problem:
        left_transport = np.zeros(3)
        right_transport = np.zeros(3)

    while t < final_time:
        print(f"\rCalculating with Euler{'.' * (counter % 3 + 1):<3}", end="", flush=True)

        tau = get_time_step(primitives["u"][physical], primitives["c"][physical], domain["dx"][physical], cfl)

        remaining = final_time - t
        last_step = tau >= remaining
        tau = min(tau, remaining)

        if not np.isfinite(tau) or tau <= 0 or t + tau == t:
            raise ValueError("Invalid time step: time cannot advance.")

        if open_problem:
            rate, left_flux, right_flux = compute_residual(q_current, domain, reconstruction=reconstruction, pressure_fallback=pressure_fallback, return_boundary_fluxes=True)
        else:
            rate = compute_residual(q_current, domain, reconstruction=reconstruction, pressure_fallback=pressure_fallback)

        q_next = q_current.copy()
        q_next[:, physical] = q_current[:, physical] + tau*rate

        q_next = apply_boundary_conditions(domain, q_next, bc_settings["left"], "left")
        q_next = apply_boundary_conditions(domain, q_next, bc_settings["right"], "right")

        save_due = (counter + 1) % save_every == 0 or last_step

        if save_due:
            rez = np.zeros(5)

            for k in range(5):
                # Change between consecutive time steps, physical cells only.
                difference = q_next[k, physical] - q_current[k, physical]

                # Volume-weighted RMS difference: no component or time scaling.
                rez[k] = np.sqrt(np.sum(volumes*difference**2) / total_volume)

        if open_problem:
            # Explicit Euler uses boundary fluxes from the start of the step.
            left_transport += tau*left_flux[[0, 1, 3]]
            right_transport += tau*right_flux[[0, 1, 3]]

        q_current = q_next
        t = final_time if last_step else t + tau
        counter += 1

        primitives = get_primitives(q_current)

        if save_due:
            history["times"].append(t)
            history["rez"].append(rez.copy())

            for name in primitives:
                history["primitives"][name].append(primitives[name].copy())

            if results_folder is not None:
                save_snapshot(primitives, t, domain, results_folder, counter, rez=rez)

            min_p = np.min(primitives["p"][physical])
            max_p = np.max(primitives["p"][physical])
            max_speed = np.max(np.abs(primitives["u"][physical]))
            max_Mach = np.max(np.abs(primitives["u"][physical] / primitives["c"][physical]))

            print(f"\rIteration: {counter} | t={t:.3e} | dt={tau:.2e} | min_p={min_p:.3e} | max_p={max_p:.3e} | max_u={max_speed:.3e} | max_Mach={max_Mach:.3e}", flush=True, )
            print("Rez [gas, liquid, momentum, energy, alpha]: " + " ".join(f"{value:.3e}" for value in rez), flush=True)

    history["times"] = np.array(history["times"])
    history["rez"] = np.array(history["rez"])

    for name in history["primitives"]:
        history["primitives"][name] = np.array(history["primitives"][name])

    history["last_iteration"] = counter

    if open_problem:
        history["conservation"] = open_conservation_check(q_initial, q_current, domain, left_transport, right_transport, results_folder=results_folder)
    else:
        conservation_check(q_initial, q_current, domain)

    return q_current, history