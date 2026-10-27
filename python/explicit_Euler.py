"""Explicit Euler integration and its CFL time-step calculation."""

import numpy as np
from conditions import apply_boundary_conditions
from solver_functions import get_primitives, compute_residual
from miscellaneous import conservation_check
from data_io import save_snapshot


def get_time_step(u, c, dx, cfl=0.5):
    """Return the smallest cell time step allowed by the CFL condition."""
    return cfl * np.min(dx / (np.abs(u) + c))


def explicit_euler(q_initial, domain, final_time, bc_settings, cfl=0.5, save_every=100, reconstruction="muscl", pressure_fallback="first_order", results_folder=None):


    # final_time >= 0, cfl > 0 and save_every is a positive integer.
    # q_initial: (5, n_total), with at least one ghost per side.
    # reconstruction: "muscl"/"first_order"; fallback: "first_order"/"gradual".
    t = 0.0
    counter = 0
    q_current = q_initial.copy()
    physical = domain["physical_ind"]
    q_current = apply_boundary_conditions(domain, q_current, bc_settings["left"], "left")
    q_current = apply_boundary_conditions(domain, q_current, bc_settings["right"], "right")
    # Save the initial state.
    primitives = get_primitives(q_current)

    history = {"times": [t],
               "primitives": {} }

    for name in primitives:
        history["primitives"][name] = [primitives[name].copy()]
        
    if results_folder is not None:
        save_snapshot(primitives, t, domain, results_folder, counter)

    while t < final_time:
        print(f"\rCalculating with Euler{'.' * (counter % 3 + 1):<3}", end="", flush=True)
        tau = get_time_step(primitives["u"][physical], primitives["c"][physical], domain["dx"][physical], cfl)
        if not np.isfinite(tau) or tau <= 0 or t + tau == t: raise ValueError("Invalid time step: time cannot advance.")
        remaining = final_time - t
        last_step = tau >= remaining
        tau = min(tau, remaining)
        rate = compute_residual(q_current, domain, reconstruction=reconstruction, pressure_fallback=pressure_fallback)
        q_next = q_current.copy()
        q_next[:, physical] = q_current[:, physical] + tau*rate
        q_next = apply_boundary_conditions(domain, q_next, bc_settings["left"], "left")
        q_next = apply_boundary_conditions(domain, q_next, bc_settings["right"], "right")
        q_current = q_next
        t = final_time if last_step else t + tau
        counter += 1

        # primitive update
        primitives = get_primitives(q_current)

        if counter % save_every == 0 or t >= final_time:
            history["times"].append(t)

            for name in primitives:
                history["primitives"][name].append(primitives[name].copy())
                
            if results_folder is not None:
                save_snapshot(primitives, t, domain, results_folder, counter)

            min_p = np.min(primitives["p"][physical])
            max_p = np.max(primitives["p"][physical])
            max_speed = np.max(abs(primitives["u"][physical]))

            print(f"\rIteration: {counter} | t={t:.3e} | dt={tau:.2e} | min_p={min_p:.3e} | max_p={max_p:.3e} | max_u={max_speed:.3e}", flush=True)

    # Each primitive array has shape (number of saved times, N_total).
    history["times"] = np.array(history["times"])

    for name in history["primitives"]:
        history["primitives"][name] = np.array(history["primitives"][name])

    mass_error, energy_error = conservation_check(q_initial, q_current, domain)
    return q_current, history
