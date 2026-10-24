"""Runge-Kutta Fourth order integration with CFL take from time-step calculation."""

import numpy as np
from conditions import apply_boundary_conditions
from solver_functions import get_primitives, compute_residual
from miscellaneous import conservation_check


def get_time_step(u, c, dx, cfl=0.5):
    """Return the smallest cell time step allowed by the CFL condition."""
    return cfl * np.min(dx / (np.abs(u) + c))


def rk4(q_initial, domain, final_time, bc_settings, cfl=0.5, save_every=100):

    if not isinstance(save_every, (int, np.integer)) or save_every <= 0: raise ValueError("save_every must be a positive integer.")
    if not np.isfinite(final_time) or final_time < 0: raise ValueError("final_time must be finite and nonnegative.")
    if not np.isfinite(cfl) or cfl <= 0: raise ValueError("cfl must be finite and positive.")

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

    while t < final_time:
        print(f"\rCalculating with RK4 {'.' * (counter % 3 + 1):<3}", end="", flush=True)
        #Assuming same CFL conditions holds for RK4 and Euler - have to check this
        tau = get_time_step(primitives["u"][physical], primitives["c"][physical], domain["dx"][physical], cfl)
        if not np.isfinite(tau) or tau <= 0 or t + tau == t: raise ValueError("Invalid time step: time cannot advance.")
        remaining = final_time - t
        last_step = tau >= remaining
        tau = min(tau, remaining)
        # Change this into a inner loop for k1, k2, k3, k4 in RK4
        # k1
        k1 = compute_residual(q_current, domain)
        q_1 = q_current.copy()
        q_1[:, physical] = q_current[:, physical] + 0.5*tau*k1
        q_1 = apply_boundary_conditions(domain, q_1, bc_settings["left"], "left")
        q_1 = apply_boundary_conditions(domain, q_1, bc_settings["right"], "right")
        # k2
        k2 = compute_residual(q_1, domain)
        q_2 = q_current.copy()
        q_2[:, physical] = q_2[:, physical] + 0.5*tau*k2
        q_2 = apply_boundary_conditions(domain, q_2, bc_settings["left"], "left")
        q_2 = apply_boundary_conditions(domain, q_2, bc_settings["right"], "right")
        # k3
        k3 = compute_residual(q_2, domain)
        q_3 = q_current.copy()
        q_3[:, physical] = q_3[:, physical] + tau*k3
        q_3 = apply_boundary_conditions(domain, q_3, bc_settings["left"], "left")
        q_3 = apply_boundary_conditions(domain, q_3, bc_settings["right"], "right")
        # k4
        k4 = compute_residual(q_3, domain)
        # Final
        q_current[:, physical] = q_current[:, physical] + tau/6*(k1+2*k2+2*k3+k4)
        q_current = apply_boundary_conditions(domain, q_current, bc_settings["left"], "left")
        q_current = apply_boundary_conditions(domain, q_current, bc_settings["right"], "right")
        
            
        t = final_time if last_step else t + tau
        counter += 1

        # primitive update
        primitives = get_primitives(q_current)

        if counter % save_every == 0 or t >= final_time:
            history["times"].append(t)

            for name in primitives:
                history["primitives"][name].append(primitives[name].copy())

            max_p = np.max(primitives["p"][physical])
            max_speed = np.max(abs(primitives["u"][physical]))

            print(f"\rIteration: {counter}, t: {t:.6g} s, tau: {tau:6g}, max p: {max_p:.6g} Pa, max |u|: {max_speed:.6g} m/s")

    # Each primitive array has shape (number of saved times, N_total).
    history["times"] = np.array(history["times"])

    for name in history["primitives"]:
        history["primitives"][name] = np.array(history["primitives"][name])

    mass_error, energy_error = conservation_check(q_initial, q_current, domain)

    return q_current, history
