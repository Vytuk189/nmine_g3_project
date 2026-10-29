"""Run one fixed-back-pressure case, save results, reload and plot."""

import numpy as np
import matplotlib.pyplot as plt
from construct import build_domain
from domain_rules import gaussian_node_density, g3_nozzle_area
from conditions import initialize_state, g3_initial_p, g3_initial_u, g3_initial_t_g, g3_initial_t_l, g3_initial_alpha_g
from explicit_Euler import explicit_euler
from data_io import create_results_folder, load_primitives
from plotting import plot_domain

# DOMAIN
x_bounds = [0, 10]
n_cells = 100
n_ghost = (1, 1)
area_rule = g3_nozzle_area
node_density_rule = gaussian_node_density #gaussian_node_density or uniform_node_density
show_domain = True
x_throat = 5.0

# RUN SETTINGS
final_time = 2.0          # Simulation duration [s].
back_pressure = 10e5      # Prescribed outlet pressure [Pa].
folder_name = "g3_open_testing_N100_CFL_modified_muscl_test_modified_momentum_cal"


# EXPLICIT EULER
cfl = 0.25
save_every = 500
reconstruction = "muscl"  # "first_order" or "muscl"
pressure_fallback = "gradual"

# BUILD DOMAIN AND INITIAL STATE
domain = build_domain(m=x_bounds, node_density_rule=node_density_rule, area_rule=area_rule, n_cells=n_cells, n_ghost=n_ghost)
if show_domain: plot_domain(domain, area_rule)

q_initial = initialize_state(domain, area_rule, g3_initial_p, g3_initial_t_g, g3_initial_t_l, g3_initial_u, g3_initial_alpha_g)

bc_settings = {
    "left": "g3_inlet",
    "right": {"type": "g3_outlet", "pressure": back_pressure}
}

# CREATE OUTPUT FOLDER AND SAVE SETTINGS
results_folder = create_results_folder(domain, folder_name)

np.savez(results_folder / "run_settings.npz", final_time=final_time, back_pressure=back_pressure,
         cfl=cfl, save_every=save_every, reconstruction=reconstruction, pressure_fallback=pressure_fallback)

# RUN ONCE
# The solver applies boundary conditions and saves snapshots during integration.
q_final, history = explicit_euler(q_initial, domain, final_time, bc_settings,
    cfl=cfl, save_every=save_every, reconstruction=reconstruction, pressure_fallback=pressure_fallback,
    results_folder=results_folder, open_problem=True)


# RELOAD THE ACTUAL OUTPUT FOLDER
cell_centres, loaded_history = load_primitives(folder_name)
times = loaded_history["times"]
primitives = loaded_history["primitives"]

# TIME HISTORIES
# Interpolate velocity and sound speed to the throat.
throat_velocity = np.array([np.interp(x_throat, cell_centres, u) for u in primitives["u"]])
throat_sound_speed = np.array([np.interp(x_throat, cell_centres, c) for c in primitives["c"]])
throat_mach = throat_velocity / throat_sound_speed

# Actual pressure in the last physical cell.
outlet_pressure = primitives["p"][:, -1] / 1e5

# THROAT VELOCITY AND OUTLET PRESSURE OVER TIME
fig, ax1 = plt.subplots()
ax1.plot(times, throat_velocity)
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Throat velocity [m/s]")
ax1.grid(True)

ax2 = ax1.twinx()
ax2.plot(times, outlet_pressure, color="red")
ax2.set_ylabel("Outlet-cell pressure [bar]", color="red")
ax2.tick_params(axis="y", colors="red")

fig.tight_layout()
plt.show()

# THROAT MACH NUMBER AND OUTLET PRESSURE OVER TIME
fig, ax1 = plt.subplots()
ax1.plot(times, throat_mach)
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Throat Mach number [-]")
ax1.grid(True)

ax2 = ax1.twinx()
ax2.plot(times, outlet_pressure, color="red")
ax2.set_ylabel("Outlet-cell pressure [bar]", color="red")
ax2.tick_params(axis="y", colors="red")

fig.tight_layout()
plt.show()

# FINAL SOLUTION OVER POSITION
final_pressure = primitives["p"][-1] / 1e5
final_velocity = primitives["u"][-1]
final_mach = final_velocity / primitives["c"][-1]

plt.figure()
plt.plot(cell_centres, final_pressure)
plt.xlabel("Position [m]")
plt.ylabel("Pressure [bar]")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(cell_centres, final_velocity)
plt.xlabel("Position [m]")
plt.ylabel("Velocity [m/s]")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(cell_centres, final_mach)
plt.axhline(1.0, color="black", linestyle="--", label="Mach 1")
plt.xlabel("Position [m]")
plt.ylabel("Mach number [-]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

labels = ["Gas partial density", "Liquid partial density", "Momentum", "Energy", "Gas volume fraction"]

plt.figure()
for k, label in enumerate(labels):
    plt.semilogy(loaded_history["times"], loaded_history["rez"][:, k], label=label)
plt.xlabel("Time [s]")
plt.ylabel("Volume-weighted RMS change per step")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

