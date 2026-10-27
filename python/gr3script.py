"""Configure a case, solve it, save the results, reload them and plot.

Requires NumPy, SciPy and Matplotlib.
All physical inputs use SI units.
"""

from construct import build_domain
from domain_rules import gaussian_node_density, g3_nozzle_area
from conditions import initialize_state, g3_initial_p, g3_initial_u, g3_initial_t_g, g3_initial_t_l, g3_initial_alpha_g, apply_boundary_conditions
from solver_functions import get_primitives
from solvers import solvers
from data_io import save_primitives, load_primitives, create_results_folder
from plotting import plot_domain, simple_plot
import matplotlib.pyplot as plt


# DOMAIN
x_bounds = [0, 10]          # Domain boundaries [m].
n_cells = 200              # Physical cells, excluding ghosts.
n_ghost = (1, 1)
node_density_rule = gaussian_node_density  # Or gaussian_node_density.
area_rule = g3_nozzle_area   # Or linear_area, nozzle_area.

domain = build_domain(m=x_bounds, node_density_rule=node_density_rule, area_rule=area_rule, n_cells=n_cells, n_ghost=n_ghost)
plot_domain(domain, area_rule)

# INITIAL CONDITIONS
q_initial = initialize_state(domain, area_rule, g3_initial_p, g3_initial_t_g, g3_initial_t_l, g3_initial_u, g3_initial_alpha_g)

# BOUNDARY CONDITIONS
bc_settings = {"left": "g3_inlet", "right": "g3_outlet"}
q_initial = apply_boundary_conditions(domain, q_initial, bc_settings["left"], "left")
q_initial = apply_boundary_conditions(domain, q_initial, bc_settings["right"], "right")

initial_primitives = get_primitives(q_initial)
initial_pressures = initial_primitives["p"]
cell_centers = domain["x_centres"]
plt.plot(cell_centers, initial_pressures)
plt.xlabel("Position (m)")
plt.ylabel("Pressure (Pa)")
plt.title("Initial Pressure Distribution")
plt.show()
initial_u = initial_primitives["u"]
plt.plot(cell_centers, initial_u)
plt.xlabel("Position (m)")
plt.ylabel("Velocity (m/s)")
plt.title("Initial Velocity Distribution")
plt.show()
print(initial_pressures[0])


# Initial testing run
# SOLVER SETTINGS
solver_name = "explicit_euler"
final_time = 2           # Simulation duration [s]; an initial trial, not a steady-state guarantee.
cfl = 0.5 # Time step recalculated from the current velocity and sound speed.
save_every = 100           # Save primitives and print diagnostics every 100 time steps.
folder_name = "g3_explicit_initial"

# CREATE OUTPUT FOLDER BEFORE STARTING
results_folder = create_results_folder(domain, folder_name)

solver_options = {"cfl": cfl, "save_every": save_every, "results_folder": results_folder}

# RUN: initial state and subsequent snapshots are written during integration.
solver = solvers[solver_name]
q_final, history = solver(q_initial, domain, final_time, bc_settings, **solver_options)

# RELOAD
cell_centres, loaded_history = load_primitives("g3_explicit_initial")

# PLOT OUTLET VELOCITY OVER TIME
# Loaded data contains physical cells only; -1 selects the last physical cell.
times = loaded_history["times"]
outlet_velocity = loaded_history["primitives"]["u"][:, -1]

plt.figure()
plt.plot(times, outlet_velocity)
plt.xlabel("Time [s]")
plt.ylabel("Outlet velocity [m/s]")
plt.grid(True)
plt.tight_layout()
plt.show()