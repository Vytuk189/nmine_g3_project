"""Edit the settings below, then run this file to simulate, save, reload and plot.

Requires NumPy and Matplotlib. Run from the project folder: python main.py.
All physical inputs use SI units. Close each plot window to continue.
"""

from construct import build_domain
from domain_rules import uniform_node_density, gaussian_node_density, uniform_area, linear_area, nozzle_area
from conditions import initialize_state, p_rule, u_rule, t_g_rule, t_l_rule, alpha_g_rule
from solvers import solvers
from data_io import save_primitives, load_primitives
from plotting import plot_domain, simple_plot

# WORKFLOW
run_simulation = True       # True: compute and save. False: load existing results and plot only.
show_domain = True         # Show the mesh before computing; close its window to start the solver.
folder_name = "task14rk4"   # New results: data/folder_name, with _1, _2, ... if it already exists.
load_folder = "task14rk4"   # Used only when run_simulation=False; a name under data or a folder path.
solver_name = "rk4"         # Choose "explicit_euler", "rk4" or "implicit_euler".

# DOMAIN (used only for a new simulation)
x_bounds = [0, 10]          # Left and right domain boundaries [m].
n_cells = 500              # Number of physical cells, excluding ghost cells.
n_ghost = (1, 1)           # Number of ghost cells on the left and right.
node_density_rule = uniform_node_density  # Options: uniform_node_density, gaussian_node_density.
area_rule = uniform_area   # Options: uniform_area, linear_area, nozzle_area; area is in m².
# Define additional area and mesh-density functions in domain_rules.py and import them above.

# INITIAL CONDITIONS
# Select functions here; edit their profiles or define new ones in conditions.py.
# Each rule takes x [m] and returns one scalar in the units indicated.
initial_pressure_rule = p_rule         # Pressure [Pa].
initial_velocity_rule = u_rule         # Velocity [m/s].
initial_gas_temperature_rule = t_g_rule # Gas temperature [K].
initial_liquid_temperature_rule = t_l_rule # Liquid temperature [K].
initial_gas_fraction_rule = alpha_g_rule  # Gas volume fraction, strictly between 0 and 1.
initial_integration_intervals = 100    # Trapezoidal subintervals per cell for initial averages.

# BOUNDARIES AND DURATION
final_time = 0.001          # Integrate from t=0 to this time [s].
bc_settings = {"left": "reflective", "right": "reflective"}
# "reflective": closed wall; ghost momentum changes sign.
# "zerograd": ghost values copy the nearest physical cell.
# "periodic": use on both sides, with one ghost cell per side in this implementation.
# "fixed": prescribe the external state in conditions.fixed_state(side);
#          set n_ghost to 0 on that side (for example (0, 1) for a fixed left boundary).
# All other boundary choices require at least one ghost cell on the relevant side.

# SOLVER SETTINGS
# Only the selected block runs. Each block declares exactly its solver's options.
# save_every counts completed time steps, not inner fixed-point iterations.
# The initial and final states are always saved, even between save_every intervals.
if solver_name == "explicit_euler":
    cfl = 0.5              # Recalculate tau = cfl*min(dx/(|u|+c)) at each time step.
    save_every = 50        # Save all primitives and print a summary every this many steps.
    solver_options = {"cfl": cfl, "save_every": save_every}

elif solver_name == "rk4":
    cfl = 0.5              # Same time-step estimate as explicit Euler; four stages per step.
    save_every = 50
    solver_options = {"cfl": cfl, "save_every": save_every}

elif solver_name == "implicit_euler":
    tau = 1e-5             # Prescribed time step [s]; only the final step may be shortened.
    eps_tol = 1e-6         # Required relative fixed-point defect in EVERY physical cell.
    limit = 1000          # Maximum fixed-point iterations for each time step.
    relax_factor = 0.5    # Fraction of the correction applied per iteration: 0 < value <= 1.
    save_every = 50
    # This solver uses relaxed fixed point. A large tau can still prevent convergence.
    # Time and the inner iteration counter overwrite one line; summaries start new lines.
    solver_options = {"tau": tau, "eps_tol": eps_tol, "limit": limit, "relax_factor": relax_factor, "save_every": save_every}

else:
    raise ValueError(f"Unknown solver '{solver_name}'. Choose explicit_euler, rk4 or implicit_euler.")

# PLOTS (used for both new simulations and previously saved results)
# Each entry is (primitive name or two names, requested time [s], x limits [m]).
# The nearest SAVED time is plotted; there is no interpolation in time.
# Two names use separate y axes; the second curve and axis are red.
# Use None for the full x range, or [x_left, x_right] to zoom. Use [] to skip all plots.
# Names include p, u, c, t_g, t_l, alpha_g, alpha_l, rho, rho_g, rho_l, e, e_total,
# e_g, e_l, h_g, h_l, c_g, c_l and k. Values are plotted in their stored SI units.
plot_requests = [("p", 0.001, None), (["u", "c"], 0.001, None), (["t_g", "t_l"], 0.001, None), (["alpha_g", "alpha_l"], 0.001, None), (["rho_g", "rho_l"], 0.001, None)]
# Examples: ("p", 0.0005, [3, 7]) or (["t_g", "t_l"], final_time, None).


# COMPUTATION AND DATA WORKFLOW (normally no edits needed below this line)
if run_simulation:
    domain = build_domain(m=x_bounds, node_density_rule=node_density_rule, area_rule=area_rule, n_cells=n_cells, n_ghost=n_ghost)
    if show_domain: plot_domain(domain, area_rule)

    q_initial = initialize_state(domain, area_rule, initial_pressure_rule, initial_gas_temperature_rule, initial_liquid_temperature_rule, initial_velocity_rule, initial_gas_fraction_rule, n=initial_integration_intervals)
    solver = solvers[solver_name]
    q_final, history = solver(q_initial, domain, final_time, bc_settings, **solver_options)
    results_folder = save_primitives(history, domain, folder_name)
else:
    results_folder = load_folder

# All solution plots use data loaded back from disk (physical cells only).
cell_centres, loaded_history = load_primitives(results_folder)
for primitive_names, plot_time, x_limits in plot_requests:
    simple_plot(loaded_history, cell_centres, primitive_names, time=plot_time, xlim=x_limits)
