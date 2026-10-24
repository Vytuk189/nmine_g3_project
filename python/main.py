"""Configure a case, solve it, save the results, reload them and plot.

Requires NumPy, SciPy and Matplotlib.
Run from the python directory: python main.py.
All physical inputs use SI units.
"""

from construct import build_domain
from domain_rules import uniform_node_density, gaussian_node_density, uniform_area, linear_area, nozzle_area
from conditions import initialize_state, p_rule, u_rule, t_g_rule, t_l_rule, alpha_g_rule
from solvers import solvers
from data_io import save_primitives, load_primitives
from plotting import plot_domain, simple_plot


# WORKFLOW
run_simulation = True       # False: load an existing result folder and plot only.
show_domain = True         # Close the domain plot to begin the computation.
folder_name = "task14ienewton"    # Existing names receive suffixes _1, _2, etc.
load_folder = "task14cn"    # Used when run_simulation=False.
solver_name = "implicit_euler"  # explicit_euler, rk4, implicit_euler, crank_nicolson.


# DOMAIN
x_bounds = [0, 10]          # Domain boundaries [m].
n_cells = 500              # Physical cells, excluding ghosts.
n_ghost = (1, 1)
node_density_rule = uniform_node_density  # Or gaussian_node_density.
area_rule = uniform_area   # Or linear_area, nozzle_area.


# INITIAL CONDITIONS
# Each rule takes x [m] and returns the corresponding scalar.
initial_pressure_rule = p_rule
initial_velocity_rule = u_rule
initial_gas_temperature_rule = t_g_rule
initial_liquid_temperature_rule = t_l_rule
initial_gas_fraction_rule = alpha_g_rule
initial_integration_intervals = 100


# BOUNDARIES AND DURATION
final_time = 0.001
bc_settings = {"left": "reflective", "right": "reflective"}

# reflective: wall; reflected ghost momentum changes sign.
# zerograd: ghosts copy the nearest physical cell.
# periodic: use on both sides, with one ghost cell per side.
# fixed: define the external state in conditions.fixed_state(side);
#        use zero ghost cells on that side.


# SOLVER SETTINGS
# save_every counts completed physical time steps, not nonlinear iterations.
# The initial and final states are always included in history.

if solver_name == "explicit_euler":
    cfl = 0.5
    save_every = 50

    solver_options = {"cfl": cfl, "save_every": save_every}

elif solver_name == "rk4":
    cfl = 0.5
    save_every = 50

    solver_options = {"cfl": cfl, "save_every": save_every}

elif solver_name == "implicit_euler":
    tau = 1e-5
    save_every = 50
    eps_tol = 1e-6
    limit = 1000

    nonlinear_method = "newton"  # Options: "newton" or "fpi".
    relax_factor = None         # None: 1.0 for Newton, 0.25 for FPI.
    eps_j = 1e-6
    q_reference = None
    jacobian_every = 10          # Rebuild Jacobianon every nth Newton iteration
    solver_options = {"tau": tau, "save_every": save_every, "eps_tol": eps_tol, "limit": limit, "relax_factor": relax_factor, "nonlinear_method": nonlinear_method, "eps_j": eps_j, "q_reference": q_reference, "jacobian_every": jacobian_every}

elif solver_name == "crank_nicolson":
    tau = 1e-5
    save_every = 50
    eps_tol = 1e-6
    limit = 1000

    nonlinear_method = "newton"
    relax_factor = None
    eps_j = 1e-6
    q_reference = None
    jacobian_every = 10

    solver_options = {"tau": tau, "save_every": save_every, "eps_tol": eps_tol, "limit": limit, "relax_factor": relax_factor, "nonlinear_method": nonlinear_method, "eps_j": eps_j, "q_reference": q_reference, "jacobian_every": jacobian_every}
else:
    raise ValueError(f"Unknown solver '{solver_name}'. Choose explicit_euler, rk4, implicit_euler or crank_nicolson.")


# PLOTS
# Each entry: (primitive name or two names, requested time [s], x limits [m]).
# The nearest saved time is selected; there is no time interpolation.
# Two names use separate y axes, with the second curve and axis red.
# Use None for the full x range, [x_left, x_right] to zoom, or [] to skip plots.
# Values are plotted in their saved SI units.

plot_requests = [("p", 0.001, None), (["u", "c"], 0.001, None), (["t_g", "t_l"], 0.001, None), (["alpha_g", "alpha_l"], 0.001, None), (["rho_g", "rho_l"], 0.001, None)]


# COMPUTATION AND DATA WORKFLOW
if run_simulation:
    domain = build_domain(m=x_bounds, node_density_rule=node_density_rule, area_rule=area_rule, n_cells=n_cells, n_ghost=n_ghost)

    if show_domain:
        plot_domain(domain, area_rule)

    q_initial = initialize_state(domain, area_rule, initial_pressure_rule, initial_gas_temperature_rule, initial_liquid_temperature_rule, initial_velocity_rule, initial_gas_fraction_rule, n=initial_integration_intervals)

    solver = solvers[solver_name]
    q_final, history = solver(q_initial, domain, final_time, bc_settings, **solver_options)

    results_folder = save_primitives(history, domain, folder_name)

else:
    results_folder = load_folder


# Solution plots use data loaded from disk, containing physical cells only.
cell_centres, loaded_history = load_primitives(results_folder)

for primitive_names, plot_time, x_limits in plot_requests:
    simple_plot(loaded_history, cell_centres, primitive_names, time=plot_time, xlim=x_limits)