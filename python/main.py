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
folder_name = "task15cnfpi"    # Existing names receive suffixes _1, _2, etc.
load_folder = "task14cn"    # Used when run_simulation=False.
solver_name = "crank_nicolson"  # explicit_euler, rk4, implicit_euler, crank_nicolson.


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
initial_integration_intervals = 100  # Midpoint subintervals per physical cell.


# BOUNDARIES AND DURATION
final_time = 0.001
bc_settings = {"left": "reflective", "right": "reflective"}

# reflective: wall; reflected ghost momentum changes sign.
# zerograd: ghosts copy the nearest physical cell.
# periodic: use on both sides; multiple ghost cells wrap successive cells.
# fixed: define the external state in conditions.fixed_state(side);
#        use at least one ghost cell on that side, as for all boundaries.


# SOLVER SETTINGS
# Use the exact option strings below. Counts (save_every, limit, Jacobian/growth
# intervals) must be positive integers. tau, cfl, eps_tol and eps_j must be positive.
# min_tau must be <= tau; max_retries is a nonnegative integer; final_time >= 0.
# Q has shape (5, total cells); build_domain supplies the expected indexing.
# save_every counts completed physical time steps, not nonlinear iterations.
# The initial and final states are always included in history.
reconstruction = "muscl"  # Options: "muscl" or "first_order"; applies to all solvers.
pressure_fallback = "first_order"  # MUSCL pressure fallback: "first_order" or "gradual".
# Both use the existing hard-coded 5000 Pa threshold.
# This setting has no effect when reconstruction="first_order".
convergence_check = "teacher"  # Implicit solvers only: "scaled" or "teacher".
# scaled: unrelaxed component-scaled corrections in physical cells (current check).
# teacher: norm(q_next-q_m, 2)/norm(q_next, 2), full matrices including ghosts.

if solver_name == "explicit_euler":
    cfl = 0.1
    save_every = 50

    solver_options = {"cfl": cfl, "save_every": save_every}

elif solver_name == "rk4":
    cfl = 0.1
    save_every = 50

    solver_options = {"cfl": cfl, "save_every": save_every}

elif solver_name == "implicit_euler":
    tau = 1e-6
    save_every = 50
    eps_tol = 1e-16          # Tolerance for the selected convergence check.
    limit = 200
    min_tau = 1e-12        # Smallest permitted step after a failed attempt.
    max_retries = 12       # Maximum halvings for one physical time step.

    nonlinear_method = "newton"  # Options: "newton" or "fpi".
    relax_factor = 1.0      # (0, 1]; use a smaller value to damp nonlinear updates.
    eps_j = 1e-6
    q_reference = None     # Or five positive scales [gas mass, liquid mass, momentum, energy, alpha].
    jacobian_every = 20     # Inner Newton iterations: rebuild on 1, 11, 21, ...
    jacobian_step_every = 10 # Outer steps: 1 refreshes each step; e.g. 5 reuses between steps 1, 6, 11, ...
    # Both Jacobian settings affect Newton only. Inner rebuilds still apply.
    # New dt or a failed attempt forces a fresh matrix regardless of the interval.
    adaptive_tau = True   # Default: once halved, retain the smaller step.
    # True: recover toward the original tau after consecutive easy solves.
    growth_after = 3        # Positive integer: required consecutive easy steps.
    growth_iterations = 8   # Positive integer: an easy solve uses at most this many iterations.
    growth_rate = 0.5       # (0, 1): recent error contraction factor must be <= this.
    # Recovery controls nonlinear difficulty, not time-discretisation accuracy.

    solver_options = {"tau": tau, "save_every": save_every, "eps_tol": eps_tol, "limit": limit, "relax_factor": relax_factor, "nonlinear_method": nonlinear_method, "eps_j": eps_j, "q_reference": q_reference, "jacobian_every": jacobian_every, "convergence_check": convergence_check}
    solver_options["min_tau"] = min_tau
    solver_options["max_retries"] = max_retries
    solver_options.update(jacobian_step_every=jacobian_step_every, adaptive_tau=adaptive_tau, growth_after=growth_after, growth_iterations=growth_iterations, growth_rate=growth_rate)

elif solver_name == "crank_nicolson":
    tau = 1e-6
    save_every = 50
    eps_tol = 1e-16          # Tolerance for the selected convergence check.
    limit = 200
    min_tau = 1e-12        # Smallest permitted step after a failed attempt.
    max_retries = 12       # Maximum halvings for one physical time step.

    nonlinear_method = "newton"
    relax_factor = 1.0      # (0, 1]; full updates by default.
    eps_j = 1e-6
    q_reference = None     # Or five positive scales [gas mass, liquid mass, momentum, energy, alpha].
    jacobian_every = 20     # Inner Newton iterations: rebuild on 1, 11, 21, ...

    jacobian_step_every = 10 # Outer steps: 1 refreshes each step; e.g. 5 reuses between steps 1, 6, 11, ...
    # Both Jacobian settings affect Newton only. Inner rebuilds still apply.
    # New dt or a failed attempt forces a fresh matrix regardless of the interval.
    adaptive_tau = True   # Default: once halved, retain the smaller step.
    # True: recover toward the original tau after consecutive easy solves.
    growth_after = 3        # Positive integer: required consecutive easy steps.
    growth_iterations = 8   # Positive integer: an easy solve uses at most this many iterations.
    growth_rate = 0.5       # (0, 1): recent error contraction factor must be <= this.
    # Recovery controls nonlinear difficulty, not time-discretisation accuracy.

    solver_options = {"tau": tau, "save_every": save_every, "eps_tol": eps_tol, "limit": limit, "relax_factor": relax_factor, "nonlinear_method": nonlinear_method, "eps_j": eps_j, "q_reference": q_reference, "jacobian_every": jacobian_every, "convergence_check": convergence_check}
    solver_options["min_tau"] = min_tau
    solver_options["max_retries"] = max_retries
    solver_options.update(jacobian_step_every=jacobian_step_every, adaptive_tau=adaptive_tau, growth_after=growth_after, growth_iterations=growth_iterations, growth_rate=growth_rate)


solver_options["reconstruction"] = reconstruction
solver_options["pressure_fallback"] = pressure_fallback


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
