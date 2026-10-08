"""Case settings and the complete simulation/save/load/plot workflow."""

from construct import build_domain
from domain_rules import uniform_node_density, uniform_area
from conditions import initialize_state, p_rule, u_rule, t_g_rule, t_l_rule, alpha_g_rule
from solver_functions import primitive_to_q
from solvers import solvers
from data_io import save_primitives, load_primitives
from plotting import plot_domain, simple_plot

# Workflow settings
run_simulation = True
show_domain = True
folder_name = "task15"
solver_name = "explicit_euler"

# Domain inputs (define new rules in domain_rules.py, then import them above)
x_bounds = [0, 10]
n_cells = 500
n_ghost = (1, 1)
node_density_rule = uniform_node_density
area_rule = uniform_area

# Solver inputs (edit the initial profiles and fixed states in conditions.py)
final_time = 0.001
cfl = 0.1
save_every = 50
bc_settings = {"left": "reflective", "right": "reflective"}

# Plot inputs: (primitive identifier(s), requested time [s], optional x limits)
plot_requests = [("p", 0.001, [2, 8]), (["u", "c"], 0.001, None), (["t_g", "t_l"], 0.0005, None)]


if run_simulation:
    domain = build_domain(m=x_bounds, node_density_rule=node_density_rule, area_rule=area_rule, n_cells=n_cells, n_ghost=n_ghost)
    if show_domain: plot_domain(domain, area_rule)

    q_initial = initialize_state(domain, area_rule, p_rule, t_g_rule, t_l_rule, u_rule, alpha_g_rule, primitive_to_q)
    solver = solvers[solver_name]
    q_final, history = solver(q_initial, domain, final_time, bc_settings, cfl=cfl, save_every=save_every)
    results_folder = save_primitives(history, domain, folder_name)
else:
    results_folder = folder_name

# All solution plots use data loaded back from disk.
cell_centres, loaded_history = load_primitives(results_folder)
for primitive_names, plot_time, x_limits in plot_requests:
    simple_plot(loaded_history, cell_centres, primitive_names, time=plot_time, xlim=x_limits)
