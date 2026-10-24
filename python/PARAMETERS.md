# Configuring and running the solver

Keep the Python files together, open a terminal in that folder and run `python main.py`. NumPy and Matplotlib are required. Edit the settings in `main.py`; the bottom of the file performs the shared build → initialise → solve → save → reload → plot workflow. Close each plot window to continue.

## Which settings belong to which solver?

All three solvers receive `q_initial`, `domain`, `final_time`, `bc_settings` and `save_every`.

| Solver | Additional settings declared in its main.py block |
| --- | --- |
| `explicit_euler` | `cfl`: recalculates the time step from current velocity, sound speed and cell width. |
| `rk4` | `cfl`: the same time-step estimate, with four RK stages per time step. |
| `implicit_euler` | Required `tau` [s]; `eps_tol`, `limit`, `relax_factor` for relaxed fixed-point iteration. No CFL argument. |

`save_every` counts completed time steps. All primitive variables are saved at these intervals, plus the initial and final states. Implicit Euler's `limit` counts inner nonlinear iterations and is unrelated to `save_every`.

The default selected solver remains RK4. To switch, change `solver_name`, then edit its block. Implicit Euler shortens its prescribed `tau` only for the final step; nonlinear convergence can still require a smaller time step.

## How the parameters flow

| Call | What the caller supplies | What is handled internally |
| --- | --- | --- |
| `build_domain(m, node_density_rule, area_rule, n_cells, n_ghost)` | Geometry, rule functions and mesh size | Faces, centres, volumes and indices |
| `initialize_state(domain, area_rule, p_rule, t_g_rule, t_l_rule, u_rule, alpha_g_rule, n=100)` | Mesh, initial profiles and quadrature resolution | Calls `primitive_to_q` directly |
| `implicit_euler(q_initial, domain, final_time, bc_settings, tau, ...)` | Initial state, common integration settings and its solver options | Calls `fixed_point_euler` directly for each step |
| `fixed_point_euler(q_initial, tau, domain, bc_settings, eps_tol, limit, relax_factor, time)` | One step's starting state and nonlinear controls | Residual and boundary updates; `time` is only a progress label |
| `compute_residual(q, domain)` | Current state including populated ghosts and mesh | MUSCL reconstruction, Rusanov flux and source terms |
| `rusanov_flux(q_interface_left, q_interface_right)` | Two reconstructed interface states | Calls `q_to_primitive` directly |
| `get_primitives(q)` | Conservative state | Calls `q_to_primitive` directly |
| `save_primitives(history, domain, folder_name)` | Full history and mesh | Selects physical cells and creates a unique output folder |
| `load_primitives(folder_name)` | Saved folder name or path | Returns physical-cell coordinates and history |
| `simple_plot(history, cell_centres, names, time, xlim=None)` | Loaded history, requested quantity/time and optional limits | Selects the nearest saved time and creates one or two y axes |

Rule functions remain parameters because selecting geometry and initial profiles is part of configuring a case.

The generic `nonlinear_solver` parameter was removed from implicit Euler: it now explicitly uses the backward Euler fixed-point solver. The existing `fixed_point_CN` helper is retained, but is not a registered time integrator and is not interchangeable with the backward Euler solver; it still returns a rate, as in the uploaded framework.

## Data and plotting

- In-memory conservative state: `(5, number of cells including ghosts)`.
- In-memory history: `history['times']` and `history['primitives'][name]`, with primitive arrays shaped `(saved times, cells including ghosts)`.
- Saved and reloaded primitive arrays contain only physical cells, with rows representing time and columns representing space.
- New runs save under `data/folder_name` relative to the working directory, adding suffixes if necessary.
- To plot an existing run, set `run_simulation=False` and select `load_folder`. The mesh and initial state are not rebuilt.
- `plot_requests` selects quantities, requested times and optional x limits. Two quantities use separate axes, with the second in red. Stored SI units are used without conversion.

