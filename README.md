# One-dimensional two-phase finite-volume solver

A Python framework for one-dimensional gas–liquid flow with a stiffened-gas equation of state, MUSCL reconstruction and Rusanov numerical fluxes. The state uses a common pressure and velocity, separate phase temperatures, and a transported gas volume fraction. Cross-sectional area and mesh density can vary along the domain.

The normal user workflow is to **configure `python/main.py`, run a solver, save its results, and plot the saved data**. You can also load existing results in a separate script without running a simulation.

Four time integrators are registered: explicit Euler, classical RK4, backward Euler and Crank–Nicolson. The two implicit methods use Newton iteration by default; relaxed fixed-point iteration is selected with `nonlinear_method="fpi"`. This README accompanies the updated solver scripts, including Jacobian reuse, component-scaled convergence, selectable reconstruction and abbreviated progress printing. Apply those scripts together with this README; documentation alone does not add these features to an older checkout.

## Contents

- [Quick start](#quick-start)
- [Project structure](#project-structure)
- [Configure a run in main.py](#configure-a-run-in-mainpy)
- [Data structures and array shapes](#data-structures-and-array-shapes)
- [Saving and loading results](#saving-and-loading-results)
- [Plotting and extracting data](#plotting-and-extracting-data)
- [Main function reference](#main-function-reference)
- [Numerical method and convergence](#numerical-method-and-convergence)
- [Common mistakes and limitations](#common-mistakes-and-limitations)

## Quick start

Install Python, NumPy, SciPy and Matplotlib. The Python source files are inside the **`python` subdirectory**, not the repository root.

```bash
git clone https://github.com/Vytuk189/nmine_g3_project.git
cd nmine_g3_project/python
python -m pip install numpy scipy matplotlib
python main.py
```

Close each Matplotlib window to continue. With `show_domain=True`, the mesh plot opens before the computation starts.

Run from the `python` directory consistently. Saving and loading use the **current working directory**, so running `python python/main.py` from the repository root puts new results in a different `data` directory.

To inspect an existing run without computing again, edit these settings in `main.py`:

```python
run_simulation = False
load_folder = "task14cn"  # The actual saved folder name, including any suffix.
```

Then run `python main.py`. The solution plots are produced from disk. In this workflow, `show_domain` applies only to new simulations because the domain itself is not loaded from disk.

## Project structure

All paths below are relative to the repository root.

| File or directory | Role |
| --- | --- |
| [`python/main.py`](python/main.py) | User settings and the complete build → initialise → solve → save → reload → plot workflow. |
| [`python/construct.py`](python/construct.py) | Builds cell faces, centres, geometry arrays and physical/ghost indices. Integrates cell volumes. |
| [`python/domain_rules.py`](python/domain_rules.py) | Selectable mesh-density and cross-sectional-area functions. |
| [`python/conditions.py`](python/conditions.py) | Initial profiles, initial cell averages, fixed external states and ghost-cell boundary updates. |
| [`python/solver_functions.py`](python/solver_functions.py) | Equation of state, state conversions, primitive extraction, MUSCL, Rusanov flux and spatial residual. |
| [`python/solvers.py`](python/solvers.py) | Dictionary mapping solver names to callable time integrators. |
| [`python/explicit_Euler.py`](python/explicit_Euler.py) | Explicit Euler integration and its CFL time-step calculation. |
| [`python/rk4.py`](python/rk4.py) | Four-stage RK4 integration and its CFL time-step calculation. |
| [`python/implicit_Euler.py`](python/implicit_Euler.py) | Complete backward Euler time loop; selects `newton_euler` or `fixed_point_euler`. |
| [`python/crank_nicolson.py`](python/crank_nicolson.py) | Complete Crank–Nicolson time loop; selects `newton_CN` or `fixed_point_CN`. |
| [`python/nonlinear_solvers.py`](python/nonlinear_solvers.py) | Four separate one-step functions: Newton and FPI for both implicit schemes; shared reference-scale and sparse Jacobian helpers. |
| [`python/data_io.py`](python/data_io.py) | Writes physical-cell primitive histories to text files and reads them back. |
| [`python/plotting.py`](python/plotting.py) | Mesh plotting and one- or two-axis plots of saved primitive variables. |
| [`python/miscellaneous.py`](python/miscellaneous.py) | Total mass and total energy comparison between initial and final states. |
| [`python/temp.py`](python/temp.py) | Standalone explanatory MUSCL diagram; not part of the simulation workflow. |
| [`python/data/`](python/data/) | Saved result folders included in the repository; new runs also write here when launched from `python`. |

## Configure a run in main.py

### Workflow and domain

| Setting | Meaning |
| --- | --- |
| `run_simulation` | `True` builds and solves a new case; `False` loads existing primitive files. |
| `show_domain` | Shows the mesh before a new computation. |
| `folder_name` | Base name for a new output folder under `data`. Existing folders cause a numeric suffix. |
| `load_folder` | Existing result folder name or path for load-only mode. |
| `solver_name` | One of the four names in the solver table below. |
| `x_bounds` | `[x_left, x_right]` in metres. |
| `n_cells` | Number of physical cells, excluding ghosts. |
| `n_ghost` | Ghost counts `(left, right)`, e.g. `(1, 1)`. |
| `node_density_rule` | Function of position controlling relative mesh density. Larger values place more cells locally. |
| `area_rule` | Function returning cross-sectional area in m². |
| `final_time` | End time in seconds; each integration call starts its time counter at zero. |
| `initial_integration_intervals` | Number of midpoint subintervals per cell when constructing initial averages. |

Existing domain rules are `uniform_node_density`, `gaussian_node_density`, `uniform_area`, `linear_area` and `nozzle_area`. Define additional rules in `domain_rules.py`, import them in `main.py`, and assign the desired function object. Pass `uniform_area`, not `uniform_area(x)`.

### Initial and boundary conditions

The five initial rule functions take a scalar position `x` and return pressure [Pa], velocity [m/s], gas temperature [K], liquid temperature [K], or gas volume fraction [dimensionless]. Select them using the `initial_*_rule` variables in `main.py`. Their definitions live in `conditions.py`.

The checked-in configuration currently selects backward Euler on `[0, 10]` with 500 physical cells, one ghost per side, reflective boundaries and `final_time=0.001`. The pressure rule is 500 bar for `4 < x < 6` and 10 bar elsewhere; velocity is zero, both temperatures are 293.15 K, and gas volume fraction is 0.1. Edit these profiles to select a different physical problem; output folder names do not select initial conditions.

Boundary settings form a dictionary:

```python
bc_settings = {"left": "reflective", "right": "reflective"}
```

| Boundary name | Implementation | Ghost requirement |
| --- | --- | --- |
| `reflective` | Mirrors physical states into ghosts and changes the sign of momentum only. | At least one; enough physical cells to mirror each ghost. |
| `zerograd` | Copies the nearest physical state into every ghost on that side. | At least one. |
| `periodic` | Wraps successive physical cells into ghost cells in spatial order. | At least one; select periodic conditions on both sides. |
| `fixed` | Fills ghosts with `fixed_state(side)`. Does not overwrite the edge physical cell. | At least one, allowing the edge physical cell to have a MUSCL slope. |

For example, a fixed left boundary and reflective right boundary can use `n_ghost=(1, 1)`. Edit `fixed_state` in `conditions.py` to change prescribed states.

### Solver options

Each solver block in `main.py` creates `solver_options`. The shared call is:

```python
solver = solvers[solver_name]
q_final, history = solver(q_initial, domain, final_time, bc_settings, **solver_options)
```

`**solver_options` expands the dictionary into named function arguments.

| `solver_name` | Time-step parameter | Additional options |
| --- | --- | --- |
| `explicit_euler` | `cfl` | `save_every` |
| `rk4` | `cfl` | `save_every` |
| `implicit_euler` | Required `tau` [s] | `save_every`, `nonlinear_method`, `eps_tol`, `limit`, `relax_factor`, `eps_j`, `q_reference`, `jacobian_every` |
| `crank_nicolson` | Required `tau` [s] | `save_every`, `nonlinear_method`, `eps_tol`, `limit`, `relax_factor`, `eps_j`, `q_reference`, `jacobian_every` |

Explicit methods recalculate `tau = cfl * min(dx / (abs(u) + c))` using physical cells. Implicit methods use the supplied `tau`. All methods shorten the final step to reach `final_time`.

`save_every` counts **completed time steps**, not RK stages or nonlinear iterations. It controls history snapshots and summary printing. The initial and final states are retained even if they fall outside the interval. Snapshots accumulate in memory; the main workflow writes files after the solver finishes successfully, not at every snapshot.

### Selecting Newton or fixed-point iteration

Newton is the default for both implicit time integrators. The full state is returned by the nonlinear solve; the outer time loop must not apply another `tau * rate` update.

```python
from implicit_Euler import implicit_euler
from crank_nicolson import crank_nicolson

# Default: Newton, with the Jacobian rebuilt every third inner iteration.
q_final, history = implicit_euler(q_initial, domain, final_time, bc_settings, tau=1e-5)

# Explicitly select relaxed fixed point.
q_final, history = crank_nicolson(q_initial, domain, final_time, bc_settings, tau=1e-5, nonlinear_method="fpi", relax_factor=0.5)

# Rebuild the Jacobian at every Newton iteration.
q_final, history = crank_nicolson(q_initial, domain, final_time, bc_settings, tau=1e-5, jacobian_every=1)
```

The examples assume that the domain, initial state and boundary settings have already been constructed. Each call starts a separate integration from `t=0`.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `tau` | Required | Prescribed physical time step [s]; only the final step is shortened. |
| `nonlinear_method` | `"newton"` | `"newton"` or `"fpi"`. |
| `eps_tol` | `1e-6` | Maximum component-scaled unrelaxed correction in every physical cell. |
| `limit` | `1000` | Maximum inner iterations for each physical time step. |
| `relax_factor` | `1.0` | Unrelaxed for both Newton and FPI. Supply a value in `(0, 1]` to change it. |
| `eps_j` | `1e-6` | Newton finite-difference perturbation factor; unrelated to the stopping tolerance. |
| `q_reference` | `None` | Automatic reference scales, or five finite positive component scales. Used only by Newton. |
| `jacobian_every` | `3` | Positive integer: rebuild and factorise on inner iterations `1, 1+k, 1+2*k, ...`. Used only by Newton. |
| `save_every` | `100` | Completed physical time steps between snapshots and summary lines. |
| `reconstruction` | `"muscl"` | Spatial reconstruction for every integrator: `"muscl"` or `"first_order"`. |

`eps_j`, `q_reference` and `jacobian_every` are ignored when `nonlinear_method="fpi"`. The simplified Newton implementation uses **fixed damping**, not automatic backtracking. It does not automatically reduce `tau` or recover from an invalid state.

Inside each implicit solver settings block in `main.py`, use:

```python
tau = 1e-5
save_every = 50
eps_tol = 1e-6
limit = 1000
nonlinear_method = "newton"  # Or "fpi".
relax_factor = 1.0         # Unrelaxed for both; set e.g. 0.25 explicitly.
eps_j = 1e-6
q_reference = None
jacobian_every = 3

solver_options = {"tau": tau, "save_every": save_every, "eps_tol": eps_tol, "limit": limit, "nonlinear_method": nonlinear_method, "relax_factor": relax_factor, "eps_j": eps_j, "q_reference": q_reference, "jacobian_every": jacobian_every}
```

Both implicit methods contain their own full time loop. There is no `integrate_implicit` dispatcher or additional boundary-condition wrapper. The existing `apply_boundary_conditions` function is called directly wherever required.

### Progress output

The implicit methods use short labels:

| Label | Meaning |
| --- | --- |
| `IE` | Implicit (backward) Euler |
| `CN` | Crank–Nicolson |
| `NWT` | Newton iteration |
| `FP` | Fixed-point iteration |
| `err` | Maximum component-scaled unrelaxed correction across physical cells |

Examples of the changing progress line and a completed-step summary:

```text
CN | t=1.0000e-05 | NWT 2/1000 | err=8.291e-04
CN | t=2.0000e-05 | NWT converged
Iteration: 50 | t=5.000e-04 | dt=1.00e-05 | min_p=1.000e+06 | max_p=4.000e+07 | max_u=2.000e+01
```

These are illustrative values. Summary `min_p` and `max_p` are minimum and maximum physical-cell pressure [Pa], and `max_u` is maximum absolute velocity [m/s]; `t` and `dt` are seconds. During the inner solve, `t` is the beginning of the current physical time step. After convergence it advances to the accepted time. Inner-iteration and converged messages overwrite the same line; the summary starts a new line after every `save_every` completed steps and after the final step. The old inner counter is not a physical-step counter.

The changing inner progress statements use `\r`, `end=""` and `.ljust(76)`. The leading carriage return occupies one character in the string, leaving at least 75 visible characters including padding. Spaces erase remnants of a longer previous message. The summary omits `end=""`, so it ends with a newline. It includes extra diagnostics and may wrap on a narrow terminal; the changing inner progress line remains compact. No `shutil` import is needed.

Use a terminal **at least 76 columns wide** for the usual message lengths. `.ljust` sets a minimum width; it does not truncate text. Unusually large counters or values can still make a message wider than the terminal. A width of 150 previously caused wrapping even when the visible text itself was short.

For example, the Newton progress print inside `newton_CN` is:

```python
print(f"\rCN | t={time:.4e} | NWT {counter}/{limit} | err={max_error:.3e}".ljust(76), end="", flush=True)
```

A stable implicit time discretisation does not guarantee convergence of its nonlinear solver for an arbitrary `tau`.

## Data structures and array shapes

Use these symbols throughout the examples:

- `N = domain["n_cells"]`: physical cells.
- `Nt = domain["n_total"]`: all cells, including left and right ghosts.
- `S`: number of saved time snapshots, including the initial state.
- `physical = domain["physical_ind"]`: integer indices of physical cells in a full state array.

### Shape reference

| Object | Python type / shape | Includes ghosts? |
| --- | --- | --- |
| One cell's conservative state | NumPy array `(5,)` | One selected cell |
| `q_initial`, `q_current`, `q_final` | NumPy array `(5, Nt)` | Yes |
| `q_to_primitive(q, eps_alpha=1e-8, eps_m=1e-12)` | Dictionary of scalar values | One selected cell |
| `get_primitives(q)[name]` | NumPy array `(Nt,)` | Yes |
| `compute_residual(q, domain, reconstruction="muscl", pressure_fallback="first_order")` | NumPy array `(5, N)` | No |
| Solver `history["times"]` | NumPy array `(S,)` | Not spatial |
| Solver `history["primitives"][name]` | NumPy array `(S, Nt)` | Yes |
| Loaded `history["primitives"][name]` | NumPy array `(S, N)` | No |
| Loaded `cell_centres` | NumPy array `(N,)` | No |

**A conservative state uses `[component, cell]`. A primitive history uses `[saved_time, cell]`.** There is no component axis in a primitive history array: the dictionary key selects the quantity.

### Conservative state: q

Each cell's five-component state is

$$
q_j = [\alpha_g\rho_g,\ \alpha_l\rho_l,\ \rho u,\ \rho E,\ \alpha_g]_j^T,
\qquad \rho=\alpha_g\rho_g+\alpha_l\rho_l,\quad E=e+u^2/2.
$$

| Row | Stored quantity | Meaning | SI unit |
| --- | --- | --- | --- |
| `q[0, :]` | `alpha_g * rho_g` | Gas mass per unit mixture volume, not gas-phase density alone | kg/m³ |
| `q[1, :]` | `alpha_l * rho_l` | Liquid mass per unit mixture volume | kg/m³ |
| `q[2, :]` | `rho * u` | Mixture momentum per unit volume | kg/(m²·s) |
| `q[3, :]` | `rho * E` | Mixture total energy per unit volume | J/m³ |
| `q[4, :]` | `alpha_g` | Gas volume fraction; `alpha_l = 1 - alpha_g` | Dimensionless |

The project calls this the conservative vector, although the fifth equation contains a nonconservative volume-fraction source. Cell values are volume averages; they are **not already multiplied by cell volume**. Pressure and temperature are recovered from the state, not stored as additional rows.

```python
physical = domain["physical_ind"]
j = physical[0]
q_cell = q_final[:, j]                # All five components in the first physical cell: (5,).
q_physical = q_final[:, physical]     # All physical states: (5, N).
rho = q_final[0, :] + q_final[1, :]    # Mixture density including ghosts: (Nt,).
u = q_final[2, :] / rho              # Mixture velocity including ghosts.
```

`primitive_to_q(p, u, t_g, t_l, alpha_g)` converts one scalar primitive state into a `(5,)` array. For an entire domain, use `initialize_state`, which integrates the rule functions rather than merely sampling cell centres.

### Primitive dictionaries: one cell versus one snapshot

```python
from solver_functions import q_to_primitive, get_primitives

cell = q_to_primitive(q_final[:, j])  # Dictionary of scalar values.
p_cell = cell["p"]

primitives = get_primitives(q_final) # Dictionary of 1D arrays for one time.
p_all = primitives["p"]             # Shape (Nt,).
p_physical = primitives["p"][physical]
```

The dictionary contains both primitive and derived thermodynamic quantities:

| Key | Meaning | SI unit |
| --- | --- | --- |
| `p` | Common pressure | Pa |
| `u` | Common velocity | m/s |
| `t_g`, `t_l` | Gas and liquid temperatures | K |
| `alpha_g`, `alpha_l` | Gas and liquid volume fractions | Dimensionless |
| `rho` | Mixture density | kg/m³ |
| `rho_g`, `rho_l` | Intrinsic gas and liquid densities | kg/m³ |
| `e` | Mixture specific internal energy | J/kg |
| `e_total` | Mixture specific total energy, `e + u**2/2` | J/kg |
| `e_g`, `e_l` | Phase specific internal energies | J/kg |
| `h_g`, `h_l` | Phase specific enthalpies | J/kg |
| `c` | Mixture sound speed | m/s |
| `c_g`, `c_l` | Phase sound speeds | m/s |
| `k` | Compressibility coefficient used in the volume-fraction source | Dimensionless |

`q[3, j]` is **energy density**; `primitives["e_total"][j]` is **specific energy**. For cell energy, use either `q[3, j] * volume[j]` or `rho[j] * e_total[j] * volume[j]`, not both density factors.

### History: snapshots over time

Every time integrator returns:

```python
q_final, history = solver(...)
```

`q_final` contains only the final conservative state. `history` contains selected **primitive snapshots**, not all conservative states and not every internal nonlinear iterate.

Its structure is:

```python
history = {
    "times": ...,        # Array of shape (S,), in seconds.
    "primitives": {
        "p": ...,        # Array of shape (S, Nt) before saving.
        "u": ...,        # Same shape.
        "rho": ...,      # Same shape.
        # All other keys from the primitive table above.
    }
}
```

These ellipses describe the layout; the solver fills them with actual NumPy arrays. During integration, snapshots are collected as lists and converted to arrays before return.

```python
times = history["times"]
p_initial = history["primitives"]["p"][0, physical]
p_final = history["primitives"]["p"][-1, physical]
p_at_first_cell = history["primitives"]["p"][:, physical[0]]
```

The first axis is the saved-time index. The second is the cell index. Saved times need not be evenly spaced because CFL-based time steps vary, and the last step may be shortened. Always read `history["times"]` instead of reconstructing times from row numbers.

After loading from disk the structure is the same, but its arrays have shape `(S, N)` because ghosts were removed. **Do not apply the original `physical_ind` again to loaded arrays.**

For 500 physical cells and one ghost per side:

| Quantity | Shape / indices |
| --- | --- |
| Full conservative state | `(5, 502)` |
| Full physical indices | `1, 2, ..., 500` |
| Solver pressure history with `S` snapshots | `(S, 502)` |
| Loaded pressure history | `(S, 500)` |
| Loaded physical-cell column indices | `0, 1, ..., 499` |

### Domain dictionary

`build_domain(...)` returns a dictionary used by initialisation, residuals, boundaries, plotting and saving.

| Key | Shape or type | Meaning |
| --- | --- | --- |
| `m` | `(2,)` | Physical left/right domain limits [m]. |
| `n_cells` | Integer | Physical-cell count. |
| `n_ghost_left`, `n_ghost_right` | Integers | Ghost counts. |
| `n_total` | Integer | Sum of physical and ghost counts. |
| `x_faces` | `(Nt + 1,)` | All face positions, including ghost geometry [m]. |
| `x_centres` | `(Nt,)` | Cell-centre positions [m]. |
| `dx` | `(Nt,)` | Cell widths [m]. |
| `area_faces` | `(Nt + 1,)` | Cross-sectional areas at faces [m²]. |
| `area_left`, `area_right` | Each `(Nt,)` | Areas at each cell's left and right face [m²]. |
| `area_centres` | `(Nt,)` | Area evaluated at cell centres [m²]. |
| `area_average` | `(Nt,)` | Cell volume divided by width [m²]. |
| `volumes` | `(Nt,)` | Integrated cell volumes [m³]. |
| `total_ind` | `(Nt,)`, integer | All cell indices. |
| `physical_ind` | `(N,)`, integer | Physical indices in full arrays. |
| `ghost_left_ind`, `ghost_right_ind` | Integer arrays | Ghost indices on each side. |

Cell `j` lies between `x_faces[j]` and `x_faces[j + 1]`. Its volume is the integral of area over that interval, not necessarily `area_centres[j] * dx[j]`.

## Saving and loading results

### Save a solver history

```python
from data_io import save_primitives

results_folder = save_primitives(history, domain, "my_run")
```

**Inputs:** the history returned by a solver, its matching domain, and a simple output folder name. The history is expected to contain all cells, including ghosts.

**Return:** a `pathlib.Path` pointing to the newly created result directory. Keep this returned path: it may differ from the requested name because an existing folder caused a suffix.

The function:

1. Creates `Path.cwd() / "data"` if needed.
2. Selects `my_run`, then `my_run_1`, `my_run_2`, etc. until it finds an unused directory.
3. Saves all history times to `times.txt`.
4. Saves physical-cell centres to `x_centres.txt`.
5. Saves each primitive array after selecting its physical columns.

It does not overwrite an earlier run or modify the input history. Use a simple base name such as `"my_run"`; the save interface is not a general nested-directory manager.

### File format

Every result folder contains these whitespace-separated numeric text files, without headers:

| File | Contents | Shape when loaded |
| --- | --- | --- |
| `times.txt` | Saved times in seconds, one per line | `(S,)` |
| `x_centres.txt` | Physical-cell positions in metres, one per line | `(N,)` |
| `p.txt` | Pressure for every saved time and physical cell | `(S, N)` |
| `u.txt` | Velocity, with the same row/column ordering | `(S, N)` |
| `<name>.txt` | One file for every other primitive key | `(S, N)` |

For each primitive file, **rows are saved times and columns are physical cells**. Entry `[s, j]` corresponds to `times[s]` and `x_centres[j]`. Files use scientific notation with `fmt="%.18e"`; values remain in SI units. There is no extra time or coordinate column inside `p.txt`, `u.txt`, etc.

### Load a saved run

```python
from data_io import load_primitives

cell_centres, loaded_history = load_primitives("my_run")
```

The loader accepts:

| Argument | Meaning |
| --- | --- |
| `"my_run"` | `data/my_run` under the current working directory. |
| `"data/my_run"` | An explicit relative directory path. |
| `"../other_results/my_run"` | A relative directory path outside the current folder. |
| An absolute directory path | That exact directory. |
| `results_folder` returned by saving | The directory just created, including its suffix. |

On Windows, an absolute path can be written as `r"C:\Users\YourName\results\my_run"`.

The return values are:

```python
cell_centres                         # NumPy array (N,).
loaded_history["times"]              # NumPy array (S,).
loaded_history["primitives"]["p"]    # NumPy array (S, N).
```

The loader preserves two-dimensional primitive arrays even for a single saved time or a single physical cell. Each primitive file must have shape `(len(times), len(cell_centres))`; otherwise it raises `ValueError`.

It treats every other `*.txt` file in the folder as a primitive file. Keep unrelated text files outside result folders. For older results it translates `Tg→t_g`, `Tl→t_l`, `ag→alpha_g`, `al→alpha_l`, `E→e_total`, and `K→k`.

### What is not saved

The `data_io.py` interface described here exposes **only `save_primitives` and `load_primitives`**. It does not save the full `domain` dictionary, conservative-state arrays, ghost cells, rule functions, boundary settings, solver options or restart metadata. In particular, there are no `save_domain` / `load_domain` functions in this revision.

Loaded coordinates are sufficient for solution plots, but do not determine the original cell faces, areas, volumes or ghost geometry. `plot_domain` still requires the original/rebuilt domain and its area rule. Keep the case settings if you need that geometry later. Saved primitive output is a post-processing format, not a complete restart checkpoint.

Do not pass a loaded history directly back to `save_primitives` with the original domain: the save function would try to select physical indices from arrays whose ghost columns have already been removed.

### Read files without the project loader

For independent analysis or interoperability with other programs:

```python
from pathlib import Path
import numpy as np

folder = Path("data/my_run")
times = np.loadtxt(folder / "times.txt", ndmin=1)
x = np.loadtxt(folder / "x_centres.txt", ndmin=1)
p = np.loadtxt(folder / "p.txt", ndmin=2)

assert p.shape == (len(times), len(x))
```

## Plotting and extracting data

### Plot a saved quantity

Run this from the `python` directory, or save it as a separate script there:

```python
from data_io import load_primitives
from plotting import simple_plot

x, history = load_primitives("task14cn")
simple_plot(history, x, "p", time=0.001)
simple_plot(history, x, ["t_g", "t_l"], time=0.001, xlim=[3, 7])
```

Despite the first parameter being named `primitives` in its definition, **`simple_plot` expects the full history dictionary**, containing both `times` and `primitives`. Pass `history`, not `history["primitives"]` or `get_primitives(q_final)`.

`time` is a requested physical time in seconds, **not a row index**. The function selects `argmin(abs(saved_times - time))`; it does not interpolate. The title displays the selected saved time. A request outside the saved interval selects the nearest endpoint.

One identifier produces a black curve on one y axis. Two identifiers produce a black first curve and a red second curve on a separate y axis. `xlim=None` uses the automatic range; `[x_left, x_right]` restricts the view. Values are not converted automatically: pressure is plotted in Pa, temperatures in K, etc. The function displays the figure and returns `None`.

### Extract a spatial profile or a time series

```python
import numpy as np
from data_io import load_primitives

x, history = load_primitives("task14cn")
times = history["times"]
p = history["primitives"]["p"]

p_final = p[-1, :]                       # Final spatial pressure profile: (N,).
s = np.argmin(np.abs(times - 0.0005))
p_at_time = p[s, :]                      # Profile at the nearest saved time.
j = np.argmin(np.abs(x - 5.0))
p_at_location = p[:, j]                  # Time series at the nearest cell centre: (S,).
p_single_value = p[s, j]                 # One scalar.
p_final_bar = p_final / 1e5              # Convert explicitly if bar are desired.
```

For plotting a solver history before saving, select matching physical columns and coordinates:

```python
physical = domain["physical_ind"]
x = domain["x_centres"][physical]
physical_history = {
    "times": history["times"],
    "primitives": {name: values[:, physical] for name, values in history["primitives"].items()}
}
simple_plot(physical_history, x, "p", time=final_time)
```

## Main function reference

### Geometry and initialisation

| Function | Inputs and behavior | Returns |
| --- | --- | --- |
| `volume_calculator(area_rule, x_left, x_right, n=10000)` | Integrates area with a composite trapezoidal rule over one cell. | Scalar volume [m³]. |
| `build_domain(m, node_density_rule, area_rule, n_cells, n_ghost)` | Inverts cumulative node density to place faces; extends ghost geometry and evaluates areas/volumes. `n_ghost` may be a scalar or pair. | Domain dictionary described above. |
| `initialize_state(domain, area_rule, p_rule, t_g_rule, t_l_rule, u_rule, alpha_g_rule, n=100)` | Uses midpoint quadrature for `area(x) * primitive_to_q(...)` over each physical cell and divides by volume. Initially copies edge states into ghosts. | Conservative array `(5, Nt)`. |
| `fixed_state(side)` | Converts the prescribed left/right primitive settings to a conservative external state. | `(5,)` array. |
| `apply_boundary_conditions(domain, q, case, side, check=False, rtol=1e-3)` | Updates ghosts **in place** and returns the same array. For `fixed`, fills ghosts with the reservoir state and optionally checks the initial edge state without overwriting it. | Full conservative array `(5, Nt)`. |

After `initialize_state`, time integrators apply the chosen boundary conditions before calculating residuals. If calling `compute_residual` yourself, populate ghosts first.

### Thermodynamics and state conversion

| Function | Input | Return |
| --- | --- | --- |
| `primitive_to_q(p, u, t_g, t_l, alpha_g)` | Five scalar primitive inputs in SI units. | One conservative vector `(5,)`. |
| `q_to_primitive(q, eps_alpha=1e-8, eps_m=1e-12)` | One conservative vector `(5,)`. Bounds phase fractions/masses internally for conversion; does not repair the stored input array. | Dictionary of scalar primitive/derived values. |
| `get_primitives(q)` | Full conservative state `(5, Nt)`. Converts each column. | Dictionary of arrays `(Nt,)`. |
| `sg_eos(request, k=None, e=None, rho=None, p=None, temperature=None, q=None)` | A request string and only the inputs relevant to that request. `k=1` is gas; `k=2` is liquid. | Requested scalar thermodynamic property. |

`sg_eos` supports:

| Request | Required inputs | Output |
| --- | --- | --- |
| `rho_pT` | `k`, `p`, `temperature` | Phase density. |
| `p_erho` | `k`, `e`, `rho` | Phase pressure. |
| `e_prho` | `k`, `p`, `rho` | Phase specific internal energy. |
| `T_prho` | `k`, `p`, `rho` | Phase temperature. |
| `h_prho` | `k`, `p`, `rho` | Phase specific enthalpy. |
| `c_prho` | `k`, `p`, `rho` | Phase sound speed. |
| `p_common` | `q` with shape `(5,)` | Common pressure recovered from the mixture state. |

Material constants are defined inside `sg_eos`. Its checks reject invalid density, shifted pressure or temperature. The optional argument named `q` in this function is the state vector for `p_common`; the local EOS energy-offset constants are a different use of the same symbol.

### Reconstruction, flux and residual

| Function | Role | Return |
| --- | --- | --- |
| `minmod(a, b)` | Elementwise slope limiting: smaller magnitude for matching signs, zero otherwise. | Limited slope(s). |
| `muscl_reconstruction(q, domain, pressure_fallback="first_order")` | Reconstructs area-weighted momentum (other components directly) at left/right faces of every cell and reduces slopes when reconstructed pressure fails the 5,000 Pa safeguard. | `q_left, q_right`, each `(5, Nt)`. |
| `rusanov_flux(q_interface_left, q_interface_right)` | Uses two `(5,)` interface states to calculate the numerical flux and arithmetic-average interface velocity. | `flux` of shape `(5,)`, and scalar `u_star`. |
| `compute_residual(q, domain, reconstruction="muscl", pressure_fallback="first_order")` | Combines area-weighted face flux differences with momentum and volume-fraction sources. Requires at least one populated ghost per side; supports MUSCL or first-order states. | Rate array `(5, N)` for physical cells only. |

`q_left[:, j]` and `q_right[:, j]` are the two reconstructed faces **of cell j**. At face index `f`, the Riemann states are `q_right[:, f - 1]` from the left cell and `q_left[:, f]` from the right cell.

A residual column `i` corresponds to full-array cell `domain["physical_ind"][i]`. This is why updates use:

```python
rate = compute_residual(q_current, domain)
q_next = q_current.copy()
q_next[:, domain["physical_ind"]] += tau * rate
```

That example is an explicit update; the implicit methods solve their own equations below. `compute_residual` returns a time derivative, not a time-integrated increment and not the nonlinear solver's convergence error.

### Time integrators and nonlinear solvers

Exact public signatures:

```python
explicit_euler(q_initial, domain, final_time, bc_settings, cfl=0.5, save_every=100, reconstruction="muscl", pressure_fallback="first_order")
rk4(q_initial, domain, final_time, bc_settings, cfl=0.5, save_every=100, reconstruction="muscl", pressure_fallback="first_order")
implicit_euler(q_initial, domain, final_time, bc_settings, tau, save_every=100, eps_tol=1e-6, limit=1000, relax_factor=1.0, nonlinear_method="newton", eps_j=1e-6, q_reference=None, jacobian_every=3, reconstruction="muscl", pressure_fallback="first_order", convergence_check="scaled")
crank_nicolson(q_initial, domain, final_time, bc_settings, tau, save_every=100, eps_tol=1e-6, limit=1000, relax_factor=1.0, nonlinear_method="newton", eps_j=1e-6, q_reference=None, jacobian_every=3, reconstruction="muscl", pressure_fallback="first_order", convergence_check="scaled")
```

All return `(q_final, history)`, apply boundary conditions during evolution, collect primitive snapshots, and print a final mass/energy comparison. Calling a solver directly does not create output files; `save_primitives` performs that separate step.

`get_time_step(u, c, dx, cfl=0.5)` is defined in both explicit-method modules. Supply matching physical-cell arrays; it returns one scalar time step [s].

The four nonlinear routines are separate complete functions, each solving **one physical time step**:

```python
fixed_point_euler(q_initial, tau, domain, bc_settings, eps_tol=1e-6, limit=1000, relax_factor=1.0, time=0.0, reconstruction="muscl", pressure_fallback="first_order", convergence_check="scaled")
fixed_point_CN(q_initial, tau, domain, bc_settings, eps_tol=1e-6, limit=1000, relax_factor=1.0, time=0.0, reconstruction="muscl", pressure_fallback="first_order", convergence_check="scaled")
newton_euler(q_initial, tau, domain, bc_settings, eps_tol=1e-6, limit=1000, relax_factor=1.0, time=0.0, eps_j=1e-6, q_reference=None, jacobian_every=3, reconstruction="muscl", pressure_fallback="first_order", convergence_check="scaled")
newton_CN(q_initial, tau, domain, bc_settings, eps_tol=1e-6, limit=1000, relax_factor=1.0, time=0.0, eps_j=1e-6, q_reference=None, jacobian_every=3, reconstruction="muscl", pressure_fallback="first_order", convergence_check="scaled")
```

Every routine returns the new full conservative state `(5, Nt)`, including populated ghost cells. It does not return a residual or history. The outer integrator must not multiply this state by `tau` again. `time` is only a progress-display label; one-step routines do not advance an outer clock themselves. Direct one-step calls and full time integrators all default to `relax_factor=1.0`. Supply a numeric factor explicitly; `None` is no longer a supported automatic choice.

| Shared helper | Inputs | Return |
| --- | --- | --- |
| `reference_state(q, domain, q_reference=None)` | Full conservative state and domain; optional five component scales | Reference array `(5,)` |
| `residual_jacobian(q, r, domain, bc_settings, reference, eps_j=1e-6, reconstruction="muscl", pressure_fallback="first_order")` | Full state `(5, Nt)` with consistent ghosts, its current physical residual `(5, N)`, boundary settings and scales `(5,)` | SciPy CSC sparse matrix `(5*N, 5*N)` |

Only the reference scales and residual Jacobian are shared numerical helpers; each nonlinear method contains its own equation, update and convergence loop.

### Conservation diagnostic

`conservation_check(q_initial, q_final, domain)` integrates over physical cells:

$$
M=\sum_j \rho_j V_j,\qquad \mathcal E=\sum_j\rho_j E_j V_j.
$$

It reads mixture density directly as `q[0] + q[1]` and energy density as `q[3]`, without EOS conversion or clipping. It also reports the gas and liquid masses separately. For all four totals, it prints signed absolute and percentage changes; a zero initial total gives `nan` percent and still reports the absolute change. It returns only `(mass_error, energy_error)`, the signed mixture-mass and total-energy percentage changes `100 * (final - initial) / initial`. These are global changes, not norms or error bounds. For open/fixed boundaries, mass or energy may legitimately enter or leave; the routine does not subtract boundary flux contributions.

## Numerical method and convergence

The spatial operator is `dq/dt = R(q)`. Rusanov dissipation uses the maximum of `abs(u) + c` at the two interface states. The residual includes the area-change momentum source and the volume-fraction source based on interface velocities.

### Newton and Jacobian reuse

Newton solves the nonlinear equation defect `G(q)=0`. It does not seek `R(q)=0`, which would mean a stationary semidiscrete state.

For backward Euler:

$$
G(q)=q-q^n-\tau R(q),\qquad J_G=I-\tau J_R.
$$

For Crank–Nicolson:

$$
G(q)=q-q^n-\frac{\tau}{2}[R(q^n)+R(q)],\qquad J_G=I-\frac{\tau}{2}J_R.
$$

Here `J_R = ∂R/∂q`. Each iteration solves and updates:

$$
J_G\,\delta q=-G(q^{(m)}),\qquad q^{(m+1)}=q^{(m)}+\omega\,\delta q.
$$

`omega = relax_factor` is fixed during the solve. A value of `1.0` applies the full correction. There is **no backtracking, line search, automatic step rejection/retry, or adaptive relaxation** in this simplified version. EOS or numerical errors can terminate a run.

When `(counter - 1) % jacobian_every == 0`, calculate `J_R`, form `J_G`, and factorise it using `scipy.sparse.linalg.splu`. On the remaining iterations reuse that factorisation with `factorisation.solve(...)`. **Recalculate `R(q)` and `G(q)` every iteration**, even when the matrix is reused.

| `jacobian_every` | Rebuild iterations within one time step |
| --- | --- |
| `1` | 1, 2, 3, 4, ... |
| `2` | 1, 3, 5, 7, ... |
| `3` (default) | 1, 4, 7, 10, ... |

The counter restarts for every physical time step, so the first Jacobian and factorisation are always fresh. They are not carried across time steps. Matrix reuse is a modified Newton method: it saves work per iteration but may increase the number of iterations or prevent convergence. Compare total runtime, not just iteration count; use `jacobian_every=1` if reuse causes problems.

### Numerical Jacobian and state ordering

Flatten physical arrays by cell:

```python
q_vector = q[:, physical].flatten(order="F")
g_vector = g_m.flatten(order="F")
correction = factorisation.solve(-g_vector)
correction = correction.reshape((5, len(physical)), order="F")
```

In zero-based physical-cell indexing, `row = 5*i + equation` and `column = 5*j + component`. A `(5, 5)` block describes how the five residual equations in cell `i` respond to the five state entries in cell `j`. Ghost cells are not independent unknowns.

For the current nonperiodic MUSCL stencil, perturbing cell `j` affects physical residual cells `j-2` through `j+2`, clipped at the boundaries. The helper perturbs one component at a time in five spatial groups (`0,5,10,...`; `1,6,11,...`; etc.). Each perturbed state receives direct left and right boundary updates before `compute_residual` is evaluated.

A full rebuild uses at most 25 perturbed residual evaluations (five groups × five components), in addition to the current baseline residual already evaluated by the Newton iteration. Some groups are empty on very small meshes. For periodic boundaries the helper groups cells whose wrapped five-cell residual supports are disjoint, then extracts only the wrapped affected rows. A greedy grouping handles any positive cell count; 500 cells use five groups and 25 perturbed residual evaluations per rebuild. Other cell counts may require more groups. On tiny meshes every cell may need its own group.

The matrix is assembled as a SciPy `csc_matrix`. Perturbations use a one-sided difference, dividing by the actual floating-point change applied to each entry. The simplified implementation does not retry invalid perturbations or automatically repair a failed factorisation.

### Reference scales for perturbations

For component `b` and physical cell `j`, the nominal perturbation magnitude is

$$
h_{b,j}=\epsilon_J\max(|q_{b,j}^{(m)}|,q_b^{\mathrm{ref}}).
$$

`eps_j` is `epsilon_J`. The automatic five-entry reference is derived from the old state `q_n` once per physical time step and stays fixed during its nonlinear iterations:

| Component | Automatic reference | Units |
| --- | --- | --- |
| Gas partial density | Maximum physical-cell `abs(q[0])`, floored at `1e-3` | kg/m³ |
| Liquid partial density | Maximum physical-cell `abs(q[1])`, floored at `1e-3` | kg/m³ |
| Momentum density | Maximum physical-cell `rho * (abs(u) + c)` | kg/(m² s) |
| Total energy density | Maximum physical-cell `abs(q[3])`, floored at `1.0` | J/m³ |
| Gas volume fraction | `1.0` | Dimensionless |

The acoustic momentum scale is nonzero even when the initial velocity is zero. It is a differentiation scale, not an assumed fluid velocity. For volume fraction, the perturbation points towards the interior of `(0, 1)` and its magnitude is capped at `0.25`; at the default `eps_j` it is normally `1e-6`.

A custom `q_reference` must contain five finite positive values in the component order above. It is a set of scales, not a complete thermodynamic state, and is not a convergence tolerance. Custom values are reused at each time step. The current local magnitude can exceed a reference, in which case it determines the perturbation instead. Domain-wide maxima are practical defaults, but can be too large for cells whose values are much smaller; choose custom scales or a smaller `eps_j` when justified by derivative checks.

### Relaxed fixed-point iteration

For backward Euler, the relaxed fixed-point update is

$$
q^{(m+1)}=(1-\omega)q^{(m)}+\omega\left[q^n+\tau R(q^{(m)})\right].
$$

For Crank–Nicolson, it is

$$
q^{(m+1)}=(1-\omega)q^{(m)}+\omega\left[q^n+\frac{\tau}{2}\left(R(q^n)+R(q^{(m)})\right)\right].
$$

Here `omega = relax_factor`, `n` indexes physical time steps, and `m` indexes inner iterations. The old state `q_n` remains fixed; Crank–Nicolson also calculates `R(q_n)` once per step.

### Component-scaled convergence check

With the default `convergence_check="scaled"`, all four nonlinear routines test the **unrelaxed correction** separately in every physical cell. Let `d` denote that correction, before multiplication by `relax_factor`:

- Backward Euler FPI: `d = q_n + tau*R(q_m) - q_m`.
- Crank–Nicolson FPI: `d = q_n + tau/2*(R(q_n) + R(q_m)) - q_m`.
- Newton: solve `J_G*d = -G(q_m)` using the current stored factorisation.

The update is `q_next = q_m + relax_factor*d`. The stopping check is

$$
\epsilon_j=\max_{b=0,\ldots,4}\frac{|d_{b,j}|}{s_b},\qquad
\epsilon_j < \texttt{eps_tol}\quad\text{for every physical cell }j.
$$

The five scales `s` use the automatic values in the reference table above, evaluated from `q_n` once per physical time step. A custom `q_reference` affects Jacobian perturbations only; it does not alter the stopping scales.

```python
cell_errors = np.zeros(n_cells)
for i in range(n_cells):
    cell_errors[i] = np.max(np.abs(correction[:, i])/scale)
```

This prevents the large energy component from masking changes in volume fraction or mass. With `eps_tol=1e-6`, the volume-fraction correction must be below `1e-6` because its scale is one. The test uses the correction directly rather than subtracting two almost equal damped iterates and dividing by the relaxation factor. Small damping therefore cannot artificially shrink this reported error at the same iterate. Relaxation still changes the sequence of iterates and convergence behaviour.

These are **relative-to-characteristic-scale** tolerances, not pointwise relative tolerances: the same five scales apply to every physical cell. For FPI, the correction equals the negative time-discrete equation defect. For Newton, this is a correction test, not an independent equation-defect or true-solution-error bound, especially with a reused Jacobian. Failure to meet tolerance within `limit` raises `RuntimeError`.

### Spatial reconstruction and pressure safeguard

For MUSCL, only momentum is temporarily area-weighted: `q_rec[2] = area_average*q[2]`. Minmod slopes reconstruct this quantity at cell faces, then division by `area_left` and `area_right` restores momentum density before flux and pressure calculations. Other components are reconstructed directly. Constant-area reconstruction is unchanged up to floating-point roundoff.

Set `reconstruction="first_order"` to use the cell-average state on both faces, without MUSCL or its face-pressure slope limiter. The selection passes through all four integrators, nonlinear solvers, and perturbed residual evaluations in the Jacobian.

MUSCL uses minmod slopes. The default `pressure_fallback="first_order"` replaces both face states of a cell by its original conservative cell average if either face fails the pressure check. With `pressure_fallback="gradual"`, a 40-step bisection search reduces both face deviations by the same factor to satisfy the safeguard. The gradual option avoids abruptly discarding the full slope when only a small reduction is needed. If the cell-average state itself fails the safeguard, the code falls back to its cell-average face states; it does not repair that average. The literal `5000` threshold and `>=` comparison are retained for both modes; the gradual mode retains 40 bisection iterations; no `p_min` parameter is introduced. This gradual reduction intentionally differs from immediately setting the entire slope to zero whenever a reconstructed face fails. The 5,000 Pa value is a numerical safeguard in this implementation, not a universal physical threshold.

## Common mistakes and limitations

| Symptom or question | What to check |
| --- | --- |
| `ModuleNotFoundError: scipy` | Install with `python -m pip install scipy` using the interpreter that runs the project. |
| Newton becomes slower or fails after enabling reuse | Try `jacobian_every=1`; an older Jacobian can need more iterations. |
| Newton produces an invalid state | The simplified version has no backtracking. Reduce `tau` or the fixed `relax_factor` and inspect the state. |
| Progress wraps onto successive lines | Use the short labels and a terminal at least 76 columns wide; remove old `.ljust(150)` formatting. |
| Characters remain after `converged` | Keep `.ljust(76)` on the abbreviated progress and converged messages. `\r` alone does not clear the line. |
| Files appear in an unexpected place | `Path.cwd()` determines the data directory. Run from `python`, or use an explicit path when loading. |
| Loading opens an older run | A new save may have created `_1`, `_2`, etc. Use the returned `results_folder` or the exact suffixed name. |
| Array-size mismatch after loading | Loaded arrays already exclude ghosts. Do not apply `domain["physical_ind"]` again. |
| `simple_plot` reports a missing key | Pass the full history, and use a key from the primitive table. A single snapshot dictionary is not a history. |
| The requested plot time differs from its title | The nearest saved time is selected. Reduce `save_every` for more temporal samples. |
| Pressure appears too large for a bar-scale plot | Stored pressure is in Pa. Divide by `1e5` in a custom plot if required. |
| Fixed boundary rejects the mesh | Use at least one ghost on that side and edit `fixed_state(side)`. |
| Importing `main` unexpectedly starts a simulation | This file executes at module level. Import individual functions from their modules instead. |
| Can a saved folder reproduce the mesh or restart the solver? | Not by itself in this revision: geometry/settings and conservative/ghost states are not saved. |

The configured RK4 time-step estimate uses the same CFL expression as explicit Euler; this is an implementation choice, not an automatic stability guarantee for every case. Likewise, reducing a nonlinear tolerance does not replace spatial/time-step refinement studies. The focused regression checks can be run with `python tests/check_updates.py` from the project root. They do not replace full Task 15 reference comparisons or mesh/time-step refinement studies.

Midpoint initialisation avoids sampling a discontinuity exactly on a cell face. It does not integrate arbitrary discontinuities inside a cell exactly. Geometry volumes retain their existing quadrature; for variable area the two quadratures can introduce small initial averaging errors.

## Comparing MUSCL pressure fallbacks

In main, keep `reconstruction="muscl"` and select `pressure_fallback="first_order"` (default, assignment fallback) or `pressure_fallback="gradual"` (previous bisection). The option passes through every solver and Newton Jacobian evaluation. `reconstruction="first_order"` instead disables MUSCL everywhere and makes the pressure-fallback selection inactive. The threshold remains literal 5000 Pa.

All four nonlinear one-step functions and both implicit time integrators now default to `relax_factor=1.0`. Set e.g. `relax_factor=0.25` explicitly to damp updates. Main exposes this option in each implicit block. The component-scaled unrelaxed convergence check remains the default; `convergence_check="teacher"` selects the full-matrix norm ratio described below.

All four integrators print minimum pressure as `min_p` at the existing summary interval and final time, alongside maximum pressure and speed, using physical cells only. These are accepted cell-average pressures, not reconstructed face pressures or intermediate nonlinear iterates. A positive cell minimum does not prove no reconstructed face crossed the safeguard. Values between summary intervals are not reported by this diagnostic.

For a controlled comparison, keep the mesh, tau, initial conditions, nonlinear method, tolerance and Jacobian interval fixed; change only the fallback. Use `save_every=1` to see pressure at every accepted time step. Default relaxation one does not guarantee FPI convergence for an arbitrary time step.

The ZIP includes the original repository data folders for completeness. They are existing results, not outputs produced with these new defaults. Python bytecode caches and Git metadata are omitted. The private email attachment is not included.

Run `python tests/check_updates.py` and `python tests/check_fallback_options.py` from the extracted repository root for the regression checks. These short integration tests verify implementation behaviour; they do not establish the teacher's claims over the full Task 15 transient.

## Thermodynamic and periodic-Jacobian optimizations

`q_to_pressure(q, eps_alpha=1e-8, eps_m=1e-12)` recovers pressure with the same partial-mass/volume-fraction bounds and `sg_eos("p_common")` validity checks as full primitive recovery. MUSCL uses it for the literal 5000 Pa safeguard. The fallback selection and bisection behaviour are unchanged.

`q_to_flow(q, eps_alpha=1e-8, eps_m=1e-12)` returns only `p`, `u`, `c`, `alpha_g` and `k`, which are the quantities used by Rusanov fluxes and source terms. It avoids computing unused phase energies, enthalpies and full primitive dictionaries. Sound speeds retain the original EOS checks and arithmetic. `q_to_primitive` and `get_primitives` remain unchanged for complete output, saving and plotting.

Periodic Jacobian grouping uses the modulo-wrapped set of affected residual cells for each perturbed cell. Perturbations share a group only if these sets are disjoint. This changes Jacobian construction cost, not its finite-difference formula or `jacobian_every`. Ghosts are updated for every perturbed state. The existing five-group nonperiodic path is unchanged.

Run `python tests/check_optimizations.py` to compare minimal recovery with full recovery, verify clipping/rejections, check periodic grouped Jacobians against individual perturbations for 1–20 cells with one/two ghosts, both reconstructions and both fallback modes, and count 25 perturbed evaluations for a 500-cell periodic rebuild. The other two test suites remain included.

## Choosing the nonlinear convergence check

Set the shared option in main:

```python
convergence_check = "scaled"  # Current/default check.
# convergence_check = "teacher"  # Literal full-matrix 2-norm ratio.
```

Main forwards it only to the implicit Euler/CN blocks. Both integrators pass it to their selected Newton/FPI routine. Explicit Euler and RK4 do not use a nonlinear stopping check. The parameter is also available in direct calls to all four nonlinear one-step functions, with default `"scaled"`.

| Mode | Error calculation | Cells | Relaxation |
| --- | --- | --- | --- |
| `"scaled"` | Maximum over components/cells of abs(unrelaxed correction)/component scale | Physical cells | Applied damping does not directly reduce the measured correction |
| `"teacher"` | `np.linalg.norm(q_next - q_m, 2) / np.linalg.norm(q_next, 2)` | Full state, including updated ghosts | Measures the actual applied/damped update |

Both modes accept when `error < eps_tol` and print that selected error in the same progress line. The default scaled criterion is mathematically unchanged. Teacher mode uses the matrix spectral norm (largest singular value), matching MATLAB `norm(Q,2)` when Q is a matrix; it is neither a flattened Euclidean norm, a Frobenius norm nor a separate test per cell. The literal full-state expression is evaluated after boundary updates. If the teacher uses a vector or excludes ghosts before applying the expression, that would be a different comparison.

A zero numerator and denominator is assigned error zero; a zero denominator with a nonzero numerator, or a nonfinite state/error, fails. For physical valid states the ordinary ratio is used without a denominator floor. Teacher mode has no component scaling and can be dominated by energy; a small relaxation factor can reduce its reported error. This option is provided for direct diagnostic comparisons.

The same eps_tol does not have the same meaning in the two modes. Keep all other settings fixed when comparing their behaviour. Jacobian reuse, relaxation defaults, pressure fallback, the literal 5000 Pa threshold, min_p printing and data formats are unchanged.

Run `python tests/check_convergence_options.py` from the project root to verify the literal ratio and ghost inclusion, damping, strict tolerance, default scaled behaviour, invalid choices and integration/history for both methods with both nonlinear solvers.
