# One-dimensional two-phase finite-volume solver

A Python framework for one-dimensional gas–liquid flow with a stiffened-gas equation of state, MUSCL reconstruction and Rusanov numerical fluxes. The state uses a common pressure and velocity, separate phase temperatures, and a transported gas volume fraction. Cross-sectional area and mesh density can vary along the domain.

The normal user workflow is to **configure `python/main.py`, run a solver, save its results, and plot the saved data**. You can also load existing results in a separate script without running a simulation.

Four time integrators are registered: explicit Euler, classical RK4, backward Euler and Crank–Nicolson. The two implicit methods use relaxed fixed-point iteration. This README describes the implementation currently in this repository; method names do not by themselves establish accuracy or convergence for a particular case.

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

Install Python, NumPy and Matplotlib. The Python source files are inside the **`python` subdirectory**, not the repository root.

```bash
git clone https://github.com/Vytuk189/nmine_g3_project.git
cd nmine_g3_project/python
python -m pip install numpy matplotlib
python main.py
```

Close each Matplotlib window to continue. With `show_domain=True`, the mesh plot opens before the computation starts.

Run from the `python` directory consistently. Saving and loading use the **current working directory**, so running `python python/main.py` from the repository root puts new results in a different `data` directory.

To inspect an existing run without computing again, edit these settings in `main.py`:

```python
run_simulation = False
load_folder = "task14cn"  # The actual saved folder name, including any suffix.
```

Then run `python main.py`. The solution plots are produced from disk. In this checked-in version, `show_domain` applies only to new simulations because the domain itself is not loaded from disk.

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
| [`python/implicit_Euler.py`](python/implicit_Euler.py) | Backward Euler time loop with a prescribed time step. |
| [`python/crank_nicolson.py`](python/crank_nicolson.py) | Crank–Nicolson time loop with a prescribed time step. |
| [`python/nonlinear_solvers.py`](python/nonlinear_solvers.py) | Relaxed fixed-point solvers for one backward Euler or Crank–Nicolson step. |
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
| `initial_integration_intervals` | Number of trapezoidal subintervals per cell when constructing initial averages. |

Existing domain rules are `uniform_node_density`, `gaussian_node_density`, `uniform_area`, `linear_area` and `nozzle_area`. Define additional rules in `domain_rules.py`, import them in `main.py`, and assign the desired function object. Pass `uniform_area`, not `uniform_area(x)`.

### Initial and boundary conditions

The five initial rule functions take a scalar position `x` and return pressure [Pa], velocity [m/s], gas temperature [K], liquid temperature [K], or gas volume fraction [dimensionless]. Select them using the `initial_*_rule` variables in `main.py`. Their definitions live in `conditions.py`.

The checked-in configuration currently selects Crank–Nicolson on `[0, 10]` with 500 physical cells, one ghost per side, reflective boundaries and `final_time=0.001`. The pressure rule is 500 bar for `4 < x < 6` and 10 bar elsewhere; velocity is zero, both temperatures are 293.15 K, and gas volume fraction is 0.1. Edit these profiles to select a different physical problem; output folder names do not select initial conditions.

Boundary settings form a dictionary:

```python
bc_settings = {"left": "reflective", "right": "reflective"}
```

| Boundary name | Implementation | Ghost requirement |
| --- | --- | --- |
| `reflective` | Mirrors physical states into ghosts and changes the sign of momentum only. | At least one; enough physical cells to mirror each ghost. |
| `zerograd` | Copies the nearest physical state into every ghost on that side. | At least one. |
| `periodic` | Copies the opposite edge physical state into ghosts. | Use one ghost per side and periodic conditions on both sides; multiple ghosts are not filled as a full periodic sequence. |
| `fixed` | Supplies `fixed_state(side)` as the external face state in the residual. Does not overwrite the edge physical cell. | Zero ghosts on the fixed side. |

For example, a fixed left boundary and reflective right boundary require `n_ghost=(0, 1)`. Edit `fixed_state` in `conditions.py` to change prescribed states.

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
| `implicit_euler` | Required `tau` [s] | `save_every`, `eps_tol`, `limit`, `relax_factor` |
| `crank_nicolson` | Required `tau` [s] | `save_every`, `eps_tol`, `limit`, `relax_factor` |

Explicit methods recalculate `tau = cfl * min(dx / (abs(u) + c))` using physical cells. Implicit methods use the supplied `tau`. All methods shorten the final step to reach `final_time`.

`save_every` counts **completed time steps**, not RK stages or fixed-point iterations. It controls history snapshots and summary printing. The initial and final states are retained even if they fall outside the interval. Snapshots accumulate in memory; the main workflow writes files after the solver finishes successfully, not at every snapshot.

For implicit methods, `limit` caps inner iterations per step; `eps_tol` controls their stopping criterion; `relax_factor` lies in `(0, 1]`. A stable implicit time discretisation does not guarantee that its fixed-point solver converges for any `tau`.

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
| `q_to_primitive(q[:, j])` | Dictionary of scalar values | One selected cell |
| `get_primitives(q)[name]` | NumPy array `(Nt,)` | Yes |
| `compute_residual(q, domain)` | NumPy array `(5, N)` | No |
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

The checked-in `data_io.py` currently exposes **only `save_primitives` and `load_primitives`**. It does not save the full `domain` dictionary, conservative-state arrays, ghost cells, rule functions, boundary settings, solver options or restart metadata. In particular, there are no `save_domain` / `load_domain` functions in this revision.

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
| `initialize_state(domain, area_rule, p_rule, t_g_rule, t_l_rule, u_rule, alpha_g_rule, n=100)` | Integrates `area(x) * primitive_to_q(...)` over each physical cell and divides by volume. Initially copies edge states into ghosts. | Conservative array `(5, Nt)`. |
| `fixed_state(side)` | Converts the prescribed left/right primitive settings to a conservative external state. | `(5,)` array. |
| `apply_boundary_conditions(domain, q, case, side, check=False, rtol=1e-3)` | Updates ghosts **in place** and returns the same array. For `fixed`, optionally checks the initial edge state but does not overwrite it. | Full conservative array `(5, Nt)`. |

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
| `muscl_reconstruction(q, domain)` | Reconstructs left/right faces of every cell and reduces slopes when reconstructed pressure fails the 5,000 Pa safeguard. | `q_left, q_right`, each `(5, Nt)`. |
| `rusanov_flux(q_interface_left, q_interface_right)` | Uses two `(5,)` interface states to calculate the numerical flux and arithmetic-average interface velocity. | `flux` of shape `(5,)`, and scalar `u_star`. |
| `compute_residual(q, domain)` | Combines area-weighted face flux differences with momentum and volume-fraction sources. Requires populated ghosts. | Rate array `(5, N)` for physical cells only. |

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
explicit_euler(q_initial, domain, final_time, bc_settings, cfl=0.5, save_every=100)
rk4(q_initial, domain, final_time, bc_settings, cfl=0.5, save_every=100)
implicit_euler(q_initial, domain, final_time, bc_settings, tau, save_every=100, eps_tol=1e-6, limit=1000, relax_factor=0.25)
crank_nicolson(q_initial, domain, final_time, bc_settings, tau, save_every=100, eps_tol=1e-6, limit=1000, relax_factor=0.25)
```

All return `(q_final, history)`, apply boundary conditions during evolution, collect primitive snapshots, and print a final mass/energy comparison. Calling a solver directly does not create output files; `save_primitives` performs that separate step.

`get_time_step(u, c, dx, cfl=0.5)` is defined in both explicit-method modules. Supply matching physical-cell arrays; it returns one scalar time step [s].

`fixed_point_euler(...)` and `fixed_point_CN(...)` each accept `q_initial, tau, domain, bc_settings, eps_tol=1e-6, limit=1000, relax_factor=0.25, time=0.0`. They solve **one** time step and return the new full conservative state, not a residual or history. The outer integrator must not multiply their return value by `tau` again. `time` is only a progress-display label.

### Conservation diagnostic

`conservation_check(q_initial, q_final, domain)` integrates over physical cells:

$$
M=\sum_j \rho_j V_j,\qquad \mathcal E=\sum_j\rho_j E_j V_j.
$$

It prints and returns `(mass_error, energy_error)`, the signed percentage changes `100 * (final - initial) / initial`. These are global changes, not norms or error bounds. For open/fixed boundaries, mass or energy may legitimately enter or leave; the routine does not subtract boundary flux contributions.

## Numerical method and convergence

The spatial operator is `dq/dt = R(q)`. Rusanov dissipation uses the maximum of `abs(u) + c` at the two interface states. The residual includes the area-change momentum source and the volume-fraction source based on interface velocities.

For backward Euler, the relaxed fixed-point update is

$$
q^{(m+1)}=(1-\omega)q^{(m)}+\omega\left[q^n+\tau R(q^{(m)})\right].
$$

For Crank–Nicolson, it is

$$
q^{(m+1)}=(1-\omega)q^{(m)}+\omega\left[q^n+\frac{\tau}{2}\left(R(q^n)+R(q^{(m)})\right)\right].
$$

Here `omega = relax_factor`, `n` indexes physical time steps, and `m` indexes inner iterations. The old state `q_n` remains fixed; Crank–Nicolson also calculates `R(q_n)` once per step.

Both nonlinear solvers check the relative Euclidean norm of the successive-iterate change separately in each physical cell:

$$
\epsilon_j=\frac{\lVert q_j^{(m+1)}-q_j^{(m)}\rVert_2}{\max(\lVert q_j^{(m+1)}\rVert_2,\mathrm{tiny})}.
$$

Every physical cell must satisfy `epsilon_j < eps_tol`. This is a **five-component vector-norm check**, not five separate relative tests and not a direct backward Euler/Crank–Nicolson equation-defect check. Component scales and the relaxation factor affect the reported change. Failure to meet tolerance within `limit` raises `RuntimeError`; the normal main workflow then does not proceed to saving a completed run.

MUSCL uses minmod slopes. When necessary, a bisection search reduces both face deviations of a cell by the same factor between zero and one to satisfy the pressure safeguard. This avoids abruptly discarding the full slope when only a small reduction is needed. If the cell-average state itself fails the safeguard, the code falls back to its cell-average face states; it does not repair that average. The 5,000 Pa value is a numerical safeguard in this implementation, not a universal physical threshold.

## Common mistakes and limitations

| Symptom or question | What to check |
| --- | --- |
| Files appear in an unexpected place | `Path.cwd()` determines the data directory. Run from `python`, or use an explicit path when loading. |
| Loading opens an older run | A new save may have created `_1`, `_2`, etc. Use the returned `results_folder` or the exact suffixed name. |
| Array-size mismatch after loading | Loaded arrays already exclude ghosts. Do not apply `domain["physical_ind"]` again. |
| `simple_plot` reports a missing key | Pass the full history, and use a key from the primitive table. A single snapshot dictionary is not a history. |
| The requested plot time differs from its title | The nearest saved time is selected. Reduce `save_every` for more temporal samples. |
| Pressure appears too large for a bar-scale plot | Stored pressure is in Pa. Divide by `1e5` in a custom plot if required. |
| Fixed boundary rejects the mesh | Set the corresponding ghost count to zero and edit `fixed_state(side)`. |
| Importing `main` unexpectedly starts a simulation | This file executes at module level. Import individual functions from their modules instead. |
| Can a saved folder reproduce the mesh or restart the solver? | Not by itself in this revision: geometry/settings and conservative/ghost states are not saved. |

The configured RK4 time-step estimate uses the same CFL expression as explicit Euler; this is an implementation choice, not an automatic stability guarantee for every case. Likewise, reducing a nonlinear tolerance does not replace spatial/time-step refinement studies. Existing result folders are examples for analysis, not an automated validation suite.
