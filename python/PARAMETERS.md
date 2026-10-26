# Configuring and running the solver

Install NumPy, SciPy and Matplotlib. From `python/`, run `python main.py`. Edit the settings at the top of main; the shared build → initialise → solve → save → reload → plot workflow follows below. Close the domain plot to continue.

## Solver selection

Set `solver_name` to one of these names. All four functions receive the initial conservative state, domain, final time, boundary settings, `save_every`, and `reconstruction`.

| Solver | Time step | Additional options |
| --- | --- | --- |
| `explicit_euler` | `cfl`, set to 0.1 in main | None |
| `rk4` | `cfl`, set to 0.1 in main | None |
| `implicit_euler` | Required `tau` in seconds | Nonlinear controls below |
| `crank_nicolson` | Required `tau` in seconds | Nonlinear controls below |

The selected solver in main is implicit Euler. Implicit solvers have no CFL argument. Every integrator shortens its final step to reach `final_time`. All return `(q_final, history)`.

`reconstruction="muscl"` is the default. Select `"first_order"` for piecewise constant face states. The option is passed consistently into nonlinear iterations and Jacobian evaluation.

## Nonlinear controls

| Option | Default | Meaning |
| --- | --- | --- |
| `nonlinear_method` | `"newton"` | Choose Newton or `"fpi"`. |
| `eps_tol` | `1e-6` | Every component-scaled unrelaxed correction must be below this in every physical cell. |
| `limit` | `1000` | Maximum iterations within each physical step. |
| `relax_factor` | `1.0` | Unrelaxed for Newton and FPI. Supply a value in (0, 1] to change it. |
| `eps_j` | `1e-6` | Finite-difference perturbation factor for Newton. |
| `q_reference` | `None` | Automatic perturbation scales, or five positive custom values. Does not change convergence scales. |
| `jacobian_every` | `3` | Rebuild/factorise on iterations 1, 4, 7, …; use 1 for every iteration. |

The last three options affect Newton only. Main explicitly uses `jacobian_every=10`, while direct solver calls default to 3. FPI and Newton for IE and CN remain four separate full functions. They return the new full conservative state, including ghosts, not a rate. There is no backtracking or automatic step retry. Implicit time stability does not guarantee nonlinear convergence.

Convergence scales come from the old state once per step: maximum gas/liquid partial densities floored at 1e-3 kg/m³, maximum rho*(abs(u)+c) for momentum, maximum absolute energy density floored at 1 J/m³, and 1 for alpha. The maximum absolute unrelaxed correction divided by its scale is checked in each cell. See the root README for equations and limitations.

## Geometry, initial and boundary conditions

Edit domain rules in `domain_rules.py`; edit initial profiles and `fixed_state` in `conditions.py`. Initial averages use `initial_integration_intervals` midpoint samples per cell. All boundaries now require at least one ghost per side, including fixed boundaries. Fixed ghosts contain the reservoir state and the physical edge remains an evolved cell. Periodic boundaries support successive wrapped ghosts; select periodic on both sides.

MUSCL reconstructs area-weighted momentum and converts it back at faces. Its pressure safeguard uses literal 5000 Pa. Select `pressure_fallback="first_order"` (default) to replace failed face states by their cell average, or `"gradual"` for the previous 40-step reduction. There is no p_min option.

## History, files and plots

`save_every` counts completed physical time steps, not nonlinear iterations. Initial and final snapshots are always included. In-memory `q` is `(5, total_cells)` and history primitives are `(saved_times, total_cells)`. Time is stored in `history["times"]`.

`save_primitives(history, domain, folder_name)` writes physical cells only under the current working directory's `data/`, adding suffixes to avoid overwriting. `load_primitives(folder_or_path)` returns `(cell_centres, history)` with primitive arrays `(saved_times, physical_cells)`. Do not apply `physical_ind` again after loading.

Set `run_simulation=False` to load and plot an existing run. `plot_requests` contains `(names, time, xlim)` entries; two primitive names use two y axes, the second red. The nearest saved time is selected. SI units are retained.

The current I/O saves primitive histories and physical coordinates, not full domain geometry or restart states. Plotting the mesh still requires the domain and area rule. Conservation diagnostics integrate the conservative arrays directly and report phase mass, mixture mass and total energy changes; open boundaries can exchange these quantities.

## Comparing MUSCL pressure fallbacks

In main, keep `reconstruction="muscl"` and select `pressure_fallback="first_order"` (default, assignment fallback) or `pressure_fallback="gradual"` (previous bisection). The option passes through every solver and Newton Jacobian evaluation. `reconstruction="first_order"` instead disables MUSCL everywhere and makes the pressure-fallback selection inactive. The threshold remains literal 5000 Pa.

All four nonlinear one-step functions and both implicit time integrators now default to `relax_factor=1.0`. Set e.g. `relax_factor=0.25` explicitly to damp updates. Main exposes this option in each implicit block. The component-scaled unrelaxed convergence check remains the default; `convergence_check="teacher"` selects the full-matrix norm ratio described below.

All four integrators print minimum pressure as `min_p` at the existing summary interval and final time, alongside maximum pressure and speed, using physical cells only. These are accepted cell-average pressures, not reconstructed face pressures or intermediate nonlinear iterates. A positive cell minimum does not prove no reconstructed face crossed the safeguard. Values between summary intervals are not reported by this diagnostic.

For a controlled comparison, keep the mesh, tau, initial conditions, nonlinear method, tolerance and Jacobian interval fixed; change only the fallback. Use `save_every=1` to see pressure at every accepted time step. Default relaxation one does not guarantee FPI convergence for an arbitrary time step.

Thermodynamic recovery and periodic Jacobian grouping are optimized internally; no new user settings are required. `jacobian_every` and all default numerical controls remain unchanged. See the root README for the two minimal recovery helpers and verification commands.

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
