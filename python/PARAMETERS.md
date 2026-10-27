# Settings reference

Edit `main.py`, then run `python main.py` from the `python` directory. Inputs use SI units. The full numerical and data-format explanation is in the root README.

## Case and workflow

- `solver_name`: `"explicit_euler"`, `"rk4"`, `"implicit_euler"`, `"crank_nicolson"`.
- `x_bounds`: increasing pair; `n_cells`: positive integer; `n_ghost`: positive integer or pair for these solvers. Reflective ghost counts must not exceed physical cells.
- `node_density_rule` and `area_rule`: positive functions; choose in `domain_rules.py`.
- Initial rules in `conditions.py`: pressure/temperatures positive; gas fraction strictly between zero and one.
- `bc_settings`: each side is `"reflective"`, `"zerograd"`, `"fixed"` or `"periodic"`; use periodic on both sides. Define fixed states in `conditions.py`.
- `final_time >= 0`; `save_every` is a positive integer counting accepted physical steps.
- `reconstruction`: `"muscl"` or `"first_order"`.
- `pressure_fallback`: `"first_order"` or `"gradual"`; same hard-coded 5000 Pa threshold, used only by MUSCL.
- `run_simulation=False` loads `load_folder` for plotting. `folder_name` is a bare new-result name; existing folders get numeric suffixes.
- `plot_requests`: `(one primitive name or two names, time, None or [xL, xR])`. The nearest saved time is selected.

## Explicit Euler / RK4

`cfl > 0` sets `dt = cfl*min(dx/(abs(u)+c))`; main uses 0.1. `save_every` controls snapshots and summary printing. Boundary conditions are refreshed between RK stages.

## Implicit Euler / Crank–Nicolson

| Argument | Function default | Meaning / required input |
| --- | --- | --- |
| `tau` | Required | Positive initial step [s], also the recovery ceiling. |
| `nonlinear_method` | `"newton"` | `"newton"` or `"fpi"`. |
| `eps_tol` | `1e-6` | Positive tolerance for the selected check. |
| `convergence_check` | `"scaled"` | `"scaled"` or `"teacher"`; main currently selects teacher. |
| `limit` | `1000` | Positive maximum inner iteration count; main uses 200. |
| `relax_factor` | `1.0` | Fixed relaxation in (0, 1]. |
| `eps_j` | `1e-6` | Positive Newton perturbation fraction. |
| `q_reference` | `None` | Automatic scales, or five positive values in Q order. |
| `jacobian_every` | `3` | Positive inner rebuild interval; main uses 10. |
| `jacobian_step_every` | `1` | Positive outer refresh interval; 1 means every time step. |
| `min_tau` | `1e-12` | Positive minimum allowed after halving; <= tau. |
| `max_retries` | `12` | Nonnegative maximum halvings per physical step. |
| `adaptive_tau` | `False` | False: retain halvings. True: permit step recovery. |
| `growth_after` | `3` | Positive consecutive easy-step count before doubling. |
| `growth_iterations` | `8` | Positive maximum iterations for an easy step. |
| `growth_rate` | `0.5` | Required recent contraction factor, between zero and one. |

Newton-specific arguments are ignored by FPI. No validation is done for configuration typos: use the exact choices above. Physical and numerical failure checks remain.

## Matrix reuse

`jacobian_every=10` rebuilds on inner iterations 1, 11, 21, ... . `jacobian_step_every=5` requests fresh starts on outer steps 1, 6, 11, ...; otherwise iteration 1 reuses the last stored factorisation. Inner rebuilds still happen. Any time-step change clears the cache. A failed attempt with only an inherited matrix gets a fresh-matrix retry before halving. Set both intervals to 1 for a fresh matrix every iteration.

## Time-step policy and diagnostics

The default retains the reduced step for the entire remaining run. With `adaptive_tau=True`, three consecutive easy solves double the next step, capped by the original tau. Easy means no failed attempt, iterations <= growth_iterations and recent contraction <= growth_rate. One-iteration solves qualify without a measured rate. Failures reset the easy-step count. This is an effort heuristic, not an accuracy estimate.

CFL is calculated from the accepted start state and actual step. Printed `r` is the geometric mean of the last up to three error ratios. A small value means rapid error decrease, not guaranteed physical/time accuracy. `NaN` means no ratio yet.

`history["diagnostics"]` contains arrays `times`, `dt`, `cfl`, `iterations`, `error`, `rate`, `retries` and `jacobian_builds` for every accepted step. Primitive snapshots still follow save_every and include initial/final states. The diagnostics are saved/restored separately as `diagnostics.npz`; old result folders still load.

## Stopping definitions (unchanged)

- Scaled: max over physical components/cells of abs(unrelaxed correction)/component scale.
- Teacher: `np.linalg.norm(q_next-q_m, 2)/np.linalg.norm(q_next, 2)` on full matrices including updated ghosts. This is the spectral norm; damping influences this error.

`q_reference` changes only Newton perturbation scales, not convergence scales. All four nonlinear functions return the accepted full Q. Optional `info={}` receives iteration diagnostics; Newton additionally accepts the outer loop's `jacobian_cache`. Ordinary users only set the controls in main.

Run focused checks with `python tests/check_implicit_diagnostics.py` from the repository root.
