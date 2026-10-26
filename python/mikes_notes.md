# Errors and spec deviations in the Python code (`Project_pyton/python`)

These are flagged only. **The Python code has not been modified**, and the MATLAB port reproduces these behaviours on purpose.
Each entry names the file and function, quotes the code marked **ERROR**, and gives a **Corrected** version (not applied).
Severity: **[major]** wrong results or unsound behaviour in some supported configuration · **[minor]** small or latent effect ·
**[deviation]**  deliberate departure from the PDF that should be justified · **[doc]** documentation only.

---

## P-01 · `solver_functions.py` → `muscl_reconstruction` — momentum is not reconstructed as A·ρu for variable area  [major]
PDF p.20–21 (§6.1.1) requires reconstructing (Aρu)_j with the cell-average area and dividing the face values by the face area.
Otherwise the area change alone creates artificial momentum jumps at the faces. The code reconstructs ρu directly:

**ERROR:**
```python
sigma[:, 1:-1] = minmod(delta_left, delta_right)
q_left = q - sigma*(x - faces[:-1])
q_right = q + sigma*(faces[1:] - x)
```
**Corrected:**
```python
q_rec = q.copy()
q_rec[2] *= domain["area_average"]                 # (A rho u)_j, A_j = V_j / dx_j
delta_left = (q_rec[:, 1:-1] - q_rec[:, :-2])/(x[1:-1] - x[:-2])
delta_right = (q_rec[:, 2:] - q_rec[:, 1:-1])/(x[2:] - x[1:-1])
sigma[:, 1:-1] = minmod(delta_left, delta_right)
q_left = q_rec - sigma*(x - faces[:-1])
q_right = q_rec + sigma*(faces[1:] - x)
q_left[2] /= domain["area_left"]                   # back to rho u at the faces
q_right[2] /= domain["area_right"]
# (the pressure safeguard below must then use q_left/q_right after this division,
#  and the fallback "q_left[:, j] = q[:, j]" keeps using the un-weighted q)
```
Effect: none for constant area (Task 15/Groups 4–6), but the nozzle cases (Groups 1–3, 8) are affected.

My comment: Yes, this needs to be fixed for variable area

---

## P-02 · `nonlinear_solvers.py` → `fixed_point_euler`, `fixed_point_CN`, `newton_euler`, `newton_CN` — convergence test is blind to α and momentum  [major]
**ERROR:**
```python
difference = np.linalg.norm(q_next[:, j] - q_m[:, j], 2)
magnitude = np.linalg.norm(q_next[:, j], 2)
cell_errors[i] = difference/max(magnitude, np.finfo(float).tiny)
...
if np.all(cell_errors < eps_tol):
```
The five components of q differ by about 9 orders of magnitude. In the Task 15 initial state, a typical cell has
q = [1.19, 1254, 0, 1.05e8, 0.1] (kg/m³, kg/m³, kg/(m² s), J/m³, –), so ‖q‖ ≈ 1.05e8 comes entirely from ρE. With
eps_tol = 1e-6 the test accepts **absolute** changes of about 100 in every component. That means 8 % of the liquid partial density,
about 100 kg/(m² s) of momentum, and any change at all in α_g (whose range is only 0–1). Convergence is in practice tested on the energy alone.
The PDF formula (p.38) has the same weakness; see also R-22 in `resource_errors.md`.

With relaxation there is a second problem: q_next − q_m = ω·(q̃ − q_m), so with ω = 0.25 the test is effectively 4× looser than eps_tol.

**Corrected** (component-scaled and relaxation-independent, still per cell):
```python
scale = np.array([max(np.max(np.abs(q_n[0, physical])), 1e-3),      # kg/m3
                  max(np.max(np.abs(q_n[1, physical])), 1e-3),      # kg/m3
                  reference_state(q_n, domain)[2],                  # rho (|u| + c), kg/(m2 s)
                  np.max(np.abs(q_n[3, physical])),                 # J/m3
                  1.0])                                             # alpha
step = (q_next[:, physical] - q_m[:, physical]) / relax_factor      # unrelaxed update
cell_errors = np.max(np.abs(step) / scale[:, None], axis=0)
```

My comment: This is correct, but not how the project asks us to define it. Not sure if we should follow this or that.

---

## P-03 · `solver_functions.py` → `muscl_reconstruction` — pressure safeguard differs from the PDF  [deviation]
PDF p.20: "If pressure falls below p_min then revert to first order scheme (zero slope)": Q^{L/R}_j = Q_j otherwise.
The code instead scales both face deviations of the cell by the largest θ ∈ [0, 1] that keeps both faces ≥ 5000 Pa
(40-step bisection). It falls back to θ = 0 only when the cell average itself fails. This is the group's fix for
`Resources/FPI_issue`, introduced in commit `cb75b80`.

**ERROR** (versus the spec):
```python
lower, upper = 0.0, 1.0
for _ in range(40):
    theta = 0.5*(lower + upper)
    if valid_faces(q[:, j] + theta*left_change, q[:, j] + theta*right_change): lower = theta
    else: upper = theta
q_left[:, j] = q[:, j] + lower*left_change
```
**Corrected (spec-literal version):**
```python
if not valid_faces(q_left[:, j], q_right[:, j]):
    q_left[:, j] = q[:, j]
    q_right[:, j] = q[:, j]
```
The deviation is justified (it keeps the reconstruction continuous in Q, see `FPI_issue_solution.md`), but it should be
documented in the report. Two related details: the threshold is hard-coded (`5000`, two places) instead of being a parameter p_min, and
the test is `>= 5000` while the PDF uses `> p_min`.

My comment: Yeah, the chatbot understood correctly what happened. Yeah, in the end we can put p_min instead of 5000.

---

## P-04 · `conditions.py` → `initialize_state` (with `p_rule`) — discontinuity on a cell face pollutes the edge cells  [minor]
**ERROR:**
```python
def p_rule(x):
    if 4 < x < 6: return 500e5
...
x = np.linspace(domain["x_faces"][j], domain["x_faces"][j + 1], n + 1)       # includes both faces
traps = 0.5*(area_q[:-1] + area_q[1:])*h
```
With 500 cells, x = 4 and x = 6 are exactly cell faces. The trapezoidal rule evaluates the 10-bar state at the face
point of the first and last high-pressure cells (the strict `<` puts x = 4 outside). The averages of cells [4.00, 4.02] and [5.98, 6.00]
are therefore mixed: p = 497.55 bar instead of 500 bar (−0.49 %), and m_g −0.49 %. The 10-bar side is unaffected.
**Corrected** (midpoint rule, so no sample sits on a face; same cost):
```python
h = domain["dx"][j] / n
x = domain["x_faces"][j] + (np.arange(n) + 0.5)*h
area_q = np.array([area_rule(xi)*primitive_to_q(p_rule(xi), u_rule(xi), t_g_rule(xi), t_l_rule(xi), alpha_g_rule(xi)) for xi in x])
q[:, j] = h*np.sum(area_q, axis=0) / domain["volumes"][j]
```

---

## P-05 · `conditions.py` → `apply_boundary_conditions` — periodic BC is wrong for more than one ghost cell  [minor, latent]
**ERROR:**
```python
elif case == "periodic":
    q[:, ghost] = q[:, opposite]          # every ghost copies the same opposite edge cell
```
With n_ghost > 1 the ghosts must copy *successive* cells from the other end. (main.py says to use one ghost cell,
so this is latent.)
**Corrected:**
```python
elif case == "periodic":
    source = physical[-1 - i] if side == "left" else physical[i]
    q[:, ghost] = q[:, source]
```

---

## P-06 · `miscellaneous.py` → `conservation_check` — conservation measured on bounded primitives, not on Q  [minor]
**ERROR:**
```python
primitives_i = get_primitives(q_initial)
...
total_mass_i = np.sum(cell_volumes*cell_densities_i)
total_e_i = np.sum(cell_volumes*cell_densities_i*cell_tot_energies_i)
```
`q_to_primitive` clips α_g to [1e-8, 1−1e-8] and the masses to ≥ 1e-12 before computing ρ and E. The diagnostic therefore
measures a modified state. It also converts every cell (19 EOS-derived fields) just to recover sums of Q.
**Corrected:**
```python
v = domain["volumes"][physical_cell_ind]
total_mass_i = np.sum(v*(q_initial[0, physical_cell_ind] + q_initial[1, physical_cell_ind]))
total_e_i = np.sum(v*q_initial[3, physical_cell_ind])
# same for q_final
```
For mixture problems, the per-phase masses (Σ V α₁ρ₁ and Σ V α₂ρ₂) are also conserved separately and are worth reporting.

Very good point, this can speed stuff up.

---

## P-07 · `solver_functions.py` → `compute_residual` / `muscl_reconstruction` — "fixed" boundary loses the edge slope  [minor]
**ERROR:** With a fixed side the domain has no ghost cell there, so the physical edge cell is the first (or last) array column.
`sigma[:, 0]` / `sigma[:, -1]` are forced to zero, so that cell silently falls back to first order:
```python
sigma[:, 1:-1] = minmod(delta_left, delta_right)    # columns 0 and -1 always get zero slope
```
**Corrected:** Keep one ghost cell on fixed sides too, filled with `fixed_state(side)` as in PDF §8.1 ("Q₁ = Q_BC"), and drop
the special case in `compute_residual`:
```python
if case == "fixed":
    for ghost in ghosts: q[:, ghost] = fixed_state(side)
```

---

## P-08 · `solver_functions.py` → `muscl_reconstruction` — no first-order reconstruction option  [minor, missing feature]
PDF §6.1.2 (p.21): "Implement it in your solver as an alternative to MUSCL scheme." The code has no switch.
**Corrected:** add a `reconstruction="muscl" | "first_order"` option. For first order, `return q.copy(), q.copy()`.

---

## P-09 · `PARAMETERS.md` — out of date with the code  [doc]
**ERROR:**
> "The generic `nonlinear_solver` parameter was removed from implicit Euler: it now explicitly uses the backward Euler
> fixed-point solver. The existing `fixed_point_CN` helper is retained, but is not a registered time integrator … it still
> returns a rate"
> Table row: "`implicit_euler` | Required `tau`; `eps_tol`, `limit`, `relax_factor` for relaxed fixed-point iteration."

**Corrected:** `implicit_euler` and `crank_nicolson` both take `nonlinear_method="newton"` (default) or `"fpi"`, plus
`eps_j`, `q_reference` and `jacobian_every` for Newton. `crank_nicolson` is registered in `solvers.py`, and `fixed_point_CN` returns the
new state, not a rate. (README.md is up to date.)

---

## P-10 · `main.py` — default settings differ from the Task 15 specification  [doc/config]
**ERROR:** `cfl = 0.5` (explicit Euler/RK4) and the comment `# Rebuild Jacobianon every nth Newton iteration`.
**Corrected:** Task 15 asks for `cfl = 0.1` (this is needed to confirm dt_end = 0.0025 ms), and "Jacobian on".
(CFL 0.5 is stable here and gives the same final state within the scheme's error. The point is reproducibility of the requested numbers.)

---

### Checked and found consistent with the PDF
EOS (`sg_eos`, all requests, including the Task 5 values), the pressure recovery and primitive recovery algorithm (§4.2), Wood's sound speed and K,
the minmod limiter and face distances (correct on non-uniform meshes, unlike PDF R-08), the Rusanov flux and u* (§6.2.2), the residual with
Option #2 sources (§6.3), the reflective/zero-gradient ghosts, the explicit Euler and RK4 stages (§9.1), the backward-Euler/CN fixed-point and Newton
equations, J_G = I − τJ_R (or I − τ/2·J_R), the 5-colour Jacobian with the j±2 stencil and the actual-perturbation divisor (§9.2.4).
The MATLAB port reproduces all of them to round-off (see `Project_matlab/README.md`, "Verification").
