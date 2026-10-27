"""Shared thermodynamics, reconstruction, flux and residual calculations."""

import numpy as np

class InvalidStateError(ValueError):
    """A conservative state is not thermodynamically admissible."""


def validate_state(q, domain):
    """Check physical cells without clipping or modifying their conservative values."""
    for j in domain["physical_ind"]:
        state = q[:, j]
        reason = None

        if not np.all(np.isfinite(state)):
            reason = "non-finite conservative component"
        elif state[0] <= 0 or state[1] <= 0:
            reason = "partial masses must be positive"
        elif not 0 < state[4] < 1:
            reason = f"gas volume fraction must be in (0, 1), got {state[4]:.6g}"
        else:
            try:
                # Direct EOS call: no clipping of masses or volume fraction.
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    pressure = sg_eos("p_common", q=state)

                if not np.isfinite(pressure) or pressure <= 0:
                    reason = f"pressure must be finite and positive, got {pressure:.6g} Pa"

            except ValueError as error:
                reason = str(error)

        if reason is not None:
            raise InvalidStateError(f"Cell {j}, x={domain['x_centres'][j]:.6g} m: {reason.rstrip('.')}.")


def sg_eos(request, k=None, e=None, rho=None, p=None, temperature=None, q=None):
    """
    Get a thermodynamic property using the stiffened-gas EOS.

    Phase: k=1 for gas, k=2 for liquid.
    All inputs must use SI units.

    Checks thermodynamic conditions:
        rho > 0, p + p_inf > 0, temperature > 0.
    """

    # gamma, p_inf [Pa], q [J/kg], Cv [J/(kg K)]
    parameters = {1: (1.4, 0, 0, 718),
                  2: (2.35, 1e9, -1.167e6, 1816)}

    if request == "p_common":
        m1, m2, momentum, energy, alpha1 = q
        alpha2 = 1 - alpha1

        if not np.all(np.isfinite(q)): raise ValueError("Q contains a non-finite value.")
        if m1 <= 0 or m2 <= 0: raise ValueError("Partial masses must be positive.")
        if alpha1 <= 0 or alpha1 >= 1: raise ValueError("Volume fraction must satisfy 0 < alpha1 < 1.")

        gamma1, p_inf1, q1, c_v1 = parameters[1]
        gamma2, p_inf2, q2, c_v2 = parameters[2]

        rho = m1 + m2
        u = momentum / rho
        rho_e = energy - 0.5*rho*u**2

        numerator = (rho_e - m1*q1 - m2*q2 - alpha1*gamma1*p_inf1 / (gamma1 - 1) - alpha2*gamma2*p_inf2 / (gamma2 - 1))
        denominator = (alpha1 / (gamma1 - 1) + alpha2 / (gamma2 - 1))
        p = numerator / denominator

        # check density, shifted pressure and temperature for each phase
        rho1 = m1 / alpha1
        rho2 = m2 / alpha2

        sg_eos("T_prho", k=1, p=p, rho=rho1)
        sg_eos("T_prho", k=2, p=p, rho=rho2)

        return p

    gamma, p_inf, q, c_v = parameters[k]

    if request == "rho_pT":
        # check inputs before dividing by temperature
        if not np.isfinite(temperature) or temperature <= 0: raise ValueError(f"Phase {k}: temperature must be positive.")
        if not np.isfinite(p) or p + p_inf <= 0: raise ValueError(f"Phase {k}: p + p_inf must be positive.")

        rho = (p + p_inf) / ((gamma - 1)*c_v*temperature)

    elif request == "p_erho":
        # Check density before recovering pressure and temperature
        if not np.isfinite(rho) or rho <= 0: raise ValueError(f"Phase {k}: density must be positive.")
        p = (gamma - 1)*rho*(e - q) - gamma*p_inf

    elif request not in ("e_prho", "T_prho", "h_prho", "c_prho"): raise ValueError("Unknown request.")

    # Check the resulting thermodynamic state
    if not np.isfinite(rho) or rho <= 0: raise ValueError(f"Phase {k}: density must be positive.")
    if not np.isfinite(p) or p + p_inf <= 0: raise ValueError(f"Phase {k}: p + p_inf must be positive.")

    temperature = (p + p_inf) / ((gamma - 1)*rho*c_v)

    if not np.isfinite(temperature) or temperature <= 0: raise ValueError(f"Phase {k}: temperature must be positive.")

    if request == "p_erho":
        return p

    elif request == "rho_pT":
        return rho

    elif request == "e_prho":
        return q + (p + gamma*p_inf) / ((gamma - 1)*rho)

    elif request == "T_prho":
        return temperature

    elif request == "h_prho":
        return q + gamma*(p + p_inf) / ((gamma - 1)*rho)

    elif request == "c_prho":
        return np.sqrt(gamma*(p + p_inf) / rho)


def primitive_to_q(p, u, t_g, t_l, alpha_g):
    """
    Convert primitive variables to the conservative vector.
    Inputs: p [Pa], u [m/s], t_g and t_l [k], alpha_g [-].
    """
    alpha_l = 1 - alpha_g
    # densities and specific internal energies by phase
    rho_g = sg_eos("rho_pT", k=1, p=p, temperature=t_g)
    rho_l = sg_eos("rho_pT", k=2, p=p, temperature=t_l)
    e_g = sg_eos("e_prho", k=1, p=p, rho=rho_g)
    e_l = sg_eos("e_prho", k=2, p=p, rho=rho_l)
    # partial masses per unit mixture volume
    m_g = alpha_g*rho_g
    m_l = alpha_l*rho_l
    rho = m_g + m_l
    momentum = rho*u
    energy = m_g*e_g + m_l*e_l + 0.5*rho*u**2
    return np.array([m_g, m_l, momentum, energy, alpha_g])


def q_to_pressure(q, eps_alpha=1e-8, eps_m=1e-12):
    """Recover only pressure, using the same bounds and EOS checks as full recovery."""
    m_g, m_l, momentum, energy, alpha_g = q
    alpha_g = min(1 - eps_alpha, max(eps_alpha, alpha_g))
    m_g = max(eps_m, m_g)
    m_l = max(eps_m, m_l)
    q_bounded = np.array([m_g, m_l, momentum, energy, alpha_g])
    return sg_eos("p_common", q=q_bounded)


def q_to_flow(q, eps_alpha=1e-8, eps_m=1e-12):
    """Recover only the quantities used by fluxes and sources.

    Full primitive dictionaries remain available through q_to_primitive.
    Keep the original arithmetic and EOS checks for pressure and sound speeds.
    """
    m_g, m_l, momentum, energy, alpha_g = q
    alpha_g = min(1 - eps_alpha, max(eps_alpha, alpha_g))
    alpha_l = 1 - alpha_g
    m_g = max(eps_m, m_g)
    m_l = max(eps_m, m_l)
    rho = m_g + m_l
    u = momentum / rho
    q_bounded = np.array([m_g, m_l, momentum, energy, alpha_g])
    p = sg_eos("p_common", q=q_bounded)
    rho_g = m_g / alpha_g
    rho_l = m_l / alpha_l
    c_g = sg_eos("c_prho", k=1, p=p, rho=rho_g)
    c_l = sg_eos("c_prho", k=2, p=p, rho=rho_l)
    c = np.sqrt(1 / (rho*(alpha_g/(rho_g*c_g**2) + alpha_l/(rho_l*c_l**2))))
    k = alpha_g*alpha_l*(rho_l*c_l**2 - rho_g*c_g**2) / (alpha_l*rho_g*c_g**2 + alpha_g*rho_l*c_l**2)
    return {"p": p, "u": u, "c": c, "alpha_g": alpha_g, "k": k}


def q_to_primitive(q, eps_alpha=1e-8, eps_m=1e-12):
    """
    Recover primitive variables from q.
    Returns a dictionary. All quantities use SI units.
    """
    m_g, m_l, momentum, energy, alpha_g = q
    # bound volume fractions and partial masses
    alpha_g = min(1 - eps_alpha, max(eps_alpha, alpha_g))
    alpha_l = 1 - alpha_g
    m_g = max(eps_m, m_g)
    m_l = max(eps_m, m_l)
    # mixture properties
    rho = m_g + m_l
    u = momentum / rho
    e_total = energy / rho
    e = e_total - 0.5*u**2
    # pressure recovery
    q_bounded = np.array([m_g, m_l, momentum, energy, alpha_g])
    p = sg_eos("p_common", q=q_bounded)
    # density by phase
    rho_g = m_g / alpha_g
    rho_l = m_l / alpha_l
    # thermodynamic properties by phase
    t_g = sg_eos("T_prho", k=1, p=p, rho=rho_g)
    t_l = sg_eos("T_prho", k=2, p=p, rho=rho_l)
    e_g = sg_eos("e_prho", k=1, p=p, rho=rho_g)
    e_l = sg_eos("e_prho", k=2, p=p, rho=rho_l)
    h_g = sg_eos("h_prho", k=1, p=p, rho=rho_g)
    h_l = sg_eos("h_prho", k=2, p=p, rho=rho_l)
    c_g = sg_eos("c_prho", k=1, p=p, rho=rho_g)
    c_l = sg_eos("c_prho", k=2, p=p, rho=rho_l)
    # speed of sound
    c = np.sqrt(1 / (rho*(alpha_g/(rho_g*c_g**2) + alpha_l/(rho_l*c_l**2))))
    # compressibility coefficient
    k = alpha_g*alpha_l*(rho_l*c_l**2 - rho_g*c_g**2) / (alpha_l*rho_g*c_g**2 + alpha_g*rho_l*c_l**2)

    primitive = {
        "p": p, "u": u,
        "t_g": t_g, "t_l": t_l,
        "alpha_g": alpha_g, "alpha_l": alpha_l,
        "rho": rho, "rho_g": rho_g, "rho_l": rho_l,
        "e": e, "e_total": e_total,
        "e_g": e_g, "e_l": e_l,
        "h_g": h_g, "h_l": h_l,
        "c_g": c_g, "c_l": c_l,
        "c": c, "k": k
    }

    return primitive


def get_primitives(q):
    """Convert q internally with q_to_primitive; return arrays including ghosts."""
    first = q_to_primitive(q[:, 0])

    primitives = {name: np.zeros(q.shape[1]) for name in first}

    for j in range(q.shape[1]):
        cell = first if j == 0 else q_to_primitive(q[:, j])
        for name in primitives:
            primitives[name][j] = cell[name]

    return primitives


def minmod(a, b):
    return np.where(a*b > 0, np.sign(a)*np.minimum(np.abs(a), np.abs(b)), 0)


def muscl_reconstruction(q, domain, pressure_fallback="first_order"):
    """Reconstruct A*rho*u, then recover rho*u using the corresponding face area.

    Other components are reconstructed directly. Keeps the existing 5000 Pa
    pressure threshold; choose first-order fallback or gradual reduction.
    """
    if pressure_fallback not in ("first_order", "gradual"): raise ValueError("pressure_fallback must be first_order or gradual.")
    x = domain['x_centres']
    faces = domain['x_faces']
    sigma = np.zeros_like(q, dtype=float)
    q_rec = np.array(q, dtype=float, copy=True)
    q_rec[2] *= domain["area_average"]
    delta_left = (q_rec[:, 1:-1] - q_rec[:, :-2])/(x[1:-1] - x[:-2])
    delta_right = (q_rec[:, 2:] - q_rec[:, 1:-1])/(x[2:] - x[1:-1])
    sigma[:, 1:-1] = minmod(delta_left, delta_right)
    q_left = q_rec - sigma*(x - faces[:-1])
    q_right = q_rec + sigma*(faces[1:] - x)
    q_left[2] /= domain["area_left"]
    q_right[2] /= domain["area_right"]

    def valid_faces(left, right):
        try:
            return q_to_pressure(left) >= 5000 and q_to_pressure(right) >= 5000
        except ValueError:
            return False

    for j in range(q.shape[1]):
        if valid_faces(q_left[:, j], q_right[:, j]):
            continue
        if pressure_fallback == "first_order" or not valid_faces(q[:, j], q[:, j]):
            q_left[:, j] = q[:, j]
            q_right[:, j] = q[:, j]
            continue
        left_change = q_left[:, j] - q[:, j]
        right_change = q_right[:, j] - q[:, j]
        lower, upper = 0.0, 1.0
        for _ in range(40):
            theta = 0.5*(lower + upper)
            if valid_faces(q[:, j] + theta*left_change, q[:, j] + theta*right_change):
                lower = theta
            else:
                upper = theta
        q_left[:, j] = q[:, j] + lower*left_change
        q_right[:, j] = q[:, j] + lower*right_change
    return q_left, q_right


def rusanov_flux(q_interface_left, q_interface_right):
    """Return the five-component flux and the interface velocity."""
    primitive_left = q_to_flow(q_interface_left)
    primitive_right = q_to_flow(q_interface_right)
    u_left, p_left, c_left = primitive_left["u"], primitive_left["p"], primitive_left["c"]
    u_right, p_right, c_right = primitive_right["u"], primitive_right["p"], primitive_right["c"]
    flux_left = np.array([q_interface_left[0]*u_left, q_interface_left[1]*u_left, q_interface_left[2]*u_left + p_left, (q_interface_left[3] + p_left)*u_left, q_interface_left[4]*u_left])
    flux_right = np.array([q_interface_right[0]*u_right, q_interface_right[1]*u_right, q_interface_right[2]*u_right + p_right, (q_interface_right[3] + p_right)*u_right, q_interface_right[4]*u_right])
    s = max(abs(u_left) + c_left, abs(u_right) + c_right)
    flux = 0.5*(flux_left + flux_right) - 0.5*s*(q_interface_right - q_interface_left)
    u_star = 0.5*(u_left + u_right)
    return flux, u_star


def compute_residual(q, domain, reconstruction="muscl", pressure_fallback="first_order"):
    """
    Compute dQ/dt for physical cells using source-term Option 2.

    Input q: shape (5, n_total), including populated ghost cells.
    Output rate: shape (5, n_cells), physical cells only.
    Boundary conditions must be applied before calling this function.
    All boundary types, including fixed states, require at least one ghost cell
    on each side. reconstruction selects MUSCL or piecewise-constant states.
    """

    q = np.asarray(q, dtype=float)
    if q.shape != (5, domain["n_total"]): raise ValueError("Q must have shape (5, N_total).")

    physical_ind, area, volumes, n_cells = domain["physical_ind"], domain["area_faces"], domain["volumes"], domain["n_cells"]
    ng_left, ng_right = domain["n_ghost_left"], domain["n_ghost_right"]
    if ng_left < 1 or ng_right < 1: raise ValueError("All boundaries require at least one ghost cell per side.")

    if pressure_fallback not in ("first_order", "gradual"): raise ValueError("pressure_fallback must be first_order or gradual.")

    if reconstruction == "muscl":
        q_left, q_right = muscl_reconstruction(q, domain, pressure_fallback=pressure_fallback)
    elif reconstruction == "first_order":
        q_left, q_right = q.copy(), q.copy()
    else:
        raise ValueError("reconstruction must be 'muscl' or 'first_order'.")

    first_face = physical_ind[0]
    flux = np.zeros((5, n_cells + 1))
    u_star = np.zeros(n_cells + 1)
    for i in range(n_cells + 1):
        face = first_face + i
        q_interface_left = q_right[:, face - 1]
        q_interface_right = q_left[:, face]
        flux[:, i], u_star[i] = rusanov_flux(q_interface_left, q_interface_right)

    rate = np.zeros((5, n_cells))
    for i, j in enumerate(physical_ind):
        area_left, area_right = area[j], area[j + 1]
        cell_volume = volumes[j]
        rate[:, i] = -(area_right*flux[:, i + 1] - area_left*flux[:, i]) / cell_volume
        primitive = q_to_flow(q[:, j])
        p, alpha_g, k = primitive["p"], primitive["alpha_g"], primitive["k"]
        # momentum source
        rate[2, i] += p*(area_right - area_left) / cell_volume
        # volume-fraction source, using interface velocities
        div_area_u = (area_right*u_star[i + 1] - area_left*u_star[i]) / cell_volume
        rate[4, i] += (alpha_g + k)*div_area_u

    return rate
