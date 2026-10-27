"""Spatial-residual monitoring, independent of the nonlinear stopping test."""

import numpy as np
from nonlinear_solvers import reference_state
from solver_functions import compute_residual, get_primitives, validate_state


def initialize_steady_history(history, q, domain, options):
    """Prepare monitoring only when steady_options is supplied to an integrator.

    options: residual_tol, mass_tol > 0; consecutive positive integer; min_time >= 0.
    Optional scale has five positive conservative scales, reference_time is in s,
    and mass_reference is a positive flow scale [kg/s]. They remain fixed during
    a pressure stage so that a changing normalisation cannot hide stagnation.
    Optional hold_time >= 0 requires a continuous settled interval before stopping.
    """
    if options is None:
        return
    physical = domain["physical_ind"]
    scale = options.get("scale")
    if scale is None:
        scale = reference_state(q, domain)
    reference_time = options.get("reference_time")
    if reference_time is None:
        c = get_primitives(q)["c"][physical]
        reference_time = (domain["m"][1]-domain["m"][0])/np.max(c)
    history["residuals"] = {name: [] for name in ("times", "values", "norm", "max_norm", "component_norms", "mass_flow", "mass_spread")}
    history["residuals"].update(scale=np.asarray(scale), reference_time=reference_time, converged=False, consecutive_count=0, steady_since=np.nan)


def save_steady_residual(history, q, domain, time, options, reconstruction, pressure_fallback):
    """Save R(Q) and a dimensionless residual at the same times as primitives.

    R is the SEMIDISCRETE right-hand side, not the Newton/FPI update or its error.
    Each component is scaled by a fixed Q scale and a fixed reference time.
    norm = max(component volume-weighted RMS); max_norm also exposes local peaks.
    Require small norm AND nearly uniform numerical mass flux for several saved
    times after min_time. Initial snapshot never counts as a steady confirmation.
    """
    if options is None:
        return False
    validate_state(q, domain)
    saved = history["residuals"]
    rate, flux, u_star = compute_residual(q, domain, reconstruction, pressure_fallback, return_flux=True)
    physical = domain["physical_ind"]
    scaled = saved["reference_time"]*rate/saved["scale"][:, None]
    weights = domain["volumes"][physical]/np.sum(domain["volumes"][physical])
    component_norms = np.sqrt(np.sum(scaled**2*weights[None, :], axis=1))
    norm = np.max(component_norms)
    first = physical[0]
    areas = domain["area_faces"][first:first+len(physical)+1]
    mass_flow = areas*(flux[0]+flux[1])
    denominator = max(abs(np.mean(mass_flow)), 1e-8*options.get("mass_reference", 1.0))
    mass_spread = np.ptp(mass_flow)/denominator
    if not np.isfinite(norm) or not np.all(np.isfinite(mass_flow)):
        raise RuntimeError("Non-finite steady-state residual or numerical mass flux.")

    values = {"times": time, "values": rate.copy(), "norm": norm, "max_norm": np.max(np.abs(scaled)), "component_norms": component_norms, "mass_flow": mass_flow, "mass_spread": mass_spread}
    for name, value in values.items():
        saved[name].append(value)
    good = time > 0 and time >= options.get("min_time", 0.0) and norm < options["residual_tol"] and mass_spread < options["mass_tol"]
    if good and saved["consecutive_count"] == 0:
        saved["steady_since"] = time
    if not good:
        saved["steady_since"] = np.nan
    saved["consecutive_count"] = saved["consecutive_count"]+1 if good else 0
    held = good and time-saved["steady_since"] >= options.get("hold_time", 0.0)
    saved["converged"] = held and saved["consecutive_count"] >= options.get("consecutive", 3)
    if time > 0:
        print(f"    steady R={norm:.3e} | mass spread={mass_spread:.3e} | checks={saved['consecutive_count']}/{options.get('consecutive', 3)}", flush=True)
    return saved["converged"]


def finalize_steady_history(history):
    """Convert snapshot lists to arrays; values has shape (saved times, 5, N)."""
    if "residuals" in history:
        for name, value in history["residuals"].items():
            history["residuals"][name] = np.asarray(value)
