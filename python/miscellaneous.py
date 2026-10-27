"""Conservation diagnostics computed directly from conservative cell averages."""

import numpy as np


def conservation_check(q_initial, q_final, domain):
    """Print total and per-phase changes; return mixture mass and energy change [%].

    Ghosts are excluded. A zero initial total has undefined percentage change
    (nan); the corresponding absolute change is still reported.
    Open boundaries can legitimately exchange mass and energy.
    """
    physical = domain["physical_ind"]
    volumes = domain["volumes"][physical]

    gas_mass_i = np.sum(volumes*q_initial[0, physical])
    gas_mass_f = np.sum(volumes*q_final[0, physical])
    liquid_mass_i = np.sum(volumes*q_initial[1, physical])
    liquid_mass_f = np.sum(volumes*q_final[1, physical])
    total_mass_i = gas_mass_i + liquid_mass_i
    total_mass_f = gas_mass_f + liquid_mass_f
    total_energy_i = np.sum(volumes*q_initial[3, physical])
    total_energy_f = np.sum(volumes*q_final[3, physical])

    initial = np.array([gas_mass_i, liquid_mass_i, total_mass_i, total_energy_i])
    final = np.array([gas_mass_f, liquid_mass_f, total_mass_f, total_energy_f])
    changes = final - initial
    percentages = np.full(4, np.nan)
    np.divide(100*changes, initial, out=percentages, where=initial != 0)

    for name, unit, change, percentage in zip(["Gas mass", "Liquid mass", "Total mass", "Total energy"], ["kg", "kg", "kg", "J"], changes, percentages):
        print(f"{name} change: {change:.6e} {unit} ({percentage:.6e} %)")

    return percentages[2], percentages[3]

# Used a definition similar to one I used in my bachelor thesis, will explain it later
def steady_state(q_previous, q_current, domain, q_reference, time_step, tolerance=1e-12, reference_time=1.0):
    """Check the scaled rate of change between two states.

    q_previous, q_current: full conservative arrays, including ghosts.
    q_reference: fixed reference state, normally q_initial.
    time_step: elapsed time between the two compared states [s].
    reference_time: fixed normalization time [s].
    tolerance: threshold for every component's volume-weighted RMS error.

    Reference components must have nonzero scales.
    Returns: is_steady, maximum error, array of individual component errors.
    """
    physical = domain["physical_ind"]
    volumes = domain["volumes"][physical]
    total_volume = np.sum(volumes)

    component_errors = np.zeros(q_current.shape[0])

    for k in range(q_current.shape[0]):
        # Use one fixed scale for this component across all physical cells.
        scale = np.max(np.abs(q_reference[k, physical]))

        if scale == 0:
            raise ValueError(f"Component {k} has a zero reference scale; choose a nonzero reference scale.")

        # Change in each physical cell between the two states.
        difference = q_current[k, physical] - q_previous[k, physical]

        # Normalize the rate of change to obtain dimensionless values.
        scaled_change = reference_time*difference / (time_step*scale)

        # Volume-weighted RMS: positive and negative changes cannot cancel.
        component_errors[k] = np.sqrt(np.sum(volumes*scaled_change**2) / total_volume)

    error = np.max(component_errors)
    is_steady = bool(np.isfinite(error) and error < tolerance)

    return is_steady, error, component_errors