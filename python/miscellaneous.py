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
