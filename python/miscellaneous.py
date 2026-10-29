"""Conservation diagnostics computed directly from conservative cell averages."""

import numpy as np
from pathlib import Path


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

def open_conservation_check(q_initial, q_final, domain, left_transport, right_transport, results_folder=None):
    """Check gas mass, liquid mass and total energy in an open domain.

    Transport arrays contain integrated numerical boundary fluxes:
    [gas mass, liquid mass, total energy], in [kg, kg, J].
    Both boundaries use signed transport in the positive x direction.
    """
    physical = domain["physical_ind"]
    volumes = domain["volumes"][physical]
    components = [0, 1, 3]

    initial = np.array([np.sum(volumes*q_initial[k, physical]) for k in components])
    final = np.array([np.sum(volumes*q_final[k, physical]) for k in components])

    inventory_change = final - initial
    net_transport = left_transport - right_transport
    errors = inventory_change - net_transport

    # Relative errors use the initial inventory as the reference.
    relative_errors = np.full(3, np.nan)
    np.divide(errors, np.abs(initial), out=relative_errors, where=initial != 0)


    names = ["Gas mass", "Liquid mass", "Total energy"]
    units = ["kg", "kg", "J"]

    lines = ["Open-domain conservation balance:"]

    for k in range(3):
        lines.append(f"{names[k]} [{units[k]}]")
        lines.append(f"  Left transport:   {left_transport[k]:.6e}")
        lines.append(f"  Right transport:  {right_transport[k]:.6e}")
        lines.append(f"  Inventory change: {inventory_change[k]:.6e}")
        lines.append(f"  Balance error:    {errors[k]:.6e}")
        lines.append(f"  Relative error:   {relative_errors[k]:.6e}")

    lines.append(f"Mixture mass balance error: {errors[0] + errors[1]:.6e} kg")

    report = "\n".join(lines)
    print("\n" + report)

    if results_folder is not None:
        report_file = Path(results_folder) / "open_conservation.txt"
        report_file.write_text(report + "\n", encoding="utf-8")

    return {
        "initial": initial,
        "final": final,
        "left_transport": left_transport.copy(),
        "right_transport": right_transport.copy(),
        "error": errors,
        "relative_error": relative_errors
    }