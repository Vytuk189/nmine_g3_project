import numpy as np

from solver_functions import get_primitives


def conservation_check(q_initial, q_final, domain):
    physical_cell_ind = domain["physical_ind"]
    cell_volumes = domain["volumes"][physical_cell_ind]

    primitives_i = get_primitives(q_initial)
    primitives_f = get_primitives(q_final)

    cell_densities_i = primitives_i["rho"][physical_cell_ind]
    cell_tot_energies_i = primitives_i["e_total"][physical_cell_ind]
    cell_densities_f = primitives_f["rho"][physical_cell_ind]
    cell_tot_energies_f = primitives_f["e_total"][physical_cell_ind]

    total_mass_i = np.sum(cell_volumes*cell_densities_i)
    total_mass_f = np.sum(cell_volumes*cell_densities_f)
    total_e_i = np.sum(cell_volumes*cell_densities_i*cell_tot_energies_i)
    total_e_f = np.sum(cell_volumes*cell_densities_f*cell_tot_energies_f)

    mass_error = 100*(total_mass_f - total_mass_i)/total_mass_i
    energy_error = 100*(total_e_f - total_e_i)/total_e_i

    print(f"Mass conservation error: {mass_error:.6e} %")
    print(f"Energy conservation error: {energy_error:.6e} %")

    return mass_error, energy_error
