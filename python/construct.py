"""Domain construction and cell-volume integration."""

import numpy as np


def volume_calculator(area_rule, x_left, x_right, n=10000):
    """Integrate positive area [m²] over x_left < x_right using integer n > 0."""
    h = (x_right - x_left) / n

    total = 0.5*(area_rule(x_left) + area_rule(x_right))

    for i in range(1, n):
        x = x_left + i*h
        total += area_rule(x)

    return h*total


def build_domain(m, node_density_rule, area_rule, n_cells, n_ghost):
    """
    get a one-dimensional finite-volume mesh

    m                  : [x_left, x_right], physical borders [m]
    node_density_rule  : function giving mesh node_density at x
    area_rule          : function giving area at x [m²]
    n_cells                  : number of physical cells
    n_ghost            : int -> same number of ghost cells on both sides (0 = none)
                         (left, right) -> one count per side, e.g. (0, 1) = right side only

    Use m[0] < m[1], positive integer n_cells, positive node density and area.
    n_ghost is a nonnegative integer or pair. The solvers require >= 1 each side;
    reflective ghost counts must not exceed the number of physical cells.
    Returns a dictionary
    """
    if np.isscalar(n_ghost): n_ghost_left = n_ghost_right = int(n_ghost)
    else: n_ghost_left, n_ghost_right = int(n_ghost[0]), int(n_ghost[1])

    x_left, x_right = m[0], m[1]
    
    # fine integration grid
    x_trap = np.linspace(x_left, x_right, max(100001, 1000*n_cells + 1))
    node_density = np.array([node_density_rule(x) for x in x_trap])

    # cumulative integral using the trapezoidal rule
    traps = (0.5*(node_density[:-1] + node_density[1:])*np.diff(x_trap))
    cumulative = np.zeros(len(x_trap))
    cumulative[1:] = np.cumsum(traps)

    # normalization
    cumulative = cumulative / cumulative[-1]
    
    # F(x) inversion
    actual_nodes = np.linspace(0, 1, n_cells + 1)
    physical_faces = np.interp(actual_nodes, cumulative, x_trap)
    
    # ghost cells (empty arrays if a side has none)
    dx_left = physical_faces[1] - physical_faces[0]
    dx_right = physical_faces[-1] - physical_faces[-2]
    left_ghost_faces = x_left - dx_left*np.arange(n_ghost_left, 0, -1)
    right_ghost_faces = x_right + dx_right*np.arange(1, n_ghost_right + 1)

    x_faces = np.concatenate((left_ghost_faces, physical_faces, right_ghost_faces))

    # geometry
    n_total = n_cells + n_ghost_left + n_ghost_right
    dx = np.diff(x_faces)
    x_centres = 0.5*(x_faces[:-1] + x_faces[1:])

    # areas
    area_faces = np.array([area_rule(x) for x in x_faces])
    area_centres = np.array([area_rule(x) for x in x_centres])

    volumes = np.zeros(n_total)
    
    for j in range(n_total):
        volumes[j] = volume_calculator(area_rule, x_faces[j], x_faces[j + 1])
    
    area_average = volumes / dx
    
    # indices of cells
    total_ind = np.arange(n_total)
    physical_ind = np.arange(n_ghost_left, n_ghost_left + n_cells)
    ghost_left_ind = np.arange(n_ghost_left)
    ghost_right_ind = np.arange(n_ghost_left + n_cells, n_total)

    return {
        "m": np.array([x_left, x_right]),
        "n_cells": n_cells, "n_ghost_left": n_ghost_left, "n_ghost_right": n_ghost_right, "n_total": n_total,
        "x_faces": x_faces, "x_centres": x_centres, "dx": dx,
        "area_faces": area_faces, "area_left": area_faces[:-1], "area_right": area_faces[1:], "area_centres": area_centres, "area_average": area_average,
        "volumes": volumes,
        "total_ind": total_ind, "physical_ind": physical_ind, "ghost_left_ind": ghost_left_ind, "ghost_right_ind": ghost_right_ind
    }
