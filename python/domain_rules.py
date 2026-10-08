"""Edit or add cross-sectional area and node-density rules here."""

import numpy as np


def uniform_node_density(x):
    return 1.0


def gaussian_node_density(x):
    return np.exp(-((x - 5)/3)**2)


def uniform_area(x):
    return 1


def nozzle_area(x):
    return 1 - 0.9*np.exp(-((x - 5)/3)**2)


def linear_area(x):
    return 1.0 + 0.1*x
