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

def g3_nozzle_area(x):  
    """Cross-sectional area for the G3 nozzle."""
    Ain, Aout, At = 1.0, 0.6, 0.1
    xL, xR, xt = 0.0, 10.0, 5.0
    if x <= xt:
        return At + (Ain - At) * ((x - xt) / (xL - xt))**2
    else:
        return At + (Aout - At) * ((x - xt) / (xR - xt))**2
